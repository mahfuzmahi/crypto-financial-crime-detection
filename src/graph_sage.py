import argparse
import random
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch_geometric.nn import SAGEConv
from sklearn.metrics import precision_score, recall_score, f1_score, average_precision_score, confusion_matrix, precision_recall_curve

from src.config import PROCESSED_DATA_DIR, RESULTS_DIR, FIGURES_DIR, SEED

#Regularizing: Weight decay, dropout, early stopping
HPARAMS = dict(hidden=128, hidden2=64, dropout=0.5, lr=0.01, weight_decay=5e-4, 
               epochs=300, patience=30)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)

class GraphSAGE(nn.Module):
    def __init__(self, in_dim, h1, h2, dropout):
        super().__init__()
        # mean aggregation, each node combines own features with average of neighbors
        self.conv1, self.bn1 = SAGEConv(in_dim, h1, aggr='mean'), nn.BatchNorm1d(h1)
        self.conv2, self.bn2 = SAGEConv(h1, h2, aggr='mean'), nn.BatchNorm1d(h2)
        #2 outputs, licit / illicit
        self.head = nn.Linear(h2, 2)
        self.dropout = dropout

    #Per layer: SAGEConv into BatchNorm into ReLU into Dropout
    def forward(self, x, edge_index):
        x = F.dropout(F.relu(self.bn1(self.conv1(x, edge_index))), self.dropout, self.training)
        x = F.dropout(F.relu(self.bn2(self.conv2(x, edge_index))), self.dropout, self.training)
        return self.head(x)
    
#Threshold based metrics for the illicit class, PR-AUC is threshold-free
def metrics(y_true, prob, thr=0.5):
    pred = (prob >= thr).astype(int)
    return dict(precision=precision_score(y_true, pred, zero_division=0),
                recall=recall_score(y_true, pred, zero_division=0),
                f1=f1_score(y_true, pred, zero_division=0),
                pr_auc=average_precision_score(y_true, prob))

def pick_threshold(y_val, p_val):
    P, R, T = precision_recall_curve(y_val, p_val)
    #Drop the last P/R point which has no threshold
    f1 = 2 * P[:-1] * R[:-1] / (P[:-1] + R[:-1] + 1e-12)
    return float(T[f1.argmax()])

#No gradients during evaluation
@torch.no_grad()
def evaluate(model, data, mask, loss_fn):
    #Eval mode turns off dropout, uses BatchNorm running stats
    model.eval()
    out = model(data.x, data.edge_index)
    loss = loss_fn(out[mask], data.y[mask]).item()
    prob = out[mask].softmax(-1)[:, 1].cpu().numpy()
    return loss, prob, data.y[mask].cpu().numpy()

def main(feature_set, seed):
    set_seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #weights_only=False needed to load PyG Data object
    data = torch.load(PROCESSED_DATA_DIR / f'elliptic_graph_{feature_set}.pt', weights_only=False).to(device)

    #inverse-frequency class weights, illicit is upweighted
    counts = torch.bincount(data.y[data.train_mask], minlength=2).float()
    loss_fn = nn.CrossEntropyLoss(weight=(counts.sum() / (2 * counts)).to(device))
    h = HPARAMS
    model = GraphSAGE(data.num_features, h['hidden'], h['hidden2'], h['dropout']).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=h['lr'], weight_decay=h['weight_decay'])
    history, best_auc, best_state, best_epoch, wait = [], -1, None, 0, 0
    t0 = time.time()
    for epoch in range(1, h['epochs'] + 1):
        model.train()
        opt.zero_grad()
        #full batch, whole graph each step, loss uses training nodes only
        out = model(data.x, data.edge_index)
        loss = loss_fn(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        opt.step()

        val_loss, val_prob, val_y = evaluate(model, data, data.val_mask, loss_fn)
        m = metrics(val_y, val_prob)
        history.append(dict(epoch=epoch, train_loss=loss.item(), val_loss=val_loss,
                            val_f1=m['f1'], val_pr_auc=m['pr_auc']))
        if epoch % 10 == 0:
            print(f"ep {epoch: 3d} | train {loss.item():.4f} | val {val_loss:.4f} | F1 {m['f1']:.4f} | PR-AUC {m['pr_auc']:.4f}")

        #early stopping on val PR-AUC, keep best weights
        if m['pr_auc'] > best_auc:
            best_auc, best_epoch, wait = m['pr_auc'], epoch, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= h['patience']:
                print(f"Early stop at epoch {epoch}, best epoch {best_epoch} (val PR-AUC {best_auc:.4f})")
                break
    train_time = time.time() - t0

    #Restore best epoch, pick threshold on val, eval test once
    model.load_state_dict(best_state)
    _, val_prob, val_y = evaluate(model, data, data.val_mask, loss_fn)
    thr = pick_threshold(val_y, val_prob)
    _, test_prob, test_y = evaluate(model, data, data.test_mask, loss_fn)
    test_m = metrics(test_y, test_prob, thr)
    print(f"Threshold (val-F1-Optimal): {thr:.3f}")
    print(f"\nTEST ({feature_set}) " + " | ".join(f"{k} {v:.4f}" for k, v in test_m.items()))
    print("Confusion Matrix [TN FP / FN TP]:\n", confusion_matrix(test_y, (test_prob >= thr).astype(int)))
    print(f"Device {device} | Train Time {train_time:.1f}s | Stopped Epoch {epoch} | Best Epoch {best_epoch}")

    RESULTS_DIR.mkdir(exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    hist = pd.DataFrame(history)
    hist.to_csv(RESULTS_DIR / f'graphsage_history_{feature_set}_s{seed}.csv', index=False)
    #one row per run with metrics, threshold, seed, and hyperparams
    pd.DataFrame([dict(model='GraphSAGE', features=feature_set, **test_m, threshold=thr, seed=seed, best_epoch=best_epoch,
                       stopped_epoch=epoch, train_time_s=train_time, device=str(device), **h),]
                       ).to_csv(RESULTS_DIR / f'graphsage_metrics_{feature_set}_s{seed}.csv', index=False)

    # Learning curve: train vs val loss with best epoch marked
    plt.figure(figsize=(7, 4))
    plt.plot(hist.epoch, hist.train_loss, label = 'train loss')
    plt.plot(hist.epoch, hist.val_loss, label = 'val loss')
    plt.axvline(best_epoch, ls='--', c='grey', label=f'Best Epoch ({best_epoch})')
    plt.xlabel('Epoch')
    plt.ylabel('Weighted Cross-Entropy Loss')
    plt.title(f'GraphSAGE learning curve ({feature_set})')
    plt.legend()
    plt.savefig(FIGURES_DIR / f'12_GraphSAGE_Learning_Curve_{feature_set}_s{seed}.png', dpi=150, bbox_inches='tight')

    #test F1 per timestep, shows collapse after step 43
    ts = data.time_step[data.test_mask].cpu().numpy()
    pred = (test_prob >= thr).astype(int)
    steps = sorted(set(ts))
    f1s = [f1_score(test_y[ts == s], pred[ts == s], zero_division=0) for s in steps]
    plt.figure(figsize=(7, 4))
    plt.bar(steps, f1s)
    plt.xlabel("Time Step")
    plt.ylabel("Illicit F1")
    plt.title(f'GraphSAGE Test F1 per Time Step ({feature_set})')
    plt.savefig(FIGURES_DIR / f'13_GraphSAGE_F1_Per_Step_{feature_set}_s{seed}.png', dpi=150, bbox_inches='tight')

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--features', choices=['local', 'all'], default='all')
    p.add_argument('--seed', type=int, default=SEED)
    args = p.parse_args()
    main(args.features, args.seed)
