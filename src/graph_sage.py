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


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)

class GraphSAGE(nn.Module):
    def __init__(self, in_dim, hidden_dims, dropout, norm='batch'):
        super().__init__()
        dims = [in_dim] + hidden_dims
        # mean aggregation, each node combines own features with average of neighbors
        self.convs = nn.ModuleList(SAGEConv(a, b, aggr='mean') for a, b in zip(dims, dims[1:]))
        make_norm = {'batch': nn.BatchNorm1d, 'layer': nn.LayerNorm, 'none': nn.Identity}[norm]
        self.norms = nn.ModuleList(make_norm(d) for d in hidden_dims)
        self.head = nn.Linear(hidden_dims[-1], 2)
        self.dropout = dropout

    def embed(self, x, edge_index):
        for conv, norm in zip(self.convs, self.norms):
            x = F.dropout(F.relu(norm(conv(x, edge_index))), self.dropout, self.training)
        return x

    def forward(self, x, edge_index):
        return self.head(self.embed(x, edge_index))
    
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

def main(args):
    set_seed(args.seed)
    fs = args.features
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    #weights_only=False needed to load PyG Data object
    suffix = '' if args.scaler == 'quantile' else f'_{args.scaler}'
    data = torch.load(PROCESSED_DATA_DIR / f'elliptic_graph_{fs}{suffix}.pt', weights_only=False).to(device)

    #Experimenting with clipping data
    if args.clip is not None:
        data.x = data.x.clamp(-args.clip, args.clip)
    if args.no_edges:
        #no neighbors: every SAGECConv reduces to a per node linear layer
        data.edge_index = torch.empty(2, 0, dtype=torch.long, device=device)
    if args.no_class_weights:
        loss_fn = nn.CrossEntropyLoss()
    else:
        counts = torch.bincount(data.y[data.train_mask], minlength=2).float()
        loss_fn = nn.CrossEntropyLoss(weight=(counts.sum() / (2 * counts)).to(device))
    #hidden sizes halve per layer
    hidden_dims = [args.hidden // 2 ** i for i in range(args.layers)]
    model = GraphSAGE(data.num_features, hidden_dims, args.dropout, norm=args.norm).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)

    history, best_auc, best_state, best_epoch, wait = [], -1, None, 0, 0
    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
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
            if wait >= args.patience:
                print(f"Early stop at epoch {epoch}, best epoch {best_epoch} (val PR-AUC {best_auc:.4f})")
                break
    train_time = time.time() - t0

    #Restore best epoch, pick threshold on val, eval test once
    model.load_state_dict(best_state)
    _, val_prob, val_y = evaluate(model, data, data.val_mask, loss_fn)
    thr = pick_threshold(val_y, val_prob)
    _, test_prob, test_y = evaluate(model, data, data.test_mask, loss_fn)
    test_m = metrics(test_y, test_prob, thr)
    pred = (test_prob >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(test_y, pred).ravel()

    print(f"Threshold (val-F1-Optimal): {thr:.3f}")
    print(f"\nTEST ({fs}{' ' + args.tag if args.tag else ''}) " + " | ".join(f"{k} {v:.4f}" for k, v in test_m.items()))
    print(f"Confusion Matrix [TN FP / FN TP]:\n[[{tn} {fp}]\n [{fn} {tp}]]")
    print(f"Device {device} | Train Time {train_time:.1f}s | Stopped Epoch {epoch} | Best Epoch {best_epoch}")

    run = f"{fs}_s{args.seed}" + (f"_{args.tag}" if args.tag else "")
    RESULTS_DIR.mkdir(exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    hist = pd.DataFrame(history)
    hist.to_csv(RESULTS_DIR / f'graphsage_history_{run}.csv', index=False)
    #one row per run with metrics, threshold, seed, and hyperparams
    pd.DataFrame([dict(model='GraphSAGE', **test_m, threshold=thr, tn=tn, fp=fp, fn=fn,
                       tp=tp, best_epoch=best_epoch,stopped_epoch=epoch, 
                       train_time_s=train_time, device=str(device), **vars(args)),]
                       ).to_csv(RESULTS_DIR / f'graphsage_metrics_{run}.csv', index=False)

    # Learning curve: train vs val loss with best epoch marked
    plt.figure(figsize=(7, 4))
    plt.plot(hist.epoch, hist.train_loss, label = 'train loss')
    plt.plot(hist.epoch, hist.val_loss, label = 'val loss')
    plt.axvline(best_epoch, ls='--', c='grey', label=f'Best Epoch ({best_epoch})')
    plt.xlabel('Epoch')
    plt.ylabel('Weighted Cross-Entropy Loss' if not args.no_class_weights else 'Cross-Entropy Loss')
    plt.title(f'GraphSAGE learning curve ({run})')
    plt.legend()
    plt.savefig(FIGURES_DIR / f'12_GraphSAGE_Learning_Curve_{run}.png', dpi=150, bbox_inches='tight')

    #test F1 per timestep, shows collapse after step 43
    ts = data.time_step[data.test_mask].cpu().numpy()
    steps = sorted(set(ts))
    f1s = [f1_score(test_y[ts == s], pred[ts == s], zero_division=0) for s in steps]
    plt.figure(figsize=(7, 4))
    plt.bar(steps, f1s)
    plt.xlabel("Time Step")
    plt.ylabel("Illicit F1")
    plt.title(f'GraphSAGE Test F1 per Time Step ({run})')
    plt.savefig(FIGURES_DIR / f'13_GraphSAGE_F1_Per_Step_{run}.png', dpi=150, bbox_inches='tight')
    plt.close('all')

    return model, data

def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument('--features', choices=['local', 'all'], default='all')
    p.add_argument('--seed', type=int, default=SEED)

    #model and optimiser
    p.add_argument('--layers', type=int, default=2)
    p.add_argument('--hidden', type=int, default=128)
    p.add_argument('--dropout', type=float, default=0.5)
    p.add_argument('--norm', choices=['batch', 'layer', 'none'], default='batch')
    p.add_argument('--lr', type=float, default=0.01)
    p.add_argument('--wd', type=float, default=5e-4) #L2 weight decay
    p.add_argument('--epochs', type=int, default=300)
    p.add_argument('--patience', type=int, default=30) #set >= epochs to disable early stopping

    #experiments
    p.add_argument('--scaler', choices=['quantile', 'standard'], default='quantile')
    p.add_argument('--clip', type=float, default=None)
    p.add_argument('--no-class-weights', action='store_true')
    p.add_argument('--no-edges', action='store_true')
    p.add_argument('--tag', default='')
    return p.parse_args(argv)

if __name__ == '__main__':
    main(parse_args())
