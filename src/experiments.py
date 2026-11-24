#Experiment: Does adding GraphSAGE embeddings to Random Forest help?
#Trains GraphSAGE, extracts 64d node embeddings, then fits Random Forest
import argparse
import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import RandomForestClassifier

from src.config import RESULTS_DIR, SEED
from src.graph_sage import main as train_graphsage, parse_args as sage_args
from src.baseline_models import evaluate

def rf_with_embeddings(feature_set, seed):
    #1. Train GraphSage
    model, data = train_graphsage(sage_args(['--features', feature_set, '--seed', str(seed),
                                             '--tag', 'rf_emb']))

    #2. Embeddings from last hidden layer
    model.eval()
    with torch.no_grad():
        emb = model.embed(data.x, data.edge_index).cpu().numpy()

    #quantile transformed featueres
    X = data.x.cpu().numpy()
    y = data.y.cpu().numpy()

    train, test = data.train_mask.cpu().numpy(), data.test_mask.cpu().numpy()

    variants = {
        'RF raw features': X,
        'RF raw + embeddings': np.hstack([X, emb]),
        'RF embeddings only': emb
    }

    rows = []
    for name, features in variants.items():
        rf = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=SEED, n_jobs=-1)
        rf.fit(features[train], y[train])
        row = evaluate(name, feature_set, 'test_1_29', y[test], rf.predict_proba(features[test])[:, 1])
        row.update(seed=seed, n_features=features.shape[1])
        rows.append(row)

    #4. Save results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / f'experiment_rf_embeddings_{feature_set}_s{seed}.csv'
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"\nSaved to {out}")

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--features', choices=['local', 'all'], default='all')
    p.add_argument('--seed', type=int, default=1)
    args = p.parse_args()
    rf_with_embeddings(args.features, args.seed)
