#Builds Elliptic Transaction graph as a PyTorch Geometric Data object for GraphSAGE
#Nodes = Transactions, Edges = Bitcoin flows
import argparse
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import QuantileTransformer, StandardScaler
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected

from src.config import RAW_DATA_DIR, PROCESSED_DATA_DIR, SEED
#Temporal split: train steps 1-29, validate 30-34, test 35-49
TRAIN_END, VAL_END = 29, 34

FEATURE_SETS = {
    #93 local features (transaction's own info)
    'local': [f'feat_{i}' for i in range(1, 94)],
    #93 local + 72 aggregated features (summaries of node neighbors)
    'all': [f'feat_{i}' for i in range(1, 166)]
}

#Col 0 = txId, col1 = time step, cols 2-166 = anonymised features
def build_graph(feature_set='all', scaler_name='quantile'):
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    feats = pd.read_csv(RAW_DATA_DIR / 'elliptic_txs_features.csv', header=None)
    feats.columns = ['txId', 'time_step'] + [f'feat_{i}' for i in range(1, 166)]
    classes = pd.read_csv(RAW_DATA_DIR / 'elliptic_txs_classes.csv')
    edges = pd.read_csv(RAW_DATA_DIR / 'elliptic_txs_edgelist.csv')

    #left merge keeps all transactions, including unlabeled
    df = feats.merge(classes, on='txId', how='left')
    #1 = illicit, 0 = licit, -1 = unknown (kept in graph for message passing, excluded from loss)
    df['label'] = df['class'].map({'1': 1, '2':0}).fillna(-1).astype(int)

    #txId -> node index
    id2idx = pd.Series(np.arange(len(df)), index=df['txId'])
    src = id2idx.loc[edges['txId1']].to_numpy()
    dst = id2idx.loc[edges['txId2']].to_numpy()
    #undirected so info flows both ways along transactions
    edge_index = to_undirected(torch.tensor(np.vstack([src, dst]), dtype=torch.long))

    #masks select labelled nodes per time window, no test leakage
    ts = df['time_step'].to_numpy()
    labeled = df['label'].to_numpy() >= 0
    train_mask = labeled & (ts <= TRAIN_END)
    val_mask = labeled & (ts > TRAIN_END) & (ts <= VAL_END)
    test_mask = labeled & (ts > VAL_END)

    cols = FEATURE_SETS[feature_set]
    if scaler_name == 'quantile':
        scaler = QuantileTransformer(output_distribution='normal', n_quantiles=1000,
                                         random_state=SEED)
    else:
        scaler = StandardScaler()
    #Quantile -> normal transform, fitted on training period
    scaler.fit(df.loc[ts <= TRAIN_END, cols])
    #Rank-based scaling removes outliers and is robust to feature drift
    x = scaler.transform(df[cols]).astype(np.float32)

    data = Data(
        x=torch.from_numpy(x),
        edge_index=edge_index,
        y=torch.tensor(df['label'].to_numpy(), dtype=torch.long),
        #per timestep evaluation
        time_step=torch.tensor(ts, dtype=torch.long),
        train_mask=torch.from_numpy(train_mask),
        val_mask=torch.from_numpy(val_mask),
        test_mask=torch.from_numpy(test_mask),
    )

    #one file per feature set
    suffix = '' if scaler_name == 'quantile' else f'_{scaler_name}'
    out = PROCESSED_DATA_DIR / f'elliptic_graph_{feature_set}{suffix}.pt'
    torch.save(data, out)
    print(data)
    print(f"train/val/test labeled nodes: {train_mask.sum():,} / {val_mask.sum():,} / {test_mask.sum():,}")
    print(f"illicit rate train/val/test: "
          f"{df.loc[train_mask, 'label'].mean():.2%} / {df.loc[val_mask, 'label'].mean():.2%} / {df.loc[test_mask, 'label'].mean():.2%}")
    print(f"saved in {out}")
    return data

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--features', choices=FEATURE_SETS, default='all')
    p.add_argument('--scaler', choices=['quantile', 'standard'], default='quantile')
    args = p.parse_args()
    build_graph(args.features, args.scaler)
