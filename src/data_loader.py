import pandas as pd
import numpy as np
import os

RAW_DATA_DIR = 'data/raw'
PROCESSED_DATA_DIR = 'data/processed'

def load_and_split_data():
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    
    print("Loading raw CSV files from data/raw/...")
    classes_df = pd.read_csv(os.path.join(RAW_DATA_DIR, 'elliptic_txs_classes.csv'))
    features_df = pd.read_csv(os.path.join(RAW_DATA_DIR, 'elliptic_txs_features.csv'), header=None)
    
    # 167 total columns: txId (col 0), time_step (col 1), feat_1 to feat_165 (cols 2-166)
    features_df.columns = ['txId', 'time_step'] + [f'feat_{i}' for i in range(1, 166)]
    
    # Merge classes and features
    data = pd.merge(features_df, classes_df, on='txId')
    
    # Map classes: '1' -> Illicit (1), '2' -> Licit (0), 'unknown' -> NaN
    data['label'] = data['class'].map({'1': 1, '2': 0, 'unknown': np.nan})
    
    # Filter for labeled data
    labeled_data = data.dropna(subset=['label']).copy()
    labeled_data['label'] = labeled_data['label'].astype(int)
    
    # Temporal Train / Test Split
    train_df = labeled_data[labeled_data['time_step'] <= 34]
    test_df = labeled_data[labeled_data['time_step'] > 34]
    
    # Export split files
    train_df.to_csv(os.path.join(PROCESSED_DATA_DIR, 'train_split.csv'), index=False)
    test_df.to_csv(os.path.join(PROCESSED_DATA_DIR, 'test_split.csv'), index=False)
    
    print("\nSplits successfully generated and saved to data/processed/")
    print(f"Train samples (Time 1-34): {len(train_df):,} | Illicit ratio: {train_df['label'].mean():.2%}")
    print(f"Test samples (Time 35-49): {len(test_df):,} | Illicit ratio: {test_df['label'].mean():.2%}")

if __name__ == "__main__":
    load_and_split_data()