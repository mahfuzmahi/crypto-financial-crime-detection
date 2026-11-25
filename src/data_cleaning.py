import pandas as pd
import numpy as np

from src.config import RAW_DATA_DIR, PROCESSED_DATA_DIR
from src.exploratory_data_analysis import load_data

def check_data_quality(edges):
    # load original files
    features = pd.read_csv(RAW_DATA_DIR / 'elliptic_txs_features.csv', header=None)
    classes = pd.read_csv(RAW_DATA_DIR / 'elliptic_txs_classes.csv')

    # check the cols
    print("dataset columns:")
    print("feature columns:", features.shape[1])
    print("class columns:", classes.shape[1])
    print("edge columns:", edges.shape[1])

    if features.shape[1] != 167 or classes.shape[1] != 2 or edges.shape[1] != 2:
        raise ValueError("incorrect number of columns")

    if not {'txId', 'class'}.issubset(classes.columns):
        raise ValueError("incorrect class columns")

    if not {'txId1', 'txId2'}.issubset(edges.columns):
        raise ValueError("incorrect edge columns")

    # check missing values
    print("missing values:")
    missing_features = features.isna().sum().sum()
    missing_classes = classes.isna().sum().sum()
    missing_edges = edges.isna().sum().sum()
    print("missing feature values:", missing_features)
    print("missing class values:", missing_classes)
    print("missing edge values:", missing_edges)

    # check for dupes
    print("duplicate records:")
    dup_features = features.duplicated().sum()
    dup_classes = classes.duplicated().sum()
    dup_feature_ids = features[0].duplicated().sum()
    dup_class_ids = classes['txId'].duplicated().sum()
    print("duplicate feature rows:", dup_features)
    print("duplicate class rows:", dup_classes)
    print("duplicate feature ids:", dup_feature_ids)
    print("duplicate class ids:", dup_class_ids)

    # check for invalid numbers
    print("data types:")
    nums = features.iloc[:, 1:].apply(pd.to_numeric, errors='coerce')
    bad_numbers = nums.isna().sum().sum() - features.iloc[:, 1:].isna().sum().sum()
    print("invalid numerical values:", bad_numbers)

    results = {
        'missing_feature_values': int(missing_features),
        'missing_class_values': int(missing_classes),
        'missing_edge_values': int(missing_edges),
        'duplicate_feature_rows': int(dup_features),
        'duplicate_class_rows': int(dup_classes),
        'duplicate_feature_ids': int(dup_feature_ids),
        'duplicate_class_ids': int(dup_class_ids),
        'invalid_numerical_values': int(bad_numbers)
    }
    return features[0], classes['txId'], results

def check_files(ids, class_ids):
    # check if the transaction ids match
    missing_classes = (~ids.isin(class_ids)).sum()
    missing_features = (~class_ids.isin(ids)).sum()
    print("missing class records:", missing_classes)
    print("missing feature records:", missing_features)
    return int(missing_classes), int(missing_features)

def check_connections(edges, ids):
    # check the graph connections
    bad_edges = (~edges['txId1'].isin(ids) | ~edges['txId2'].isin(ids)).sum()

    # check duplicates and self-loops
    duplicates = edges.duplicated().sum()
    self_loops = (edges['txId1'] == edges['txId2']).sum()
    print("invalid connections:", bad_edges)
    print("duplicate connections:", duplicates)
    print("self-loop connections:", self_loops)
    return int(bad_edges), int(duplicates), int(self_loops)

def check_values(data, cols):
    # check for infinite values
    values = data[cols].to_numpy(dtype=float)
    infinite = np.isinf(values).sum()

    # check the time steps
    bad_steps = (~data['time_step'].between(1, 49) | (data['time_step'] % 1 != 0)).sum()

    # check the class labels
    bad_labels = data['class_name'].isna().sum()
    print("infinite values:", infinite)
    print("invalid time steps:", bad_steps)
    print("invalid class labels:", bad_labels)
    return int(infinite), int(bad_steps), int(bad_labels)

def save_results(edges, results):
    # save the cleaned files
    folder = PROCESSED_DATA_DIR / 'cleaning'
    folder.mkdir(parents=True, exist_ok=True)
    edges = edges.drop_duplicates()
    edges.to_csv(folder / 'cleaned_edges.csv', index=False)
    pd.Series(results, name='count').to_csv(folder / 'cleaning_checks.csv')
    print("files saved to:", folder)

def main():
    # load the dataset
    data, _, edges, cols = load_data()

    # check the original files
    ids, class_ids, quality = check_data_quality(edges)

    print("dataset files:")
    missing_classes, missing_features = check_files(ids, class_ids)

    print("graph connections:")
    bad_edges, duplicates, self_loops = check_connections(edges, ids)

    print("numerical values:")
    infinite, bad_steps, bad_labels = check_values(data, cols)

    # check for any problems
    if quality['missing_feature_values'] > 0 or quality['missing_class_values'] > 0:
        raise ValueError("missing values found")

    if quality['missing_edge_values'] > 0 or quality['invalid_numerical_values'] > 0:
        raise ValueError("invalid or missing values found")

    if quality['duplicate_feature_ids'] > 0 or quality['duplicate_class_ids'] > 0:
        raise ValueError("duplicate transaction records found")

    if missing_classes > 0 or missing_features > 0 or bad_edges > 0:
        raise ValueError("transaction ids do not match")

    if infinite > 0 or bad_steps > 0 or bad_labels > 0:
        raise ValueError("invalid values found")

    results = {
        'missing_class_records': missing_classes,
        'missing_feature_records': missing_features,
        'invalid_connections': bad_edges,
        'duplicate_connections_removed': duplicates,
        'self_loops_retained': self_loops,
        'infinite_values': infinite,
        'invalid_time_steps': bad_steps,
        'invalid_labels': bad_labels
    }
    results.update(quality)
    save_results(edges, results)
    print("data cleaning checks completed")

if __name__ == "__main__":
    main()