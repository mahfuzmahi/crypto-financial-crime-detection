import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, average_precision_score

from src.config import PROCESSED_DATA_DIR, SEED

def load_data(folder, name):
    # load the training and validation data
    train = pd.read_csv(folder / f'train_{name}.csv')
    validation = pd.read_csv(folder / f'validation_{name}.csv')
    cols = [col for col in train.columns if col.startswith('feat_') or col.startswith('pc_')]
    return train, validation, cols

def check_classes(train, validation):
    # count the transactions in each class
    print("training transactions:", len(train))
    print("training licit transactions:", (train['label'] == 0).sum())
    print("training illicit transactions:", (train['label'] == 1).sum())
    print("validation transactions:", len(validation))
    print("validation licit transactions:", (validation['label'] == 0).sum())
    print("validation illicit transactions:", (validation['label'] == 1).sum())
    print("training illicit percentage:", round(train['label'].mean() * 100, 4))
    print("validation illicit percentage:", round(validation['label'].mean() * 100, 4))

def evaluate_model(train, validation, cols, name, model_name):
    # train the model
    if model_name == 'Logistic Regression':
        model = LogisticRegression(solver='liblinear', class_weight='balanced', max_iter=1000, random_state=42)
    else:
        model = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=SEED, n_jobs=-1)

    model.fit(train[cols], train['label'])

    # predict the class labels
    probabilities = model.predict_proba(validation[cols])[:, 1]
    predictions = (probabilities >= 0.5).astype(int)

    # evaluate the illicit class
    precision = precision_score(validation['label'], predictions, zero_division=0)
    recall = recall_score(validation['label'], predictions, zero_division=0)
    f1 = f1_score(validation['label'], predictions, zero_division=0)

    # calculate pr-auc
    pr_auc = average_precision_score(validation['label'], probabilities)

    return {
        'model': model_name,
        'method': name,
        'features': len(cols),
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'pr_auc': pr_auc
    }

def save_results(results):
    # save the comparison table
    folder = PROCESSED_DATA_DIR / 'aligned' / 'evaluation'
    folder.mkdir(parents=True, exist_ok=True)
    results = pd.DataFrame(results)
    results.to_csv(folder / 'evaluation_summary.csv', index=False)
    print("model evaluation summary:")
    print(results.round(4).to_string(index=False))
    print("files saved to:", folder)

def main():
    # load the original features
    folder = PROCESSED_DATA_DIR / 'aligned'
    train, validation, cols = load_data(folder / 'transformation', 'scaled')
    check_classes(train, validation)

    # evaluate all feature sets
    results = []
    for model_name in ['Logistic Regression', 'Random Forest']:
        train, validation, cols = load_data(folder / 'transformation', 'scaled')
        results.append(evaluate_model(train, validation, cols, 'All Features', model_name))
        train, validation, cols = load_data(folder / 'selection', 'filter')
        results.append(evaluate_model(train, validation, cols, 'Filter', model_name))
        train, validation, cols = load_data(folder / 'selection', 'wrapper')
        results.append(evaluate_model(train, validation, cols, 'Wrapper', model_name))
        train, validation, cols = load_data(folder / 'selection', 'embedded')
        results.append(evaluate_model(train, validation, cols, 'Embedded', model_name))
        train, validation, cols = load_data(folder / 'extraction', 'pca')
        results.append(evaluate_model(train, validation, cols, 'PCA', model_name))

    save_results(results)
    print("feature evaluation completed")

if __name__ == "__main__":
    main()