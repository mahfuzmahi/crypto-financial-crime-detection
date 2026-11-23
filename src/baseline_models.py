import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, average_precision_score, confusion_matrix

from src.config import PROCESSED_DATA_DIR, RESULTS_DIR, SEED

def evaluate(name, feat_name, y_test, probs, thr=0.5):
    preds = (probs >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, preds).ravel()
    row = dict(model=name, features=feat_name, threshold=thr,
               precision=precision_score(y_test, preds),
               recall=recall_score(y_test, preds),
               f1=f1_score(y_test, preds),
               pr_auc=average_precision_score(y_test, probs),
               tn=tn, fp=fp, fn=fn, tp=tp)
    print(f"\n-----------------{name}")
    for k in ['precision', 'recall', 'f1', 'pr_auc']:
        print(f"{k:10s}: {row[k]:.4f}")
    print(f"Confusion Matrix [TN FP / FN TP]:\n[[{tn} {fp}]\n [{fn} {tp}]]")
    return row

def run_experiments():
    print("loading train & test datasets from data/processed/...")
    train_df = pd.read_csv(PROCESSED_DATA_DIR / 'train_split.csv')
    test_df = pd.read_csv(PROCESSED_DATA_DIR / 'test_split.csv')

    local_cols = [f'feat_{i}' for i in range(1, 94)]
    all_cols = [f'feat_{i}' for i in range(1, 166)]

    feature_sets = {
        'local': local_cols,
        'all': all_cols
    }

    y_train, y_test = train_df['label'], test_df['label']
    rows = []
    for feat_name, cols in feature_sets.items():
        print(f"\n=================== feature: {feat_name} ({len(cols)}) ===================")
        X_train, X_test = train_df[cols], test_df[cols]

        # ran forest balanced
        rf = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=SEED, n_jobs=-1)
        rf.fit(X_train, y_train)
        rows.append(evaluate('RandomForest', feat_name, y_test, rf.predict_proba(X_test)[:, 1]))

        # gradient boosting; scikit-learn native, class balanced
        gb = HistGradientBoostingClassifier(class_weight='balanced', random_state=SEED)
        gb.fit(X_train, y_train)
        rows.append(evaluate('HistGradientBoosting', feat_name, y_test, gb.predict_proba(X_test)[:, 1]))

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / 'baseline_metrics.csv'
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"\nSaved to {out}")

if __name__ == "__main__":
    run_experiments()
