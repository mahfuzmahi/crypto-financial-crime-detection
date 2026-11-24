import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, average_precision_score, confusion_matrix

from src.config import PROCESSED_DATA_DIR, RESULTS_DIR, SEED
SPLITS = {
    'val':          ((1, 29), (30, 34)), #model selection
    'test':         ((1, 34), (35, 49)), #original baseline setup
    'test_1_29':    ((1, 29), (35, 49)), #GraphSAGE training window
}
def evaluate(name, feat_name, split, y_test, probs, thr=0.5):
    preds = (probs >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, preds).ravel()
    row = dict(model=name, features=feat_name, split=split, threshold=thr,
               precision=precision_score(y_test, preds, zero_division=0),
               recall=recall_score(y_test, preds),
               f1=f1_score(y_test, preds, zero_division=0),
               pr_auc=average_precision_score(y_test, probs),
               tn=tn, fp=fp, fn=fn, tp=tp)
    print(f"\n-----------------{name} ({feat_name}, {split})")
    for k in ['precision', 'recall', 'f1', 'pr_auc']:
        print(f"{k:10s}: {row[k]:.4f}")
    print(f"Confusion Matrix [TN FP / FN TP]:\n[[{tn} {fp}]\n [{fn} {tp}]]")
    return row

def run_experiments():
    print("loading train & test datasets from data/processed/...")
    full = pd.concat([pd.read_csv(PROCESSED_DATA_DIR / 'train_split.csv'),
                      pd.read_csv(PROCESSED_DATA_DIR / 'test_split.csv')], ignore_index=True)

    local_cols = [f'feat_{i}' for i in range(1, 94)]
    all_cols = [f'feat_{i}' for i in range(1, 166)]

    feature_sets = {
        'local': [f'feat_{i}' for i in range(1, 94)], #93 local features
        'all': [f'feat_{i}' for i in range(1, 166)]   #93 local + 72 aggregated features
    }

    rows = []
    for split, ((tr_lo, tr_hi), (ev_lo, ev_hi)) in SPLITS.items():
        tr = full[full['time_step'].between(tr_lo, tr_hi)]
        ev = full[full['time_step'].between(ev_lo, ev_hi)]
        print(f"\n====== split: {split} | train {tr_lo}-{tr_hi} ({len(tr):,}) | eval {ev_lo} - {ev_hi} ({len(ev):}) ======")
        for feat_name, cols in feature_sets.items():
            print(f"\n=================== feature: {feat_name} ({len(cols)}) ===================")
            X_train, Y_train, X_ev, Y_ev = tr[cols], tr['label'], ev[cols], ev['label']


            # random forest balanced
            rf = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=SEED, n_jobs=-1)
            rf.fit(X_train, Y_train)
            rows.append(evaluate('RandomForest', feat_name, split, Y_ev, rf.predict_proba(X_ev)[:, 1]))

            # gradient boosting; scikit-learn native, class balanced
            gb = HistGradientBoostingClassifier(class_weight='balanced', random_state=SEED)
            gb.fit(X_train, Y_train)
            rows.append(evaluate('HistGradientBoosting', feat_name, split, Y_ev, gb.predict_proba(X_ev)[:, 1]))

            for r in rows[-2:]:
                r.update(train_steps=f'{tr_lo}-{tr_hi}', eval_steps=f'{ev_lo}-{ev_hi}')

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / 'baseline_metrics.csv'
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"\nSaved to {out}")

if __name__ == "__main__":
    run_experiments()
