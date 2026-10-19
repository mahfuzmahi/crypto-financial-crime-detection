import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, average_precision_score, confusion_matrix

def run_experiments():
    print("loading train & test datasets from data/processed/...")
    train_df = pd.read_csv('data/processed/train_split.csv')
    test_df = pd.read_csv('data/processed/test_split.csv')

    local_cols = [f'feat_{i}' for i in range(1, 95)]
    all_cols = [f'feat_{i}' for i in range(1, 166)]

    feature_sets = {
        'local features (1-94)': local_cols,
        'full features (1-165)': all_cols
    }

    y_train, y_test = train_df['label'], test_df['label']

    for feat_name, cols in feature_sets.items():
        print(f"\n=================== feature: {feat_name} ===================")
        X_train, X_test = train_df[cols], test_df[cols]

        # ran forest balanced
        rf = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42, n_jobs=-1)
        rf.fit(X_train, y_train)
        rf_probs = rf.predict_proba(X_test)[:, 1]
        rf_preds = (rf_probs >= 0.5).astype(int)

        print("\n-------------- random forest (balanced) ")
        print(f"precision : {precision_score(y_test, rf_preds):.4f}")
        print(f"recall    : {recall_score(y_test, rf_preds):.4f}")
        print(f"F1-score  : {f1_score(y_test, rf_preds):.4f}")
        print(f"PR-AUC    : {average_precision_score(y_test, rf_probs):.4f}")
        print("confusion matrix [TN FP / FN TP]:")
        print(confusion_matrix(y_test, rf_preds))

        # gradient boosting; scikit-learn native, class balanced
        gb = HistGradientBoostingClassifier(class_weight='balanced', random_state=42)
        gb.fit(X_train, y_train)
        gb_probs = gb.predict_proba(X_test)[:, 1]
        gb_preds = (gb_probs >= 0.5).astype(int)

        print("\n-------------- HistGradientBoosting (balanced) ")
        print(f"precision : {precision_score(y_test, gb_preds):.4f}")
        print(f"recall    : {recall_score(y_test, gb_preds):.4f}")
        print(f"F1-score  : {f1_score(y_test, gb_preds):.4f}")
        print(f"PR-AUC    : {average_precision_score(y_test, gb_probs):.4f}")
        print("confusion matrix [TN FP / FN TP]:")
        print(confusion_matrix(y_test, gb_preds))

if __name__ == "__main__":
    run_experiments()