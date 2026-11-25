import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from matplotlib.patches import Patch
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression

from src.config import PROCESSED_DATA_DIR

def load_data():
    # load the transformed data
    folder = PROCESSED_DATA_DIR / 'transformation'
    train = pd.read_csv(folder / 'train_scaled.csv')
    test = pd.read_csv(folder / 'test_scaled.csv')

    cols = [f'feat_{i}' for i in range(1, 166)]

    print("training transactions:", len(train))
    print("testing transactions:", len(test))
    print("original features:", len(cols))
    return train, test, cols

def filter_selection(train, cols):
    # check correlations with the class label
    correlations = train[cols].corrwith(train['label'])
    ranked = correlations.abs().dropna().sort_values(ascending=False)

    # check correlations between features
    feature_corr = train[cols].corr().abs()

    # select 20 features
    selected = []
    for feature in ranked.index:
        if len(selected) == 20:
            break

        if all(feature_corr.loc[feature, other] < 0.9 for other in selected):
            selected.append(feature)

    results = pd.DataFrame({'feature': selected, 'correlation': correlations.loc[selected].values, 'absolute_correlation': ranked.loc[selected].values})
    print("filter method:")
    print("selected features:", len(results))
    print(results.round(4).to_string(index=False))
    return results, ranked

def wrapper_selection(train, cols):
    # use recursive feature elimination
    model = LogisticRegression(solver='liblinear', class_weight='balanced', max_iter=1000, random_state=42)
    selector = RFE(estimator=model, n_features_to_select=20,step=0.2)
    selector.fit(train[cols], train['label'])

    selected = [
        feature for feature, keep in zip(cols, selector.support_)
        if keep
    ]
    results = pd.DataFrame({'feature': selected, 'importance': abs(selector.estimator_.coef_[0])}).sort_values('importance', ascending=False)
    print("wrapper method:")
    print("selected features:", len(results))
    print(results.round(4).to_string(index=False))
    return results

def embedded_selection(train, cols):
    # use l1 regularization
    model = LogisticRegression(penalty='l1', solver='liblinear', C=0.1, class_weight='balanced', max_iter=1000, random_state=42)
    model.fit(train[cols], train['label'])
    importance = pd.Series(abs(model.coef_[0]), index=cols)

    # remove features with zero coefficients
    nonzero = importance[importance > 0]
    results = nonzero.nlargest(20).reset_index()
    results.columns = ['feature', 'importance']
    print("embedded method:")
    print("nonzero coefficients:", len(nonzero))
    print("selected features:", len(results))
    print(results.round(4).to_string(index=False))
    return results

def plot_top_features(results, column, xlabel, title, filename):
    # show the top 10 features
    top = results.head(10)

    plt.figure(figsize=(9, 5))
    plt.barh(top['feature'], top[column])
    plt.gca().invert_yaxis()
    plt.xlabel(xlabel)
    plt.ylabel('Feature')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()

def plot_heatmap(train, results, folder):
    # check correlations between selected features
    selected = results['feature'].tolist()
    correlations = train[selected].corr()

    plt.figure(figsize=(11, 9))
    sns.heatmap(correlations, cmap='coolwarm', vmin=-1, vmax=1, center=0)
    plt.title('Correlation Between Selected Features')
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(folder / 'selected_features_heatmap.png', dpi=300)
    plt.close()

def plot_comparison(ranked, results, folder):
    # compare selected and excluded features
    selected = set(results['feature'])

    colors = ['tab:blue' if feature in selected else 'lightgray' for feature in ranked.index]

    plt.figure(figsize=(11, 5))
    plt.bar(range(1, len(ranked) + 1), ranked.values, color=colors)
    plt.xlabel('Feature Rank by Absolute Correlation')
    plt.ylabel('Absolute Pearson Correlation')
    plt.title('Feature Ranking and Selected Features')
    plt.legend(handles=[Patch(color='tab:blue', label='Selected'), Patch(color='lightgray', label='Not Selected')])
    plt.tight_layout()
    plt.savefig(folder / 'feature_selection_comparison.png', dpi=300)
    plt.close()

def save_results(train, test, results, name, folder):
    # save the selected features
    results.to_csv(folder / f'{name}_features.csv', index=False)

    # save the reduced datasets
    cols = ['txId', 'time_step', 'label'] + results['feature'].tolist()
    train[cols].to_csv(folder / f'train_{name}.csv', index=False)
    test[cols].to_csv(folder / f'test_{name}.csv', index=False)

def main():
    # load the data
    train, test, cols = load_data()
    folder = PROCESSED_DATA_DIR / 'selection'
    folder.mkdir(parents=True, exist_ok=True)

    # filter method
    filtered, ranked = filter_selection(train, cols)
    save_results(train, test, filtered, 'filter', folder)

    plot_top_features(filtered, 'absolute_correlation', 'Absolute Pearson Correlation', 'Top 10 Features Selected by Pearson Correlation', folder / 'top10_selected_features.png')
    plot_heatmap(train, filtered, folder)
    plot_comparison(ranked, filtered, folder)

    # wrapper method
    wrapper = wrapper_selection(train, cols)
    save_results(train, test, wrapper, 'wrapper', folder)
    plot_top_features(wrapper, 'importance', 'Absolute Model Coefficient', 'Top 10 Features Selected by RFE', folder / 'wrapper_features.png')

    # embedded method
    embedded = embedded_selection(train, cols)
    save_results(train, test, embedded, 'embedded', folder)

    plot_top_features(embedded, 'importance', 'Absolute Model Coefficient', 'Top 10 Features Selected by L1 Regularization', folder / 'embedded_features.png')

    summary = pd.DataFrame({
        'method': ['Filter', 'Wrapper', 'Embedded'],
        'original_features': [len(cols)] * 3,
        'selected_features': [
            len(filtered),
            len(wrapper),
            len(embedded)
        ]
    })

    summary.to_csv(folder / 'selection_summary.csv', index=False)
    print("feature selection summary:")
    print(summary.to_string(index=False))
    print("files saved to:", folder)
    print("feature selection completed")

if __name__ == "__main__":
    main()