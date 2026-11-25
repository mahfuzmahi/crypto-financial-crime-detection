import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA

from src.config import PROCESSED_DATA_DIR

def load_data():
    # load the normalized data
    folder = PROCESSED_DATA_DIR / 'transformation'
    train = pd.read_csv(folder / 'train_scaled.csv')
    test = pd.read_csv(folder / 'test_scaled.csv')
    cols = [f'feat_{i}' for i in range(1, 166)]
    print("training transactions:", len(train))
    print("testing transactions:", len(test))
    print("original features:", len(cols))
    return train, test, cols

def find_correlations(train, cols):
    # find highly correlated features
    correlations = train[cols].corr().abs()
    pairs = []
    features = set()

    for i, first in enumerate(cols):
        for second in cols[i + 1:]:
            value = correlations.loc[first, second]

            if value >= 0.9:
                pairs.append([first, second, value])
                features.update([first, second])

    results = pd.DataFrame(pairs, columns=['feature_1', 'feature_2', 'absolute_correlation']).sort_values('absolute_correlation', ascending=False)
    selected = [feature for feature in cols if feature in features]
    print("highly correlated pairs:", len(results))
    print("features selected for pca:", len(selected))
    print(results.head(10).round(4).to_string(index=False))
    return selected, results

def extract_features(train, test, selected):
    # apply pca to the correlated features
    if not selected:
        raise ValueError("no highly correlated features found")

    pca = PCA(svd_solver='full')
    pca.fit(train[selected])
    variance = pca.explained_variance_ratio_  # keep at least 95% of the variance
    cumulative = np.cumsum(variance)
    count = np.searchsorted(cumulative, 0.95) + 1
    names = [f'pc_{i}' for i in range(1, count + 1)]

    # transform the training and testing data
    train_pc = pd.DataFrame(pca.transform(train[selected])[:, :count], columns=names, index=train.index)
    test_pc = pd.DataFrame(pca.transform(test[selected])[:, :count], columns=names, index=test.index)

    # replace the correlated features
    train = pd.concat([train.drop(columns=selected), train_pc], axis=1)
    test = pd.concat([test.drop(columns=selected), test_pc], axis=1)
    results = pd.DataFrame({'component': range(1, len(variance) + 1), 'explained_variance': variance, 'cumulative_variance': cumulative})
    print("original correlated features:", len(selected))
    print("principal components retained:", count)
    print("variance retained:", round(cumulative[count - 1] * 100, 4), "%")
    return train, test, results, count

def plot_correlations(pairs, folder):
    # show the top 10 correlated pairs
    top = pairs.head(10)
    labels = top['feature_1'] + ' / ' + top['feature_2']

    plt.figure(figsize=(9, 5))
    plt.barh(labels, top['absolute_correlation'])
    plt.gca().invert_yaxis()
    plt.xlabel('Absolute Pearson Correlation')
    plt.ylabel('Feature Pair')
    plt.title('Top 10 Highly Correlated Feature Pairs')
    plt.xlim(0, 1)
    plt.tight_layout()
    plt.savefig(folder / 'correlated_features.png', dpi=300)
    plt.close()

def plot_variance(results, count, folder):
    # cumulative explained variance
    plt.figure(figsize=(9, 5))
    plt.plot(results['component'], results['cumulative_variance'] * 100, label='Cumulative Variance')
    plt.axhline(y=95,color='red', linestyle='--', label='95% Variance')
    plt.axvline(x=count, color='green', linestyle='--', label=f'{count} Components')
    plt.xlabel('Number of Principal Components')
    plt.ylabel('Cumulative Explained Variance (%)')
    plt.title('PCA Explained Variance')
    plt.legend()
    plt.tight_layout()
    plt.savefig(folder / 'pca_variance.png', dpi=300)
    plt.close()

def save_results(train, test, pairs, results, selected, count, cols):
    # save the extracted features
    folder = PROCESSED_DATA_DIR / 'extraction'
    folder.mkdir(parents=True, exist_ok=True)
    train.to_csv(folder / 'train_pca.csv', index=False)
    test.to_csv(folder / 'test_pca.csv', index=False)
    pairs.to_csv(folder / 'correlated_pairs.csv', index=False)
    results.to_csv(folder / 'pca_components.csv', index=False)
    total = len(cols) - len(selected) + count

    summary = pd.DataFrame({
        'original_features': [len(cols)],
        'correlated_features': [len(selected)],
        'unchanged_features': [len(cols) - len(selected)],
        'principal_components': [count],
        'final_features': [total],
        'features_reduced': [len(cols) - total],
        'variance_retained_percent': [
            results.loc[count - 1, 'cumulative_variance'] * 100
        ]
    })

    summary.to_csv(folder / 'pca_summary.csv', index=False)
    plot_correlations(pairs, folder)
    plot_variance(results, count, folder)
    print("pca summary:")
    print(summary.round(4).to_string(index=False))
    print("files saved to:", folder)

def main():
    train, test, cols = load_data()  # load the data
    selected, pairs = find_correlations(train, cols) # find correlated features
    train, test, results, count = extract_features(train, test, selected) # extract new features
    save_results(train, test, pairs, results, selected, count, cols)
    print("feature extraction completed")

if __name__ == "__main__":
    main()