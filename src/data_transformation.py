import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import QuantileTransformer

from src.config import PROCESSED_DATA_DIR, SEED
from src.exploratory_data_analysis import load_data

def split_data(labeled):
    # split the data by time step
    train = labeled[labeled['time_step'] <= 29].copy()
    validation = labeled[labeled['time_step'].between(30, 34)].copy()
    test = labeled[labeled['time_step'] >= 35].copy()
    print("training transactions:", len(train))
    print("validation transactions:", len(validation))
    print("testing transactions:", len(test))
    print("training illicit transactions:", (train['label'] == 1).sum())
    print("validation illicit transactions:", (validation['label'] == 1).sum())
    print("testing illicit transactions:", (test['label'] == 1).sum())
    return train, validation, test

def normalize_data(data, train, validation, test, cols):
    # fit the transformer on all transactions in the training time steps
    scaler = QuantileTransformer(output_distribution='normal', n_quantiles=1000, random_state=SEED)
    scaler.fit(data.loc[data['time_step'] <= 29, cols])

    # transform the training, validation and testing data
    train_scaled = train.copy()
    validation_scaled = validation.copy()
    test_scaled = test.copy()
    train_scaled[cols] = scaler.transform(train[cols])
    validation_scaled[cols] = scaler.transform(validation[cols])
    test_scaled[cols] = scaler.transform(test[cols])
    return train_scaled, validation_scaled, test_scaled

def check_normalization(train, train_scaled):
    # compare the features from assignment 2
    features = ['feat_53', 'feat_89', 'feat_142']

    results = pd.DataFrame({
        'feature': features,
        'mean_before': train[features].mean().values,
        'mean_after': train_scaled[features].mean().values,
        'std_before': train[features].std(ddof=0).values,
        'std_after': train_scaled[features].std(ddof=0).values
    })

    for feature in features:
        print(feature, "mean before:", round(train[feature].mean(), 4))
        print(feature, "std before:", round(train[feature].std(ddof=0), 4))

    for feature in features:
        print(feature, "mean after:", round(train_scaled[feature].mean(), 4))
        print(feature, "std after:", round(train_scaled[feature].std(ddof=0), 4))
    return results

def plot_normalization(results, folder):
    # compare standard deviations before and after transformation
    results.plot(x='feature', y=['std_before', 'std_after'], kind='bar', figsize=(10, 6))
    plt.xlabel('Features')
    plt.ylabel('Standard Deviation')
    plt.title('Before and After Quantile Transformation')
    plt.legend(['Before', 'After'])
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(folder / 'normalization_comparison.png', dpi=300)
    plt.close()

def save_results(train, validation, test, results, cols):
    # save the transformed data
    folder = PROCESSED_DATA_DIR / 'aligned' / 'transformation'
    folder.mkdir(parents=True, exist_ok=True)
    columns = ['txId', 'time_step', 'label'] + cols
    train[columns].to_csv(folder / 'train_scaled.csv', index=False)
    validation[columns].to_csv(folder / 'validation_scaled.csv', index=False)
    test[columns].to_csv(folder / 'test_scaled.csv', index=False)
    results.to_csv(folder / 'normalization_results.csv', index=False)
    plot_normalization(results, folder)
    print("files saved to:", folder)

def main():
    # load the data
    data, labeled, _, cols = load_data()
    train, validation, test = split_data(labeled)
    train_scaled, validation_scaled, test_scaled = normalize_data(data, train, validation, test, cols)
    results = check_normalization(train, train_scaled)
    save_results(train_scaled, validation_scaled, test_scaled, results, cols)
    print("data transformation completed")

if __name__ == "__main__":
    main()