import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

from src.config import PROCESSED_DATA_DIR
from src.exploratory_data_analysis import load_data

def prepare_data():
    # load the labeled data
    _, labeled, _, cols = load_data()

    # split the data using time steps
    train = labeled[labeled['time_step'] <= 34].copy()
    test = labeled[labeled['time_step'] > 34].copy()
    print("training transactions:", len(train))
    print("testing transactions:", len(test))
    return train, test, cols

def scale_data(train, test, cols):
    # choose features to display
    features = ['feat_53', 'feat_89', 'feat_142']

    # check values before scaling
    before = []
    for feature in features:
        print(feature, "mean before:", round(train[feature].mean(), 4))
        print(feature, "std before:", round(train[feature].std(ddof=0), 4))
        before.append(train[feature].std(ddof=0))

    # scale all numerical features
    scaler = StandardScaler()
    train[cols] = scaler.fit_transform(train[cols])
    test[cols] = scaler.transform(test[cols])

    # check values after scaling
    after = []
    for feature in features:
        print(feature, "mean after:", round(train[feature].mean(), 4))
        print(feature, "std after:", round(train[feature].std(ddof=0), 4))
        after.append(train[feature].std(ddof=0))

    plot_results(features, before, after)
    return train, test

def plot_results(features, before, after):
    # compare standard deviations
    x = range(len(features))

    plt.figure(figsize=(8, 5))
    plt.bar([i - 0.2 for i in x], before, width=0.4, label='Before')
    plt.bar([i + 0.2 for i in x], after, width=0.4, label='After')
    plt.xticks(list(x), features)
    plt.xlabel('Features')
    plt.ylabel('Standard Deviation')
    plt.title('Before and After Z-score Normalization')
    plt.legend()
    plt.tight_layout()
    folder = PROCESSED_DATA_DIR / 'transformation'
    folder.mkdir(parents=True, exist_ok=True)
    plt.savefig(folder / 'normalization_comparison.png', dpi=300)
    plt.close()

def save_data(train, test):
    # save the scaled data
    folder = PROCESSED_DATA_DIR / 'transformation'
    folder.mkdir(parents=True, exist_ok=True)
    train.to_csv(folder / 'train_scaled.csv', index=False)
    test.to_csv(folder / 'test_scaled.csv', index=False)
    print("files saved to:", folder)

def main():
    # load, prepare, scale the data
    train, test, cols = prepare_data()
    train, test = scale_data(train, test, cols)
    save_data(train, test)
    print("data transformation completed")

if __name__ == "__main__":
    main()