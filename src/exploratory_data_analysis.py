import pandas as pd, numpy as np #Tables and math functions
import matplotlib.pyplot as plt, seaborn as sns #Plotting functions
from scipy import stats #Chi-Squared Test 
import networkx as nx #Building graph and analyzing
import random

from src.config import RAW_DATA_DIR, FIGURES_DIR, RESULTS_DIR, SEED
pd.set_option('display.width', 300)
pd.set_option('display.max_columns', 50)

def load_data():
    features = pd.read_csv(RAW_DATA_DIR / 'elliptic_txs_features.csv', header=None)#Gets feature dataset
    features.columns = ['txId', 'time_step'] + [f'feat_{i}' for i in range(1, 166)] #Identifies columns from dataset
    classes = pd.read_csv(RAW_DATA_DIR / 'elliptic_txs_classes.csv') #Classes dataframe
    edges = pd.read_csv(RAW_DATA_DIR / 'elliptic_txs_edgelist.csv') #Edges dataframe

    df = features.merge(classes, on='txId') #Combines features with classes using common attribute (txId)
    df['class_name'] = df['class'].astype(str).map({'1': 'illicit', '2': 'licit', 'unknown': 'unknown'}) #Converts to text labels
    labeled = df[df['class_name'] != 'unknown'].copy() #Creates only labeled copy of dataset
    labeled['label'] = (labeled['class_name'] == 'illicit').astype(int) #numeric target variable
    feature_columns = [c for c in df.columns if c.startswith('feat_')] #List of feature columns
    return df, labeled, edges, feature_columns

def describe_features(df, feature_columns):
    print("Shape: ", df.shape) # # of rows and cols
    print(df.dtypes.value_counts()) #How many cols of each type
    print("\nMissing-value rate:") #Missing value rate per column
    print(df.isna().mean().sort_values(ascending=False).head())
    print("Duplicate txIds: ", df['txId'].duplicated().sum()) #Count of duplicates
    print("Lowest Unique Value Counts") # # of distinct vals per column, feat 3 and 113 are near categorical
    print(df[feature_columns].nunique().sort_values().head(10))

    X = df[feature_columns] #Dataframe with feature cols
    desc = pd.DataFrame({ #Table with one row per feature, 1 col per statistic
        'mean': X.mean(), 'median': X.median(), 'mode': X.mode().iloc[0],
        'min': X.min(), 'max': X.max(), 'q1': X.quantile(0.25), 'q3': X.quantile(0.75),
        'var': X.var(), 'std': X.std(), 'skew': X.skew(), 'kurtosis': X.kurt(),
    })
    desc['midrange'] = (desc['min'] + desc['max']) / 2
    desc['range'] = desc['max'] - desc['min']
    desc['iqr'] = desc['q3'] - desc['q1']
    desc['cv'] = desc['std'] / desc['mean'].abs()
    lowOutlier, highOutlier = desc['q1'] - 1.5 * desc['iqr'], desc['q3'] + 1.5 * desc['iqr']
    #Share of outliers
    desc['outlier_rate'] = ((X < lowOutlier) | (X> highOutlier)).mean()
    #Count of features with skew above 1
    print("Features with |skew| > 1: ", (desc['skew'].abs() > 1).sum(), "of ", len(desc))
    print(desc.sort_values('skew', key=abs, ascending=False).head(10)[['skew', 'kurtosis']])
    desc.to_csv(RESULTS_DIR / 'feature_descriptive_stats.csv')
    return desc

def class_balance(df, labeled):
    print("Class counts: ") # of licit/illicit/unknown transactions
    print(df['class_name'].value_counts())
    print("Class proportions: ")
    print(df['class_name'].value_counts(normalize=True))
    print("Labeled Only Proportions (0=licit, 1=illicit):") #Balance in labelled set
    print(labeled['label'].value_counts(normalize=True))
    print("Illicit rate, train (True) vs test (False): ") #Balance of classes in test/train
    print(labeled.groupby(labeled['time_step'] <= 34)['label'].mean())

def correlations(labeled, feature_columns, desc):
    corr_label = labeled[feature_columns].corrwith(labeled['label']).sort_values(key=abs, ascending=False) 
    #Pearson correlation of each feature with label
    corr = labeled[feature_columns].corr() #Heat map
    upper = corr.where(np.triu(np.ones(corr.shape, dtype=bool), k=1)) #Upper triangle above diagonal
    high_pairs = upper.stack()[lambda s: s.abs() > 0.9] #Flatten to list with |r| > 0.9

    ct = pd.crosstab(labeled['time_step'], labeled['label']) #Contingency table
    chi2, p, dof, _ = stats.chi2_contingency(ct) #Chi-Squared Test, illicit rate on time step
    print("Feature pairs with |r| > 0.9: ", len(high_pairs)) #Print highly correlated feature pairs
    print("\nTop 10 Features correlated with label: ")
    print(corr_label.head(10)) #Show top 10 features
    print(f'Chi-square time_step vs label: chi2 = {chi2:.1f}, dof={dof}, p={p:.3g}') #Print result, chi2 to 1 decimal place and p to 3

    top10 = corr_label.index[:10] #Top 10 features for table
    cols = ['median', 'mode', 'min', 'max', 'iqr', 'skew', 'kurtosis', 'outlier_rate']
    top10_stats = desc.loc[top10, cols].copy()
    top10_stats.insert(0, 'corr_with_label', corr_label[top10])

    print(top10_stats.round(3))
    top10_stats.to_csv(RESULTS_DIR / 'top10_label_correlated_stats.csv')
    corr_label.to_csv(RESULTS_DIR / 'feature_label_correlation_stats.csv')
    return corr_label

def time_series(df, labeled):
    timeSeries = df.groupby('time_step').agg(n=('txId', 'size')) #Row per time step
    timeSeries['illicit_rate'] = labeled.groupby('time_step')['label'].mean() #Add illicit rate per step
    print(timeSeries['illicit_rate'].describe()) #Summary statistics of illicit rate for all 49 steps
    print("Volatility (std of step-to-step change):", timeSeries['illicit_rate'].diff().std()) #Show volatility
    timeSeries.to_csv(RESULTS_DIR / 'time_series_stats.csv')
    return timeSeries

def graph_stats(edges):
    G = nx.from_pandas_edgelist(edges, 'txId1', 'txId2') #Set of all transactions
    deg = pd.Series(dict(G.degree)) #Degree of all nodes
    print("Nodes: ", G.number_of_nodes()) #All transactions
    print("Edges: ", G.number_of_edges()) #All edges 
    print("Density: ", nx.density(G)) #Actual edges / max possible edges
    print(deg.describe()) #Summary of degree distribution
    print("Connected components: ", nx.number_connected_components(G)) # # of seperate subgraphs
    avg_clustering = nx.average_clustering(G)
    print("Average clustering: ", avg_clustering) #Avg fraction of node neighbors linked to each other

    sources = random.sample(list(G.nodes), 300)
    path_lengths = pd.Series([d for s in sources for d in nx.single_source_shortest_path_length(G, s).values() if d > 0])
    comp_sizes = pd.Series([len(c) for c in nx.connected_components(G)])

    graph_summary = pd.Series({
        'Nodes': G.number_of_nodes(),
        'Edges': G.number_of_edges(),
        'Average Degree': deg.mean(),
        'Median Degree': deg.median(),
        'Max Degree': deg.max(),
        'Density': nx.density(G),
        'Average Clustering Coefficient': avg_clustering,
        'Global Clustering': nx.transitivity(G),
        'Connected Components': len(comp_sizes),
        'Largest Component (nodes)': comp_sizes.max(),
        'Smallest Component (nodes)': comp_sizes.min(),
        'Mean Shortest Path (sampled)': path_lengths.mean(),
        'Median Shortest Path (sampled)': path_lengths.median(),
        'Longest Shortest Path Seen (sampled)': path_lengths.max(),
    }, name='value').to_frame()
    graph_summary.to_csv(RESULTS_DIR / 'graph_summary.csv')
    print(graph_summary.to_string())

def make_plots(df, labeled, timeSeries, corr_label):
    top = corr_label.index[0]
    second = corr_label.index[1]
    plt.figure() #Transactions per time step
    timeSeries['n'].plot(kind='line')
    plt.title('Transaction Volume Across Time Steps')
    plt.xlabel('Time Step (~2 weeks)')
    plt.ylabel('Number of Transactions')
    plt.axvline(34.5, color = 'grey')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / '01_Transaction_Volume.png', dpi=150, bbox_inches='tight')
    plt.close()

    plt.figure() #Count per class 
    df['class_name'].value_counts().plot(kind='bar')
    plt.title('Transaction Class Counts')
    plt.xlabel('Class')
    plt.ylabel('Number of Transactions')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / '02_Class_Counts.png', dpi=150, bbox_inches='tight')
    plt.close()

    plt.figure() #Top 20 features
    corr_label.head(20).plot(kind='barh')
    plt.title('Top 20 Features Associated with Illicit Transactions')
    plt.xlabel('Pearson correlation with label (negative = lower in illicit)')
    plt.ylabel('Feature')
    plt.axvline(0, color = 'black', linewidth = 0.8)
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / '03_Pearson_Features_T20.png', dpi=150, bbox_inches='tight')
    plt.close()

    plt.figure() #Distribution of top feature
    df[top].plot(kind='hist', bins=100)
    plt.yscale('log')
    plt.axvline(df[top].median(), color='black', linestyle='--', label=f'Median = {df[top].median():.2f}')
    plt.title("Distribution of Top Feature")
    plt.xlabel(f'{top} (standardized)')
    plt.ylabel('Count (log scale)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / '04_Distribution_Feature.png', dpi=150, bbox_inches='tight')
    plt.close()

    plt.figure() #Class share percentage
    df['class_name'].value_counts().plot(kind='pie', autopct='%1.1f%%')
    plt.title(f'Class Share of All {len(df):,} Transactions')
    plt.ylabel('')
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / '05_Class_Share.png', dpi=150, bbox_inches='tight')
    plt.close()

    plt.figure() #Smoothed density of top feature
    labeled.groupby('class_name')[top].plot(kind='kde', legend=True)
    plt.xlim(-1.5, 6)
    plt.title(f'Illicit Transactions Cluster at Low {top}')
    plt.xlabel(f'{top} (standardized)')
    plt.ylabel('Density')
    plt.legend(title='Class')
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / '06_Transaction_Cluster.png', dpi=150, bbox_inches='tight')
    plt.close()

    #Counts per class per step
    pd.crosstab(df['time_step'], df['class_name']).plot(kind='area')
    plt.title('Transactions Per Class Per Step')
    plt.xlabel('Time Step')
    plt.ylabel('Number of Transactions')
    plt.axvline(34.5, color='black', linestyle='--')
    plt.legend(title='Class', loc='upper right')
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / '07_Per_Class_Per_Step.png', dpi=150, bbox_inches='tight')
    plt.close()

    #Top feature distribution
    labeled.boxplot(column=top, by='class_name')
    plt.suptitle('')
    plt.title(f'Boxplot of Class Statistics')
    plt.xlabel('Class')
    plt.ylabel(f'{top} (standardised)')
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / '08_Boxplot_Class_Stats.png', dpi=150, bbox_inches='tight')
    plt.close()

    #Top vs second feature, colored by label
    labeled.plot(kind='scatter', x=top, y=second, c='label', alpha=0.3, colormap='coolwarm')
    plt.title(f'Label Correlated Features: {top} vs {second}')
    plt.xlabel(top)
    plt.ylabel(second)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / '09_Correlated_Features.png', dpi=150, bbox_inches='tight')
    plt.close()

    #Same pair, point density
    labeled.plot(kind='hexbin', x=top, y=second, gridsize=40)
    plt.title('Clustering of Transactions')
    plt.xlabel(top)
    plt.ylabel(second)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / '10_Clustering.png', dpi=150, bbox_inches='tight')
    plt.close()

    plt.figure() #Correlation matrix of 30 most label-correlated features
    sns.heatmap(labeled[corr_label.index[:30]].corr(), cmap='coolwarm', center=0)
    plt.title('Correlation Among 30 Most Label-Correlated Features')
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / '11_Heatmap.png', dpi=150, bbox_inches='tight')
    plt.close()

def main():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    random.seed(SEED)
    np.random.seed(SEED)

    df, labeled, edges, feature_columns = load_data()
    desc = describe_features(df, feature_columns)
    class_balance(df, labeled)
    corr_label = correlations(labeled, feature_columns, desc)
    timeSeries = time_series(df, labeled)
    graph_stats(edges)
    make_plots(df, labeled, timeSeries, corr_label)

if __name__ == "__main__":
    main()
