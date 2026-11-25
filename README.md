# crypto-financial-crime-detection
Detecting Illicit Bitcoin Transactions in the Elliptic Dataset using baseline models and graph models

## Repository Structure
```
|--data/
|  |-- raw/
|  |-- processed/
|--reports/figures
|--results/
|--src/
|  |--config.py
|  |--data_loader.py
|  |--exploratory_data_analysis.py
|  |--baseline_models.py
|  |--graph_data.py
|  |--graph_sage.py
|  |--experiments.py
|  |--data_cleaning.py
|  |--data_transformation.py  
|  |--feature_selection.py
|  |--feature_extraction.py
|--requirements.txt
|--README.md
```

## Setup

**Requirements:** Python 3.13

```bash
git clone https://github.com/mahfuzmahi/crypto-financial-crime-detection.git
cd crypto-financial-crime-detection

py -3.13 -m venv .venv
.venv\Scripts\activate          #Windows
# source .venv/bin/activate     #macOS / Linux

pip install -r requirements.txt
```

## Data
1. Download the dataset from Kaggle <https://www.kaggle.com/datasets/ellipticco/elliptic-data-set>
2. Put all three files (elliptic_txs_features.csv, elliptic_txs_classes.csv, elliptic_txs_edgelist.csv) into data/raw

## How to Run
Run from the **repository root** as a module (`python -m ...`) in this order:
```
python -m src.data_loader

python -m src.data_cleaning
python -m src.data_transformation
python -m src.feature_selection
python -m src.feature_extraction

python -m src.exploratory_data_analysis

python -m src.baseline_models

python -m src.graph_data --features local
python -m src.graph_data --features all

python -m src.graph_sage --features local --seed 1      #repeat with --seed 2 and --seed 3
python -m src.graph_sage --features all --seed 1        #repeat with --seed 2 and --seed 3

python -m src.experiments --features local --seed 1     #GraphSAGE embeddings + Random Forest
python -m src.experiments --features all --seed 1
```
The preprocessing scripts are the analysis for Assignment 3, models use their own quantile transform
### Ablations 
`--tag` adds a suffix to the output file names so runs don't overwrite each other
```
python -m src.graph_sage --features local --seed 2525235 --tag reference
python -m src.graph_sage --features local --seed 2525235 --norm none --tag no_bn
python -m src.graph_sage --features local --seed 2525235 --dropout 0 --tag no_dropout
python -m src.graph_sage --features local --seed 2525235 --wd 0 --tag no_wd
python -m src.graph_sage --features local --seed 2525235 --patience 300 --tag no_earlystop
```
Other options: `--layers`, `--hidden`, `--lr`, `--epochs`, `--scaler {quantile,standard}`, `--clip`, `--no-class-weights`, `--no-edges`. Run `python -m src.graph_sage --help` for the full list

## Outputs
Results are saved in the following locations:
```
data/processed/train_split.csv, test_split.csv
results/feature_descriptive_stats.csv, graph_summary.csv, *_stats.csv
reports/figures/01_* ... 11_*
results/baseline_metrics.csv
results/graphsage_metrics_{features}_s{seed}[_{tag}].csv
results/graphsage_history_{features}_s{seed}[_{tag}].csv
reports/figures/12_*
reports/figures/13_*
results/experiment_rf_embeddings_{features}_s{seed}.csv
data/processed/cleaning, transformation, selection, extraction
```
