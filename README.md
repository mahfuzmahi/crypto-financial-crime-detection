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

python -m src.exploratory_data_analysis

python -m src.baseline_models

python -m src.graph_data --features local
python -m src.graph_data --features all

python -m src.graphsage --features local --seed 1      #repeat with --seed 2 and --seed 3
python -m src.graphsage --features all --seed 1        #repeat with --seed 2 and --seed 3
```

## Outputs
Results are saved in the following locations:
```
results/baseline_metrics.csv
results/graph_sage_metrics_{features}_s{seed}.csv
results/graph_sage_history_{features}_s{seed}.csv
reports/figures/12_*
reports/figures/13_*
```
