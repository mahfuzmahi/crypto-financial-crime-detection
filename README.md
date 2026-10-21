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
|  |--dataloader.py
|  |--exploratory_data_analysis.py
|  |--baseline_models.py
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
```
