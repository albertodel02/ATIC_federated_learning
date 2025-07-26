# ATIC Federated Learning for Financial Time Series Classification

## Overview
This project demonstrates federated learning for financial time series classification, using real-world stock index data. It includes:
- **Data preprocessing** (see `data_preproces.ipynb`): transforms raw financial data into a machine-learning-ready format with engineered features.
- **Federated learning experiments** (`federated_atic_flower.py`): compares local, centralized, and federated training for binary classification of future returns.

---

## Data Preprocessing

The notebook `data_preproces.ipynb` processes raw financial data (e.g., FTSE100, DAX, SP500) from CSV files. The steps are:

1. **Data Cleaning**
   - Converts price columns from European formats (e.g., '1.234,56') to floats using a custom conversion function.
   - Computes daily log returns from the cleaned price data.

2. **Target Construction**
   - Creates a 5-day forward cumulative log return window.
   - The prediction target is binary: 1 if the absolute cumulative return exceeds the median absolute return, 0 otherwise.
   - This creates a more balanced classification problem (approximately 50% positive/negative samples).

3. **Feature Engineering**
   - Adds 5 lagged target features (Lag_1, ..., Lag_5) using future target values shifted by 1-5 days.
   - Adds lagged date columns for each lag and extracts cyclical temporal features:
     - Day of week (sin/cos encoding)
     - Month (sin/cos encoding) 
     - Day of month (sin/cos encoding)
   - All cyclical features are standardized using only the training split (first 80% of data).

4. **Final Format**
   - Drops unnecessary columns (raw dates, unneeded temporal features).
   - Saves the processed data as `*_classification_easy.csv` (e.g., `FTSE100_classification_easy.csv`).

### Data Format
- **Date Range:** 01.01.2015 to 31.12.2023
- **Columns:**
  - `Date`: The date of the observation
  - `LogReturn`: Daily log return (not standardized)
  - `Target`: Binary label (1 if absolute 5-day forward cumulative return > median, else 0)
  - `Lag_1` ... `Lag_5`: Lagged target values (future target shifted by 1-5 days)
  - `lagX_dow_sin`, `lagX_dow_cos`: Standardized cyclical day-of-week features for each lag
  - `lagX_month_sin`, `lagX_month_cos`: Standardized cyclical month features for each lag
  - `lagX_dom_sin`, `lagX_dom_cos`: Standardized cyclical day-of-month features for each lag

---

## Federated Learning Script

The main script, `federated_atic_flower.py`, implements and compares three training paradigms:

1. **Local-only:** Each client trains on its own data partition.
2. **Centralized:** A single model is trained on the full dataset.
3. **Federated (FedAvg):** Multiple clients collaboratively train a global model without sharing raw data.

### Usage

#### Server (runs all experiments and plots results):
```bash
python federated_atic_flower.py --role server --rounds 50
```

#### Clients (run in separate terminals):
```bash
python federated_atic_flower.py --role client --cid 0
python federated_atic_flower.py --role client --cid 1
python federated_atic_flower.py --role client --cid 2
```

### Command-Line Arguments

| Argument         | Type    | Default | Description |
|------------------|---------|---------|-------------|
| `--role`         | str     |         | `server` or `client`. Server runs all experiments and aggregates results; clients participate in federated training. |
| `--cid`          | int     | 0       | Client ID (used only with `--role client`). |
| `--rounds`       | int     | 50      | Number of training epochs/rounds. |
| `--data_dist`    | str     | random  | Data distribution among clients: `random` (random partition) or `quarters` (partition by calendar quarters, requires 4 clients). |
| `--num_clients`  | int     | 3       | Number of clients in the federated setup. |
| `--compare_dataset` | bool | False   | If True, compares two datasets (e.g., SP500 vs DAX) as separate clients. |
| `--epoch_variance` | bool | False   | If True, simulates clients training at different frequencies (realistic for asynchronous or resource-constrained settings). |
| `--smoothing`    | flag    | True    | If set, disables smoothing of plotted results. |
| `--retrain`      | bool    | False   | If True, retrains the last layer or from scratch on a new dataset after federated training (simulates transfer learning). |
| `--dp`           | flag    | False   | If set, enables differential privacy (DP) on client updates (simulates privacy-preserving FL). |
| `--dp-clip`      | float   | 1.0     | L2 norm clip for DP (controls sensitivity of updates). |
| `--dp-noise`     | float   | 0.1     | Noise standard deviation for DP (controls privacy/utility tradeoff). |

#### **Real-World Financial Relevance**
- **Data Distribution (`--data_dist`):** Simulates non-IID data, e.g., clients in different time zones, markets, or business cycles.
- **Client Count (`--num_clients`):** Models the number of data silos (e.g., banks, exchanges, regions).
- **Epoch Variance (`--epoch_variance`):** Reflects real-world heterogeneity in compute or data availability.
- **Differential Privacy (`--dp`, `--dp-clip`, `--dp-noise`):** Models regulatory or business requirements for privacy (e.g., GDPR, financial secrecy).
- **Retrain (`--retrain`):** Simulates transfer learning or domain adaptation (e.g., adapting a global model to a new market).

---

## Output
- **Plots:** The script produces a two-panel plot showing test loss (BCE) and accuracy over training rounds for all approaches.
- **Console Output:** Reports per-epoch/round metrics for each client, centralized, and federated model.

---

## Example Workflow
1. Preprocess your raw data using `data_preproces.ipynb` to generate `*_classification_easy.csv` files.
2. Run the federated learning experiments as described above.
3. Analyze the results to compare local, centralized, and federated learning in a realistic financial setting.

---

## Requirements
- Python 3.8+
- `flwr[simulation]`, `pandas`, `scikit-learn`, `tensorflow`, `matplotlib`

Install dependencies:
```bash
pip install flwr[simulation] pandas scikit-learn tensorflow matplotlib
```

---

## Notes
- The data covers the period from **01.01.2015 to 31.12.2023**.
- All preprocessing is done with care to avoid data leakage (scalers are fit only on training splits).
- The code is designed to be extensible for other financial time series and federated learning research.
