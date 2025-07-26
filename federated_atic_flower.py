"""
Federated vs Local vs Centralized Comparison for ATIC S&P-500 Forecasting
========================================================================

This script compares three modes of training:

 1. **Local-only**: each client trains on its slice for `rounds` epochs.
 2. **Centralized**: a single model trained on the full training set, tracking best epoch.
 3. **Federated (FedAvg)**: three-client federated learning, 1 epoch per round, requiring all clients.

It then produces a publication-ready two-panel figure showing:
 - Test MSE over federated rounds
 - Test MAE over federated rounds

With:
 - Gray bands for the mean ±1σ of local-only (clients trained in isolation)
 - Solid line for FedAvg trajectory
 - Dashed line for centralized best result

Usage:
  # Server (computes all and shows plots):
  python federated_atic_flower.py --role server --rounds 50

  # Clients (in three separate terminals):
  python federated_atic_flower.py --role client --cid 0
  python federated_atic_flower.py --role client --cid 1
  python federated_atic_flower.py --role client --cid 2

Prerequisites:
  - `sp500_full.csv` in the same folder, with a 'target' column.
  - Install dependencies:
      pip install flwr[simulation] pandas scikit-learn tensorflow matplotlib
"""

import argparse
import sys
import os
import warnings
from typing import List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow import keras
import flwr as fl
import random
import tensorflow as tf

warnings.filterwarnings("ignore", category=FutureWarning)

RANDOM_SEED = 30
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

parser = argparse.ArgumentParser()
parser.add_argument("--role",   choices=["server", "client"], required=True)
parser.add_argument("--cid",    type=int, default=0)
parser.add_argument("--rounds", type=int, default=50)
parser.add_argument("--data_dist", default="random",choices=["random", "quarters"], type=str)
parser.add_argument("--num_clients", default= 3, type=int)
parser.add_argument("--compare_dataset", default=False, type = bool)
parser.add_argument("--epoch_variance", default=False, type = bool)
parser.add_argument("--smoothing", action="store_false", dest="smoothing", help="Disable smoothing (default: enabled)")
parser.add_argument("--retrain", default=False, type = bool)
parser.add_argument("--dp", action="store_true", help="Enable differential privacy on client updates")
parser.add_argument("--dp_clip", type=float, default=1.0, help="L2 norm clip for DP (default: 1.0)")
parser.add_argument("--dp_noise", type=float, default=0.01, help="Noise stddev for DP (default: 0.1)")
parser.set_defaults(smoothing=True)
args = parser.parse_args()

NUM_CLIENTS = args.num_clients
fed_losses: List[float] = []
fed_maes:   List[float] = []

local_hist_losses = np.zeros((NUM_CLIENTS, 0)).tolist()   # will become lists of length = rounds
local_hist_maes   = np.zeros((NUM_CLIENTS, 0)).tolist()

central_hist_loss = []
central_hist_mae  = []

# Change all *_loss to *_bce, *_mae to *_acc, and update print/plot labels
fed_bces: List[float] = fed_losses
fed_accs: List[float] = fed_maes
local_hist_bces = local_hist_losses
local_hist_accs = local_hist_maes
central_hist_bce = central_hist_loss
central_hist_acc = central_hist_mae

def load_features(path: str = "Data/SP500/SP500_classification_easy.csv") -> pd.DataFrame:
    if not os.path.exists(path):
        sys.exit(f"Error: '{path}' not found.")
    df = pd.read_csv(path, index_col=0).astype(np.float32)
    df.columns = df.columns.str.lower()
    index_array = df.index.to_numpy()
    df = df.reset_index(drop=True)
    # Combine columns starting with the same Lag_'id' (from 1-5) into one column as np arrays
    for lag_id in range(5, 0, -1):
        name = f"feature_{lag_id}"
        prefix = f"lag_{lag_id}"
        date_prefix = f"lag{lag_id}"
        lag_cols = [col for col in df.columns if (col.lower().startswith(prefix) or col.startswith(date_prefix))]
        # Combine the columns into a single column of np arrays
        #df[name] = df[lag_cols].apply(lambda row: float(row.values[0]), axis=1)
        df[name] = df[lag_cols].apply(lambda row: row.values, axis=1)
        df = df.drop(columns=lag_cols)
    print(df)
    # Combine all columns whose names start with "Lag_" or "lag_" into a single column "Lag_all" as np arrays
    lag_all_cols = [col for col in df.columns if col.lower().startswith("feature")]
    if len(lag_all_cols) > 1:
        # Each row is a 1D np array; stack them to make a 2D array per row
        df["features"] = df[lag_all_cols].apply(lambda row: np.stack(row.values), axis=1)
        df = df.drop(columns=lag_all_cols)
    # Print the first row's "features" column, looping over the first dimension if it is 2D
    print(df.head(10))
    first_row = df.iloc[0]
    features = first_row["features"]
    if isinstance(features, np.ndarray) and features.ndim == 2:
        for i in range(features.shape[0]):
            print(f"features[{i}]: {features[i]}")
    else:
        print("First row 'features' is not 2D:", features)
    # Invert the order of the whole dataset (reverse rows)
    df = df.iloc[::-1].reset_index(drop=True)
    index_array = index_array[::-1]
    return df, index_array


def build_model(input_dim: int) -> keras.Model:
    model = keras.Sequential([
        #keras.layers.Input(shape=(input_dim[1],1,)),
        keras.layers.Input(shape=(input_dim[1],input_dim[2],)),
        keras.layers.LSTM(units=4, activation='relu'),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(units=1, activation='sigmoid'),
    ])
    model.compile(optimizer="adam", loss='binary_crossentropy', metrics=['accuracy'])
    return model

def dp_clip_and_noise(weights, l2_norm_clip, noise_std, seed=None):
    """
    Clips the model update to l2_norm_clip and adds Gaussian noise.
    Args:
        weights: List of np.arrays (model layers).
        l2_norm_clip: float, maximum L2 norm.
        noise_std: float, noise standard deviation.
        seed: int or None.
    Returns:
        List of np.arrays.
    """
    # Flatten all weights to compute global norm
    flat = np.concatenate([w.flatten() for w in weights])
    norm = np.linalg.norm(flat)
    # Clip
    if norm > l2_norm_clip:
        scale = l2_norm_clip / (norm + 1e-10)
        weights = [w * scale for w in weights]
    # Add noise
    rng = np.random.default_rng(seed)
    noisy_weights = [w + rng.normal(0, noise_std, w.shape) for w in weights]
    return noisy_weights


class FLClient(fl.client.NumPyClient):
    def __init__(
        self,
        cid: int,
        train_data: Tuple[np.ndarray, np.ndarray],
        test_data:  Tuple[np.ndarray, np.ndarray]
    ):
        self.cid = cid
        self.model = build_model(train_data[0].shape)
        self.x_train, self.y_train = train_data
        self.x_test, self.y_test   = test_data

    def get_parameters(self, config=None):
        return self.model.get_weights()

    def fit(self, parameters, config):
        self.model.set_weights(parameters)
        base_epochs = config.get("base_epochs", 1)
        round_num = int(config.get("round_num", 1))
        # Set base_epochs to 0 if round_num < cid*10, else use the provided base_epochs
        if args.epoch_variance:
            if (round_num-1) % (self.cid + 1) != 0:
                base_epochs = 0
        self.model.fit(
            self.x_train,
            self.y_train,
            epochs=base_epochs,
            batch_size=32,
            verbose=0,
        )
        updated_weights = self.model.get_weights()

        # Differential Privacy: clip and add noise if enabled
        if args.dp:
            # Use unique seed for each client for reproducibility
            seed = int(self.cid) + 42
            updated_weights = dp_clip_and_noise(
                updated_weights, args.dp_clip, args.dp_noise, seed=seed
            )
        return updated_weights, len(self.x_train), {}

    def evaluate(self, parameters, config):
        self.model.set_weights(parameters)
        loss, acc = self.model.evaluate(self.x_test, self.y_test, verbose=0)
        return loss, len(self.x_test), {"accuracy": acc}
    
_best_acc, _best_acc_weights = 0, None
def get_fed_eval_fn(test_data: Tuple[np.ndarray, np.ndarray]):
    def evaluate(round_num: int, parameters, config):
        model = build_model(test_data[0].shape)
        model.set_weights(parameters)
        loss, acc = model.evaluate(test_data[0], test_data[1], verbose=0)
        global _best_acc, _best_acc_weights
        if acc > _best_acc:
            _best_acc = acc
            _best_acc_weights = parameters
        # ----- NEW: skip the initial round-0 measurement -----
        if round_num == 0:          # baseline, keep lists at 50 points
            return loss, {"accuracy": acc}

        fed_bces.append(loss)
        fed_accs.append(acc)
        print(f"[FedAvg] Round {round_num:02d}  •  BCE={loss:.6f}  •  Accuracy={acc:.4f}")
        return loss, {"accuracy": acc}
    return evaluate

def start_federated(test_data: Tuple[np.ndarray, np.ndarray], num_rounds: int) -> None:
    def fit_config_fn(server_round: int):
        return {"round_num": str(server_round),"base_epochs": 1}

    # Now start the federated server as usual
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=NUM_CLIENTS,
        min_evaluate_clients=NUM_CLIENTS,
        min_available_clients=NUM_CLIENTS,
        evaluate_fn=get_fed_eval_fn(test_data),
        on_fit_config_fn=fit_config_fn,
    )
    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=num_rounds),
        strategy=strategy,
    )
    


def start_client(
    cid: int,
    train_data: Tuple[np.ndarray, np.ndarray],
    test_data:  Tuple[np.ndarray, np.ndarray]
) -> None:
    
    client = FLClient(cid, train_data, test_data)
    fl.client.start_numpy_client(
        server_address="127.0.0.1:8080",
        client=client,
    )

def smooth_results(local_hist_bces, local_hist_accs,
                   fed_bces, fed_accs,
                   central_hist_bce, central_hist_acc,
                   window=5,
                   retrain_hist_bce=None, retrain_hist_acc=None,
                   scratch_hist_bce=None,scratch_hist_acc=None):

    def running_average(x, w):
        x = np.asarray(x)
        if len(x) < w:
            return x
        return np.convolve(x, np.ones(w)/w, mode='valid')

    # Smooth local_hist_losses and local_hist_maes per client
    smoothed_local_hist_bces = [running_average(losses, window).tolist() for losses in local_hist_bces]
    smoothed_local_hist_accs   = [running_average(maes, window).tolist() for maes in local_hist_accs]

    # Smooth the rest (1D lists)
    smoothed_fed_bces        = running_average(fed_bces, window).tolist()
    smoothed_fed_accs          = running_average(fed_accs, window).tolist()
    smoothed_central_hist_bce = running_average(central_hist_bce, window).tolist()
    smoothed_central_hist_acc  = running_average(central_hist_acc, window).tolist()

    # Optionally smooth retrain results
    if retrain_hist_bce is not None and retrain_hist_acc is not None:
        smoothed_retrain_hist_bce = running_average(retrain_hist_bce, window).tolist()
        smoothed_retrain_hist_acc  = running_average(retrain_hist_acc, window).tolist()
    else:
        smoothed_retrain_hist_bce = None
        smoothed_retrain_hist_acc  = None

    # Optionally smooth retrain results
    if scratch_hist_bce is not None and scratch_hist_acc is not None:
        smoothed_scratch_hist_bce = running_average(scratch_hist_bce, window).tolist()
        smoothed_scratch_hist_acc  = running_average(scratch_hist_acc, window).tolist()
    else:
        smoothed_scratch_hist_bce = None
        smoothed_scratch_hist_acc  = None

    return (smoothed_local_hist_bces, smoothed_local_hist_accs,
            smoothed_fed_bces, smoothed_fed_accs,
            smoothed_central_hist_bce, smoothed_central_hist_acc,
            smoothed_retrain_hist_bce, smoothed_retrain_hist_acc,
            smoothed_scratch_hist_bce, smoothed_scratch_hist_acc)

def plot_paper_figure(local_hist_bces, local_hist_accs,
                      fed_bces, fed_accs,
                      central_hist_bce, central_hist_acc,
                      retrain_hist_bce=None, retrain_hist_acc=None,
                      scratch_hist_bce=None,scratch_hist_acc=None):
    
    # Apply smoothing if enabled
    if args.smoothing:
        (local_hist_bces, local_hist_accs,
         fed_bces, fed_accs,
         central_hist_bce, central_hist_acc,
         retrain_hist_bce, retrain_hist_acc,
         scratch_hist_bce,scratch_hist_acc) = smooth_results(
            local_hist_bces, local_hist_accs,
            fed_bces, fed_accs,
            central_hist_bce, central_hist_acc,
            window=5,
            retrain_hist_bce=retrain_hist_bce,
            retrain_hist_acc=retrain_hist_acc,
            scratch_hist_bce=scratch_hist_bce,
            scratch_hist_acc=scratch_hist_acc
        )
    
    rounds = np.arange(1, len(local_hist_bces[0]) + 1)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8), sharex=True)

    # --- BCE panel -------------------------------------------------
    # --- BCE panel: local mean ±1σ --------------------------------
    # Stack into array of shape (n_clients, n_rounds)
    local_arr = np.stack(local_hist_bces, axis=0)
    # Compute mean and std over clients
    mean_local_bce = local_arr.mean(axis=0)
    std_local_bce  = local_arr.std(axis=0)
    # Shade between (mean - std) and (mean + std)
    ax1.fill_between(
    rounds,
    mean_local_bce - std_local_bce,
    mean_local_bce + std_local_bce,
    color="gray",
    alpha=0.3,
    label="Local mean ±1σ"
    )
# (Optional) also plot the mean line itself
    ax1.plot(
    rounds,
    mean_local_bce,
    color="gray",
    linewidth=2,
    linestyle="--"
    )
    ax1.plot(rounds, central_hist_bce, "k--", linewidth=2,
             label="Centralised")
    ax1.plot(rounds, fed_bces, color="C0", linewidth=2,
             label="FedAvg")
    if retrain_hist_bce is not None:
        rounds_retrain = np.arange(0, len(retrain_hist_bce))
        ax1.plot(rounds_retrain, retrain_hist_bce, color="C3", linewidth=2, linestyle=":", label="FedAvg retrain last-layer")
    if scratch_hist_bce is not None:
        rounds_retrain = np.arange(0, len(scratch_hist_bce))
        ax1.plot(rounds_retrain, scratch_hist_bce, color="C4", linewidth=2, linestyle=":", label="FedAvg retrain from scratch")
    ax1.set_title("Test BCE")
    ax1.set_xlabel("Epoch / Round")
    ax1.set_ylabel("BCE")
    ax1.grid(alpha=0.3)
    # --- Accuracy panel -------------------------------------------------
    # --- Accuracy panel: local mean ±1σ ----------------------------
    local_acc_arr = np.stack(local_hist_accs, axis=0)
    mean_local_acc = local_acc_arr.mean(axis=0)
    std_local_acc  = local_acc_arr.std(axis=0)
    ax2.fill_between(
    rounds,
    mean_local_acc - std_local_acc,
    mean_local_acc + std_local_acc,
    color="gray",
    alpha=0.3,
    label="Local mean ±1σ"
    )
    ax2.plot(
    rounds,
    mean_local_acc,
    color="gray",
    linewidth=2,
    linestyle="--"
    )

    ax2.plot(rounds, central_hist_acc, "k--", linewidth=2)
    ax2.plot(rounds, fed_accs, "C0", linewidth=2)
    if retrain_hist_acc is not None:
        rounds_retrain = np.arange(0, len(retrain_hist_acc))
        ax2.plot(rounds_retrain, retrain_hist_acc, color="C3", linewidth=2, linestyle=":")
    if scratch_hist_acc is not None:
        rounds_retrain = np.arange(0, len(scratch_hist_acc))
        ax2.plot(rounds_retrain, scratch_hist_acc, color="C4", linewidth=2, linestyle=":")
    ax2.set_title("Test Accuracy")
    ax2.set_xlabel("Epoch / Round")
    ax2.set_ylabel("Accuracy")
    ax2.grid(alpha=0.3)
    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3)
    # (left,bottom,right,top)
    fig.tight_layout(rect=[0, 0.12, 1, 1])
    plt.show()

def random_client_data(n_samples, X_train_all, y_train_all):
    # Client slices
    client_slices = []
    all_indeces = np.random.permutation(np.arange(n_samples))
    size = n_samples // NUM_CLIENTS
    for cid in range(NUM_CLIENTS):
        start = cid * size
        stop = (cid + 1) * size if cid < NUM_CLIENTS - 1 else n_samples
        client_indeces = all_indeces[start:stop]
        client_slices.append((X_train_all[client_indeces], y_train_all[client_indeces]))
    return client_slices

def quarters_client_data(months,X_train_all, y_train_all):
    """
    Distribute data to clients by assigning each client a part of the year (months).
    For 4 clients, each gets 3 months (quarters). For more clients, months are split as evenly as possible.
    """
    client_slices = []
    # Always 4 clients, each gets 3 months (quarters)
    month_ranges = [(1, 3), (4, 6), (7, 9), (10, 12)]
    for m_start, m_end in month_ranges:
        client_indices = np.where((months >= m_start) & (months <= m_end))[0]
        client_slices.append((X_train_all[client_indices], y_train_all[client_indices]))
    return client_slices

# INSERT_YOUR_CODE
def custom_monthly_split(shuffled_df, dates):
    """
    Splits the dataframe into train and test sets such that every 5th month (from Jan 2015 to Dec 2023)
    is assigned to the test set, and the rest to the train set.
    """
    # Convert dates to pandas datetime if not already
    date_series = pd.to_datetime(dates)
    # Create a DataFrame to align indices
    idx_df = pd.DataFrame({
        "date": date_series,
        "orig_idx": np.arange(len(date_series))
    })
    # Only use dates within the range 2015-01-01 to 2023-12-31
    idx_df = idx_df[(idx_df["date"] >= pd.Timestamp("2015-01-01")) & (idx_df["date"] <= pd.Timestamp("2023-12-31"))]
    # Assign a "month_number" starting from 0 for Jan 2015
    idx_df["month_number"] = (idx_df["date"].dt.year - 2015) * 12 + (idx_df["date"].dt.month - 1)
    # Mark every 5th month as test
    idx_df["is_test"] = idx_df["month_number"] % 5 == 0
    # Get indices for test and train
    test_indices = idx_df[idx_df["is_test"]]["orig_idx"].values
    train_indices = idx_df[~idx_df["is_test"]]["orig_idx"].values
    # Select rows from shuffled_df
    test_df = shuffled_df.iloc[test_indices].reset_index(drop=True)
    train_df = shuffled_df.iloc[train_indices].reset_index(drop=True)
    return train_df, test_df

def retrain_data_loading(data):
    df,_ = load_features("Data/FTSE100/FTSE100_classification_easy.csv")
    split = int(len(df) * 0.8)
    train_df, test_df = df.iloc[:split], df.iloc[split:]
    # Only use a random 10% of the train set
    n_train = len(train_df)
    n_subset = max(1, int(n_train * 0.1))
    subset_indices = np.random.choice(n_train, n_subset, replace=False)
    train_df = train_df.iloc[subset_indices].reset_index(drop=True)
    X_test = np.stack(test_df["features"].to_numpy())
    y_test = test_df["target"].to_numpy()
    X_train = np.stack(train_df["features"].to_numpy())
    y_train = train_df["target"].to_numpy()
    data["X_train_retrain"] = X_train
    data["y_train_retrain"] = y_train
    data["X_test_retrain"] = X_test
    data["y_test_retrain"] = y_test
    return data

def data_loading():
    # Load and split
    df,dates = load_features()
    #train_df, test_df = custom_monthly_split(df,dates)
    split = int(len(df) * 0.8)
    train_df = df.iloc[:split]
    test_df = df.iloc[split:]
    # INSERT_YOUR_CODE
    print(f"Train set size: {len(train_df)} samples")
    print(f"Test set size: {len(test_df)} samples")
    # (samples, horizon, feature_dim)
    # Prepare test data
    X_test = np.stack(test_df["features"].to_numpy())
    y_test = test_df["target"].to_numpy()
    X_train = np.stack(train_df["features"].to_numpy())
    y_train = train_df["target"].to_numpy()
    # Compute client slices
    if args.data_dist == 'random':
        client_slices = random_client_data(len(train_df), X_train, y_train)
    elif args.data_dist == 'quarters':
        if args.num_clients == 4:
            months = np.array([int(str(date).split("/")[0]) for date in dates[:len(train_df)]])
            client_slices = quarters_client_data(months, X_train, y_train)
        else:
            sys.exit("Error: quarters data_dist requires num_clients == 4")
    # Store all data in a dict
    data_dict = {
        "X_train_all": X_train,
        "y_train_all": y_train,
        "X_test": X_test,
        "y_test": y_test,
        "client_slices": client_slices,
    }
    if args.retrain:
        data_dict = retrain_data_loading(data_dict)
    return data_dict


def data_loading_compare():
    # Load and split both datasets
    df1,_ = load_features()
    df2,_ = load_features("Data/DAX/DAX_classification_easy.csv")
    # Split each dataset into train/test
    split1 = int(len(df1) * 0.8)
    train_df1, test_df1 = df1.iloc[:split1], df1.iloc[split1:]
    split2 = int(len(df2) * 0.8)
    train_df2, test_df2 = df2.iloc[:split2], df2.iloc[split2:]
    # Prepare test data (concatenate for global test set)
    X_test1 = np.stack(test_df1["features"].to_numpy())
    y_test1 = test_df1["target"].to_numpy()
    X_test2 = np.stack(test_df2["features"].to_numpy())
    y_test2 = test_df2["target"].to_numpy()
    X_test = np.concatenate([X_test1, X_test2], axis=0)
    y_test = np.concatenate([y_test1, y_test2], axis=0)
    X_train1 = np.stack(train_df1["features"].to_numpy())
    y_train1 = train_df1["target"].to_numpy()
    X_train2 = np.stack(train_df2["features"].to_numpy())
    y_train2 = train_df2["target"].to_numpy()
    # Combine for global train
    X_train_all = np.concatenate([X_train1, X_train2], axis=0)
    y_train_all = np.concatenate([y_train1, y_train2], axis=0)
    # Each client gets one full dataset
    client_slices = [
        (X_train1, y_train1),
        (X_train2, y_train2)
    ]
    if args.num_clients != 2:
        sys.exit("Error: data_loading_compare requires num_clients == 2 (one per dataset)")
    data_dict = {
        "X_train_all": X_train_all,
        "y_train_all": y_train_all,
        "X_test": X_test,
        "y_test": y_test,
        "client_slices": client_slices,
        "X_test1": X_test1,
        "y_test1": y_test1
    }
    if args.retrain:
        data_dict = retrain_data_loading(data_dict)
    return data_dict

def retrain_last_layer(fed_weights, data):
    X_train = data['X_train_retrain']
    y_train = data['y_train_retrain']
    X_test = data['X_test_retrain']
    y_test = data['y_test_retrain']
    # Model 1: Last-layer retrain (FedAvg weights, freeze all but last)
    model = build_model(X_train.shape)
    model.set_weights(fed_weights)
    for layer in model.layers[:-1]:
        layer.trainable = False
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    retrain_hist_bce = []
    retrain_hist_acc = []
    loss, acc = model.evaluate(X_test, y_test, verbose=0)
    retrain_hist_bce.append(loss)
    retrain_hist_acc.append(acc)
    for e in range(1, args.rounds+1):
        model.fit(X_train, y_train, epochs=1, batch_size=32, verbose=0)
        loss, acc = model.evaluate(X_test, y_test, verbose=0)
        retrain_hist_bce.append(loss)
        retrain_hist_acc.append(acc)
        print(f"[FedAvg retrain last-layer][Epoch {e:02d}] BCE={loss:.6f}, Accuracy={acc:.4f}")
    # Model 2: Train from scratch (no loaded weights, all layers trainable)
    scratch_model = build_model(X_train.shape)
    scratch_hist_bce = []
    scratch_hist_acc = []
    # Add initial loss/acc before any training for scratch model
    loss, acc = scratch_model.evaluate(X_test, y_test, verbose=0)
    scratch_hist_bce.append(loss)
    scratch_hist_acc.append(acc)
    for e in range(1, args.rounds+1):
        scratch_model.fit(X_train, y_train, epochs=1, batch_size=32, verbose=0)
        loss, acc = scratch_model.evaluate(X_test, y_test, verbose=0)
        scratch_hist_bce.append(loss)
        scratch_hist_acc.append(acc)
        print(f"[Scratch retrain][Epoch {e:02d}] BCE={loss:.6f}, Accuracy={acc:.4f}")
    return retrain_hist_bce, retrain_hist_acc, scratch_hist_bce, scratch_hist_acc


def main() -> None:
    if args.compare_dataset:
        data = data_loading_compare()
    else:
        data = data_loading()
    if args.role == "server":
        # Local-only
        for cid, (Xi, yi) in enumerate(data['client_slices']):
            m = build_model(Xi.shape)
            local_hist_bces[cid] = []
            local_hist_accs[cid]   = []
            # Evaluate and append initial loss/acc before any training
            loss, acc = m.evaluate(data['X_test'], data['y_test'], verbose=0)
            local_hist_bces[cid].append(loss)
            local_hist_accs[cid].append(acc)
            print(f"[Local][Client {cid}][Epoch 00] BCE={loss:.6f}, Accuracy={acc:.4f}")
            for e in range(1,args.rounds+1):
                if args.epoch_variance:
                    if (e-1) % (cid + 1) == 0:
                        m.fit(Xi,yi,epochs=1,batch_size=32,verbose=0)
                else:
                    history = m.fit(Xi, yi, epochs=1, batch_size=32, verbose=0)
                    train_loss = history.history['loss'][0] if 'loss' in history.history else None
                    train_acc = history.history['accuracy'][0] if 'accuracy' in history.history else None
                    if train_loss is not None and train_acc is not None:
                        print(f"[Local][Client {cid}][Epoch {e:02d}] Train BCE={train_loss:.6f}, Accuracy={train_acc:.4f}")
                loss,acc=m.evaluate(data['X_test'],data['y_test'],verbose=0)
                local_hist_bces[cid].append(loss)
                local_hist_accs[cid].append(acc)
                print(f"[Local][Client {cid}][Epoch {e:02d}] BCE={loss:.6f}, Accuracy={acc:.4f}")
        # Centralized best-epoch
        m_c = build_model(data['X_train_all'].shape)
        # Evaluate and append initial loss/acc before any training
        loss, acc = m_c.evaluate(data['X_test'], data['y_test'], verbose=0)
        central_hist_bce.append(loss)
        central_hist_acc.append(acc)
        fed_bces.append(loss)
        fed_accs.append(acc)
        print(f"[Central][Epoch 00] BCE={loss:.6f}, Accuracy={acc:.4f}")
        for e in range(1,args.rounds+1):
             m_c.fit(data['X_train_all'], data['y_train_all'], epochs=1, batch_size=32, verbose=0)
             loss, acc = m_c.evaluate(data['X_test'], data['y_test'], verbose=0)       
             central_hist_bce.append(loss)
             central_hist_acc.append(acc)
             print(f"[Central][Epoch {e:02d}] BCE={loss:.6f}, Accuracy={acc:.4f}")  

        # Federated
        print("\nStarting Federated Learning …")
        # Use patched federated server to save weights
        start_federated((data['X_test'], data['y_test']), num_rounds=args.rounds)

        retrain_hist_bce, retrain_hist_acc, scratch_hist_bce, scratch_hist_acc = None, None, None, None
        if args.retrain:
            global _best_acc_weights
            retrain_hist_bce, retrain_hist_acc, scratch_hist_bce, scratch_hist_acc = retrain_last_layer(_best_acc_weights, data)

        # Plot
        print("\n=== FINAL TEST ERRORS (round 50) ===")
        print(f"Centralised      •  BCE={central_hist_bce[-1]:.6f}  Accuracy={central_hist_acc[-1]:.6f}")
        for cid in range(NUM_CLIENTS):
            print(f"Local client {cid} •  BCE={local_hist_bces[cid][-1]:.6f}  Accuracy={local_hist_accs[cid][-1]:.6f}")
        print(f"FedAvg           •  BCE={fed_bces[-1]:.6f}  Accuracy={fed_accs[-1]:.6f}")
        if retrain_hist_bce is not None:
            print(f"FedAvg retrain last-layer •  BCE={retrain_hist_bce[-1]:.6f}  Accuracy={retrain_hist_acc[-1]:.6f}")
        if scratch_hist_bce is not None:
            print(f"FedAvg retrain from scratch •  BCE={scratch_hist_bce[-1]:.6f}  Accuracy={scratch_hist_acc[-1]:.6f}")
        print("\n=== BEST-EPOCH ERRORS (lowest loss anywhere in 50 rounds) ===")

        # Centralised
        best_c_idx = int(np.argmin(central_hist_bce))
        print(f"Centralised      •  Epoch {best_c_idx+1:02d}  "
         f"BCE={central_hist_bce[best_c_idx]:.6f}  "
         f"Accuracy={central_hist_acc [best_c_idx]:.6f}")

        # Locals
        for cid in range(NUM_CLIENTS):
            best_i = int(np.argmin(local_hist_bces[cid]))
            print(  f"Local client {cid} •  Epoch {best_i+1:02d}"
                    f"BCE={local_hist_bces[cid][best_i]:.6f}"
                    f"Accuracy={local_hist_accs  [cid][best_i]:.6f}")

        # FedAvg
        best_f_idx = int(np.argmin(fed_bces))
        print(  f"FedAvg           •  Round {best_f_idx+1:02d}"
                f"BCE={fed_bces[best_f_idx]:.6f}"
                f"Accuracy={fed_accs  [best_f_idx]:.6f}")

        # FedAvg+SP500 last-layer
        if retrain_hist_bce is not None:
            best_r_idx = int(np.argmin(retrain_hist_bce))
            print(f"FedAvg retrain last-layer •  Epoch {best_r_idx+1:02d}  "
                  f"BCE={retrain_hist_bce[best_r_idx]:.6f}  "
                  f"Accuracy={retrain_hist_acc[best_r_idx]:.6f}")
        if scratch_hist_bce is not None:
            best_s_idx = int(np.argmin(scratch_hist_bce))
            print(f"FedAvg retrain from scratch •  Epoch {best_s_idx+1:02d}  "
                  f"BCE={scratch_hist_bce[best_s_idx]:.6f}  "
                  f"Accuracy={scratch_hist_acc[best_s_idx]:.6f}")

        plot_paper_figure(local_hist_bces, local_hist_accs,
                  fed_bces, fed_accs,
                  central_hist_bce, central_hist_acc,
                  retrain_hist_bce, retrain_hist_acc,
                  scratch_hist_bce,scratch_hist_acc)
    else:
        if not (0 <= args.cid < NUM_CLIENTS):
            sys.exit("Error: cid out of range.")
        start_client(args.cid, data['client_slices'][args.cid], (data['X_test'], data['y_test']))

if __name__ == "__main__":
    main()
