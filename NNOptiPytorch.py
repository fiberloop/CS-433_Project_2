import optuna
import pickle
import numpy as np
import pandas as pd

from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from classes import MLP, CustomTorchDataset as npdata   # PyTorch MLP + Dataset


# ---------------------------------------------------------
# 1. Load and preprocess data
# ---------------------------------------------------------

with open("dataset/dataset.pkl", "rb") as f:
    dataset = pickle.load(f)

print("Type of dataset:", type(dataset))
data = pd.DataFrame(dataset)
print("TYPE:", type(data["plies"].iloc[0]))
print("CONTENT:", data["plies"].iloc[0])
print("LENGTH (if list):", len(data["plies"].iloc[0]) if hasattr(data["plies"].iloc[0], "__len__") else None)


print(np.array(data).shape)  # (1_000_000, 17)

print(data.columns.tolist())

eps_global = data["eps_global"]
print(np.array(eps_global).shape)  # (1_000_000,)

eps_df = pd.DataFrame(
    eps_global.tolist(),
    columns=[f"eps{i}" for i in range(1, 7)]
)

print(eps_df.shape)
print(eps_df.head())

plies_cols = [
    "plies.0.0.FI_ft", "plies.0.0.FI_fc",
    "plies.0.0.FI_mt", "plies.0.0.FI_mc",
    "plies.45.0.FI_ft", "plies.45.0.FI_fc",
    "plies.45.0.FI_mt", "plies.45.0.FI_mc",
    "plies.90.0.FI_ft", "plies.90.0.FI_fc",
    "plies.90.0.FI_mt", "plies.90.0.FI_mc",
    "plies.-45.0.FI_ft", "plies.-45.0.FI_fc",
    "plies.-45.0.FI_mt", "plies.-45.0.FI_mc",
]
1
# Flatten the nested FI dictionaries
plies_flat = pd.json_normalize(data["plies"])

# Rename columns from "0.0.FI_ft" → "plies.0.0.FI_ft"
plies_flat.columns = [f"plies.{col}" for col in plies_flat.columns]

print(plies_flat.head())
print(plies_flat.shape)


data_df = pd.concat([eps_df, plies_flat], axis=1)
print(data_df.shape)
print(data_df.head())

#x = data_df.values  # features (numpy)
# Use only the first 6 features (eps1-eps6)

x = eps_df.values

# ---------------------------------------------------------
# 2. Build labels y (0..16)
# ---------------------------------------------------------

fi_cols = plies_cols
F = data_df[fi_cols].values  # (N, 16)

mask_failure = (F >= 1)
has_failure = mask_failure.any(axis=1)

F_valid = F.copy()
F_valid[F_valid < 1] = -np.inf

y = np.zeros(len(F), dtype=int)
max_FI = F_valid[has_failure].argmax(axis=1) + 1
y[has_failure] = max_FI

print(y[:10])
print(y.shape)

# class counts (from your earlier computation)
counts = {
    0: 209953,
    1: 87948,
    2: 111527,
    3: 51882,
    4: 57206,
    5: 29708,
    6: 23705,
    7: 20207,
    8: 12997,
    9: 87721,
    10: 111797,
    11: 51575,
    12: 56978,
    13: 29464,
    14: 24010,
    15: 20295,
    16: 13027,
}

N = data_df.shape[0]
K = len(fi_cols) + 1

# class_weights as dict for all classes 0..16
class_weights = {c: N / (K * n) for c, n in counts.items()}


# ---------------------------------------------------------
# 3. Train-test split + scaling
# ---------------------------------------------------------

x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    test_size=0.2,
    random_state=0,
    stratify=y,
)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)


# ---------------------------------------------------------
# 4. Training & evaluation helpers (pure PyTorch)
# ---------------------------------------------------------

def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0.0

    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)

        optimizer.zero_grad()
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * xb.size(0)

    return total_loss / len(loader.dataset)


def eval_f1_macro(model, loader):
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            logits = model(xb)
            preds = logits.argmax(dim=1)

            all_preds.append(preds.cpu().numpy())
            all_targets.append(yb.cpu().numpy())

    y_pred = np.concatenate(all_preds)
    y_true = np.concatenate(all_targets)
    return f1_score(y_true, y_pred, average="macro")


# ---------------------------------------------------------
# 5. Optuna objective (PyTorch MLP)
# ---------------------------------------------------------

def make_objective(train_dataset, test_dataset, class_weights_dict, input_dim, num_classes):
    # convert class_weights dict → tensor ordered by class 0..num_classes-1
    weight_list = [class_weights_dict[c] for c in range(num_classes)]
    weight_tensor = torch.tensor(weight_list, dtype=torch.float32, device=device)

    def objective(trial):
        # Hyperparameters
        hidden1 = trial.suggest_int("hidden1", 32, 512)
        hidden2 = trial.suggest_int("hidden2", 32, 512)
        lr = trial.suggest_loguniform("learning_rate", 1e-4, 1e-2)
        wd = trial.suggest_loguniform("weight_decay", 1e-6, 1e-2)
        batch_size = trial.suggest_categorical("batch_size", [128, 256, 512])
        num_epochs = trial.suggest_int("epochs", 5, 15)

        # Dataloaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        # Model (PyTorch)
        model = MLP(input_dim, hidden1, hidden2, num_classes).to(device)
        criterion = nn.CrossEntropyLoss(weight=weight_tensor)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

        best_f1 = 0.0
        for epoch in range(num_epochs):
            train_one_epoch(model, train_loader, optimizer, criterion)
            f1 = eval_f1_macro(model, test_loader)

            trial.report(f1, epoch)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

            best_f1 = max(best_f1, f1)

        return best_f1

    return objective


def run_optuna(train_dataset, test_dataset, class_weights, input_dim, num_classes, n_trials=30):
    objective = make_objective(train_dataset, test_dataset, class_weights, input_dim, num_classes)
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)

    print("Best params:", study.best_params)
    return study.best_params


# ---------------------------------------------------------
# 6. Build PyTorch datasets & launch Optuna
# ---------------------------------------------------------

train_dataset = npdata(x_train, y_train)
test_dataset = npdata(x_test, y_test)

best = run_optuna(
    train_dataset,
    test_dataset,
    class_weights,
    input_dim=x_train.shape[1],
    num_classes=17,
    n_trials=50,
)

hidden1 = best["hidden1"]
hidden2 = best["hidden2"]
eta = best["learning_rate"]
weightdecay = best["weight_decay"]
batch_size = best["batch_size"]
max_iter = best["epochs"]

print("Best hyperparameters:", best)
