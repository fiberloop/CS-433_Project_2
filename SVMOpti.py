import pickle
import numpy as np
import pandas as pd

from sklearn.metrics import accuracy_score, f1_score, recall_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline


# =========================
# Repro
# =========================
SEED = 0
np.random.seed(SEED)


# =========================
# Load dataset
# =========================
with open("dataset/dataset.pkl", "rb") as f:
    dataset = pickle.load(f)

data = pd.json_normalize(dataset)
print(np.array(data).shape)  # (1_000_000, 17)
print(data.columns.tolist())

eps_global = data["eps_global"]
print(np.array(eps_global).shape)  # (1_000_000,)

eps_df = pd.DataFrame(
    eps_global.tolist(),
    columns=[f"eps{i}" for i in range(1, 7)]
)
print(eps_df.shape)  # (1_000_000, 6)
print(eps_df.head())


plies_cols = [
    "plies.0.0.FI_ft", "plies.0.0.FI_fc", "plies.0.0.FI_mt", "plies.0.0.FI_mc",
    "plies.45.0.FI_ft", "plies.45.0.FI_fc", "plies.45.0.FI_mt", "plies.45.0.FI_mc",
    "plies.90.0.FI_ft", "plies.90.0.FI_fc", "plies.90.0.FI_mt", "plies.90.0.FI_mc",
    "plies.-45.0.FI_ft", "plies.-45.0.FI_fc", "plies.-45.0.FI_mt", "plies.-45.0.FI_mc",
]
plies_df = data[plies_cols]

data_df = pd.concat([eps_df, plies_df], axis=1)
print(data_df.shape)  # (1_000_000, 22)
print(data_df.head())


# =========================
# Features: ONLY eps (6 features)
# =========================
x = eps_df.values  # shape (N, 6)


# =========================
# Labels: 0 if no FI >= 1, else argmax(FI) + 1
# =========================
fi_cols = plies_cols  # fixed order (16 FI)
F = data_df[fi_cols].values  # (N, 16)

mask_failure = (F >= 1)
has_failure = mask_failure.any(axis=1)

F_valid = F.copy()
F_valid[F_valid < 1] = -np.inf

y = np.zeros(len(F), dtype=int)  # class 0 by default
y[has_failure] = F_valid[has_failure].argmax(axis=1) + 1  # 1..16

print(y[:10])
print(y.shape)  # (N,)


# =========================
# Class counts (computed from y)
# =========================
counts = {c: int((y == c).sum()) for c in range(17)}
print("Class counts:", counts)

N = len(y)
K = 17  # number of classes
class_weights = {c: N / (K * counts[c]) for c in range(K)}

# Normalize weights so mean = 1 (recommended)
w_arr = np.array(list(class_weights.values()), dtype=np.float32)
w_arr /= w_arr.mean()
class_weights = {c: float(w_arr[c]) for c in range(K)}

print("Mean weight:", np.mean(list(class_weights.values())))
print("Min/Max weight:", min(class_weights.values()), max(class_weights.values()))


# =========================
# (Optional) Subsample for speed
# Set n_sub = 1_000_000 to use full dataset
# =========================
n_sub = 1_000_000
df_sub = eps_df.sample(n=n_sub, random_state=SEED)
x_sub = df_sub.values
y_sub = y[df_sub.index]

print("Using n_sub =", n_sub)


# =========================
# GridSearchCV with StratifiedKFold + Pipeline (no leakage)
# =========================
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("svc", LinearSVC(dual=False, max_iter=4000))
])

C_values = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000]
param_grid = {"svc__C": C_values}

grid = GridSearchCV(
    estimator=pipe,
    param_grid=param_grid,
    scoring="recall_macro",   # minimize FN across classes
    cv=cv,
    n_jobs=-1,
    verbose=1,
    refit=True
)

# sample weights for each sample in x_sub/y_sub
sample_weight_sub = np.array([class_weights[int(yi)] for yi in y_sub], dtype=np.float32)

grid.fit(x_sub, y_sub, svc__sample_weight=sample_weight_sub)

print("\nBest params:", grid.best_params_)
print("Best CV macro recall:", grid.best_score_)


# =========================
# (Optional) Evaluate the best model on the same data (NOT a test evaluation)
# For a real final score: create a held-out test set and evaluate once.
# =========================
y_pred = grid.predict(x_sub)

recall = recall_score(y_sub, y_pred, average="macro", zero_division=0)
f1 = f1_score(y_sub, y_pred, average="macro")
acc = accuracy_score(y_sub, y_pred)

print("\n[On x_sub (not a held-out test)]")
print(f"Accuracy: {acc:.4f}")
print(f"Macro F1: {f1:.4f}")
print(f"Macro Recall: {recall:.4f}")
