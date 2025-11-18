import pickle
import csv
import numpy as np
import pandas as pd


with open("dataset/dataset.pkl", "rb") as f:
    dataset = pickle.load(f)

data = pd.json_normalize(dataset)
print(np.array(data).shape) # (1 000 000, 17)

print(data.columns.tolist())
"""['eps_global', 'plies.0.0.FI_ft', 'plies.0.0.FI_fc', 
'plies.0.0.FI_mt', 'plies.0.0.FI_mc', 'plies.45.0.FI_ft', 
'plies.45.0.FI_fc', 'plies.45.0.FI_mt', 'plies.45.0.FI_mc',
 'plies.90.0.FI_ft', 'plies.90.0.FI_fc', 'plies.90.0.FI_mt',
 'plies.90.0.FI_mc', 'plies.-45.0.FI_ft', 'plies.-45.0.FI_fc', 
 'plies.-45.0.FI_mt', 'plies.-45.0.FI_mc']"""

eps_global = data['eps_global']
print(np.array(eps_global).shape) # output (1 000 000,)

eps_df = pd.DataFrame(
    eps_global.tolist(), 
    columns=[f"eps{i}" for i in range(1, 7)]
)

print(eps_df.shape)  # output (1_000_000, 6)
print(eps_df.head())

plies_cols = ['plies.0.0.FI_ft', 'plies.0.0.FI_fc', 
'plies.0.0.FI_mt', 'plies.0.0.FI_mc', 'plies.45.0.FI_ft', 
'plies.45.0.FI_fc', 'plies.45.0.FI_mt', 'plies.45.0.FI_mc',
 'plies.90.0.FI_ft', 'plies.90.0.FI_fc', 'plies.90.0.FI_mt',
 'plies.90.0.FI_mc', 'plies.-45.0.FI_ft', 'plies.-45.0.FI_fc', 
 'plies.-45.0.FI_mt', 'plies.-45.0.FI_mc']

plies_df = data[plies_cols]

data_df = pd.concat([eps_df, plies_df], axis=1)

print(data_df.shape)    # (1_000_000, 22)
print(data_df.head())

x = data_df.values

# Build label for the all dataset

# List of the 16 FI feature columns (in a fixed order)
fi_cols = [
    'plies.0.0.FI_ft', 'plies.0.0.FI_fc', 'plies.0.0.FI_mt', 'plies.0.0.FI_mc',
    'plies.45.0.FI_ft', 'plies.45.0.FI_fc', 'plies.45.0.FI_mt', 'plies.45.0.FI_mc',
    'plies.90.0.FI_ft', 'plies.90.0.FI_fc', 'plies.90.0.FI_mt', 'plies.90.0.FI_mc',
    'plies.-45.0.FI_ft', 'plies.-45.0.FI_fc', 'plies.-45.0.FI_mt', 'plies.-45.0.FI_mc',
]

# Extract FI values, so extratc the 16 FI columns and their corresponding rows
F = data_df[fi_cols].values  # (1_000_000, 16)

# Define label: 0 if no FI > 1, otherwise argmax + 1
mask_failure = (F > 1) 
has_failure = mask_failure.any(axis=1)  #  (N,), True if at least one failure for each line

F_valid = F.copy()
F_valid[F_valid < 1] = -np.inf

y = np.zeros(len(F), dtype=int)  # class 0 by default
max_FI = F_valid[has_failure].argmax(axis=1) + 1  # class 1..16
y[has_failure] = max_FI #replacing at failure row index the correspodning class


print(y[:10])
print(y.shape) #(1_000_000,)