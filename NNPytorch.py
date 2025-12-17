import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler
from classes import CustomTorchDataset as npdata
from classes import MLP
import NNOptiPytorch as opti 
from sklearn.metrics import confusion_matrix

# 1. Load dataset
with open("dataset/dataset.pkl", "rb") as f:
    dataset = pickle.load(f)

data = pd.DataFrame(dataset)


eps_global = data["eps_global"]
eps_df = pd.DataFrame(eps_global.tolist(), columns=[f"eps{i}" for i in range(1, 7)])

plies_cols = [
    'plies.0.0.FI_ft', 'plies.0.0.FI_fc', 'plies.0.0.FI_mt', 'plies.0.0.FI_mc',
    'plies.45.0.FI_ft', 'plies.45.0.FI_fc', 'plies.45.0.FI_mt', 'plies.45.0.FI_mc',
    'plies.90.0.FI_ft', 'plies.90.0.FI_fc', 'plies.90.0.FI_mt', 'plies.90.0.FI_mc',
    'plies.-45.0.FI_ft', 'plies.-45.0.FI_fc', 'plies.-45.0.FI_mt', 'plies.-45.0.FI_mc'
]

# Flatten the nested FI dictionaries
plies_flat = pd.json_normalize(data["plies"])

# Rename columns from "0.0.FI_ft" → "plies.0.0.FI_ft"
plies_flat.columns = [f"plies.{col}" for col in plies_flat.columns]

print(plies_flat.head())
print(plies_flat.shape)

data_df = pd.concat([eps_df, plies_flat], axis=1)

# x = data_df.values
# Use only the first 6 features (eps1-eps6)

x = eps_df.values
# 2. Build target label y


F = data_df[plies_cols].values
mask_failure = (F >= 1)
has_failure = mask_failure.any(axis=1)

F_valid = F.copy()
F_valid[F_valid < 1] = -np.inf

y = np.zeros(len(F), dtype=int)
y[has_failure] = F_valid[has_failure].argmax(axis=1) + 1



# 3. class weighting to handle imbalanced dataset

counts = {
    0: 209953, 1: 87948, 2: 111527, 3: 51882, 4: 57206,
    5: 29708, 6: 23705, 7: 20207, 8: 12997, 9: 87721,
    10: 111797, 11: 51575, 12: 56978, 13: 29464, 14: 24010,
    15: 20295, 16: 13027
}

N = len(y)
num_classes = 17  # classes 0..16

class_weights_np = np.zeros(num_classes, dtype=np.float32)
for c in range(num_classes):
    class_weights_np[c] = N / (num_classes * counts[c])

# Normalize so mean = 1
class_weights_np = class_weights_np / class_weights_np.mean()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class_weights = torch.tensor(class_weights_np, dtype=torch.float32, device=device)
print("Class weights:", class_weights_np)


# 4. Train-test split + scaling

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=0, stratify=y
)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test  = scaler.transform(x_test)


# 5. PyTorch Dataset using classes.py & DataLoader

train_dataset = npdata(x_train, y_train)
test_dataset  = npdata(x_test, y_test)

batch_size = opti.batch_size
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


# 6. Define PyTorch MLP

input_dim = x_train.shape[1]  # 22 features
hidden1 = opti.hidden1
hidden2 = opti.hidden2

model = MLP(input_dim, hidden1, hidden2, num_classes).to(device)

criterion = nn.CrossEntropyLoss(weight=class_weights) #loss for classification multi-classes : only for training
#train with ADAM optimizer 
optimizer = torch.optim.Adam(model.parameters(), lr=opti.eta, weight_decay=opti.weightdecay) #L2 regularization included

# 7. Training loop

max_iter = opti.max_iter

for i in range(max_iter): #iterations through the entire training dataset
    model.train()
    total_loss = 0

    for xb, yb in train_loader: #mini-batch stochatic gradient descent
        xb = xb.to(device) 
        yb = yb.to(device)

        optimizer.zero_grad() # set gradients to zero
        logits = model(xb) #forward pass
        loss = criterion(logits, yb) #compute loss
        loss.backward() #backpropagation
        optimizer.step() #update weig
        total_loss += loss.item() * xb.size(0) # accumulate loss weighted by batch size

    epoch_loss = total_loss / len(train_loader.dataset) #mean over all samples
    print(f"Epoch {i+1}/{max_iter} — Loss = {epoch_loss:.4f}")


# 8. Test
model.eval() #evaluation mode
all_preds = []
all_targets = []

with torch.no_grad(): # no gradient computation
    for xb, yb in test_loader: 
        #mini-batch to avoid overflow of memory (GPU/CPU) 
        xb = xb.to(device)
        logits = model(xb) #forward pass
        preds = logits.argmax(dim=1).cpu().numpy() #predictions of the class index no proba 

        all_preds.append(preds)
        all_targets.append(yb.numpy())

y_pred = np.concatenate(all_preds)
y_true = np.concatenate(all_targets)

# Validation
f1 = f1_score(y_test, y_pred, average = "macro") # equally weighted average of classes
accuracy = accuracy_score(y_test, y_pred)
recall = recall_score(y_test, y_pred, average="macro", zero_division=0)
precision = precision_score(y_test, y_pred, average="macro", zero_division=0)




print(f"GLOBAL METRICS")
print(f"Accuracy: {accuracy:.4f}")
print(f"Macro F1: {f1:.4f}")
print(f"Macro Recall: {recall:.4f}")
print(f"Macro Precision: {precision:.4f}")

# Nombre de samples par classe (support)
unique, counts = np.unique(y_true, return_counts=True)

print(f"SAMPLES PER CLASS")
for u, c in zip(unique, counts):
    print(f"Classe {u}: {c} samples")

# Confusion matrix et métriques par classe
cm = confusion_matrix(y_true, y_pred)

print(f"DETAILED METRICS PER CLASS")
print(f"{'Class':<6} {'Support':<10} {'FN':<8} {'FP':<8} {'Precision':<12} {'Recall':<10} {'F1':<10}")

f1s = []
accuracies = []
for class_idx in range(17):
    TP = cm[class_idx, class_idx]
    FP = cm[:, class_idx].sum() - TP
    FN = cm[class_idx, :].sum() - TP
    support = (y_true == class_idx).sum()
    prec = TP / (TP + FP) if (TP + FP) > 0 else 0
    rec = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1_class = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
    acc_class = TP / support if support > 0 else 0
    print(f"{class_idx:<6} {support:<10} {FN:<8} {FP:<8} {prec:<12.4f} {rec:<10.4f} {f1_class:<10.4f}")
    f1s.append(f1_class)
    accuracies.append(acc_class)

# Moyennes F1 et accuracy par classe
print(f"AVERAGE PER-CLASS F1: {np.mean(f1s):.4f}")
print(f"AVERAGE PER-CLASS ACCURACY: {np.mean(accuracies):.4f}")

f1 = f1_score(y_true, y_pred, average="macro")
accuracy = accuracy_score(y_true, y_pred)

print("Accuracy:", accuracy)
print("F1 macro average :", f1)


