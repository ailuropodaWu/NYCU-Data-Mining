import pandas as pd
import numpy as np
import torch
import sys
import os
import csv

from tqdm import tqdm
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from model import VAE

sys.path.append('..')
from utils import save2csv

def loss_function(x, x_hat, mean, logvar):
    reproduction_loss = nn.functional.cross_entropy(x_hat, x)
    KLD = 0 # nn.functional.kl_div(x_hat, x, reduction='batchmean')
    return reproduction_loss

def reconstruction_error(model, data: torch.Tensor):
    model.eval()
    with torch.no_grad():
        data_reconstructed = model(data)[0]
        error = ((data - data_reconstructed) ** 2).sum(dim=1)
    return error

device = 'cuda' if torch.cuda.is_available() else 'cpu'
np.random.seed(0)

train_data = pd.read_csv('../data/train.csv')
val_data = pd.read_csv('../data/raw_data.csv')
test_data = pd.read_csv('../data/test.csv')

X_train = train_data.iloc[:, 1:].values
X_test = test_data.iloc[:, 1:].values
y_test = test_data.iloc[:, 0].values
labels = list(train_data.groupby('lettr').groups.keys())
print(labels)

# Standardize the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Convert data to PyTorch tensors
train_tensor = torch.tensor(X_train, dtype=torch.float32, device=device)
test_tensor = torch.tensor(X_test, dtype=torch.float32, device=device)

# Hyperparameters
input_dim = X_train.shape[1]
hidden_dim = 128
latent_dim = 16
lr = 1e-6
epochs = 5
batch_size = 64

root = "model"
model_info = f"epoch_{epochs}_batch_{batch_size}_lr{lr}_hidden{hidden_dim}_latent{latent_dim}"
os.makedirs(os.path.join(root, model_info), exist_ok=True)
writer = SummaryWriter(os.path.join('runs', model_info))

# Create DataLoader
train_loader = DataLoader(TensorDataset(train_tensor, train_tensor), batch_size=batch_size, shuffle=True)

# Model, loss function, and optimizer
print(f"Input Dim: {input_dim} / Device: {device}")

vae = VAE(input_dim, hidden_dim, latent_dim)
vae.to(device)
optimizer = torch.optim.Adam(vae.parameters(), lr=lr)
step = 0

# Train the VAE
for epoch in range(epochs):
    # TRAINING
    overall_loss = 0
    vae.train()
    for x_batch, _ in tqdm(train_loader, leave=False):
        optimizer.zero_grad()

        x_hat, mean, log_var = vae(x_batch)
        loss = loss_function(x_batch, x_hat, mean, log_var)
        
        overall_loss += loss.item()
            
        writer.add_scalar('Loss/train-step', loss, step)
        
        step += 1
        loss.backward()
        optimizer.step()
    writer.add_scalar('Loss/train-epoch', overall_loss, epoch)
    
    # VALIDATION   
    val_indices = np.random.randint(0, len(val_data), 1000)
    X_val = val_data.iloc[val_indices, 1:].values
    y_val = np.array([int(y not in labels) for y in val_data.iloc[val_indices, 0].values])
    X_val = scaler.fit_transform(X_val)
    val_tensor = torch.tensor(X_val, dtype=torch.float32, device=device)
    
    
    val_score = reconstruction_error(vae, val_tensor).cpu().detach().numpy()
    roc_auc = roc_auc_score(y_val, val_score)
    writer.add_scalar('Score/val', roc_auc, epoch)
    
    if (epoch+1) % 10 == 0:
        torch.save(vae.state_dict(), os.path.join(root, model_info, f"{epoch}.pt"))
    print(f'Epoch {epoch+1}, Loss: {loss.item()}, Validation Score: {roc_auc}')

# TEST
test_score = reconstruction_error(vae, test_tensor).cpu().detach().numpy()
y_test = np.array([int(y not in labels) for y in y_test])
roc_auc = roc_auc_score(y_test, test_score)
print(f"ROC-AUC: {roc_auc}")
file_path = f'{model_info}_result.csv'
save2csv(test_score, file_path)
