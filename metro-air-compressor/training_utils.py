import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import classification_report, f1_score
import numpy as np
import pandas as pd
from model import LSTMAutoencoder

def run_iteration(X_train, X_test, y_test, timestamps, iteration_name):
    print(f"\n----- Starting Iteration: {iteration_name} -----")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Model Setup ---
    input_dim = X_train.shape[2]
    model = LSTMAutoencoder(input_dim).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    train_loader = DataLoader(TensorDataset(torch.FloatTensor(X_train)), batch_size=128, shuffle=True)

    # --- Training ---
    print("Training LSTM Autoencoder...")
    model.train()
    for epoch in range(20):
        for batch, in train_loader:
            batch = batch.to(device)
            recon = model(batch)
            loss = criterion(recon, batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    print("Training complete.")

    # --- Training Errors for Threshold ---
    model.eval()
    train_errors = []
    with torch.no_grad():
        for batch, in train_loader:
            batch = batch.to(device)
            recon = model(batch)
            error = torch.mean((recon - batch) ** 2, dim=(1, 2))
            train_errors.extend(error.cpu().numpy())
    train_errors = np.array(train_errors)

    # Threshold at 98th percentile
    threshold = np.percentile(train_errors, 98)
    print(f"Anomaly Threshold (98th percentile): {threshold:.6f}")

    # --- Test Errors ---
    test_tensor = torch.FloatTensor(X_test)
    test_loader = DataLoader(test_tensor, batch_size=256, shuffle=False)

    test_errors = []
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            recon = model(batch)
            error = torch.mean((recon - batch) ** 2, dim=(1, 2))
            test_errors.extend(error.cpu().numpy())
    test_errors = np.array(test_errors)

    # --- Optional: Smooth test errors ---
    smoothed_errors = pd.Series(test_errors).rolling(window=5, center=True, min_periods=1).mean().values

    # --- Persistence rule: Anomaly if 3 consecutive above threshold ---
    anomaly_flags = (smoothed_errors > threshold).astype(int)
    persistent_anomalies = np.zeros_like(anomaly_flags)

    count = 0
    for i in range(len(anomaly_flags)):
        if anomaly_flags[i] == 1:
            count += 1
        else:
            count = 0
        if count >= 3:
            persistent_anomalies[i] = 1

    # --- Classification Report ---
    print("\n--- Classification Report ---")
    print(classification_report(y_test, persistent_anomalies, target_names=["Normal", "Anomaly"]))

    return f1_score(y_test, persistent_anomalies)
