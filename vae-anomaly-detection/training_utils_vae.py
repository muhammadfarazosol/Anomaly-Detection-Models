# training_utils_vae.py

import torch
import torch.optim as optim
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import classification_report, f1_score
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from model_vae import VAE, vae_loss_function

# --- UPDATED: Function now accepts window and quantile as arguments ---
def run_iteration(X_train_scaled, X_test_scaled, y_test_true, test_timestamps, iteration_name, window, quantile):
    """
    Runs a full training and evaluation cycle for one VAE iteration with given hyperparameters.
    """
    print(f"\n----- Running Experiment: {iteration_name} | Window={window}, Quantile={quantile} -----")

    # --- 1. Model Training ---
    n_features = X_train_scaled.shape[1]
    train_tensor = torch.FloatTensor(X_train_scaled)
    train_loader = DataLoader(dataset=train_tensor, batch_size=256, shuffle=True)

    model = VAE(n_features=n_features)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    model.train()
    for epoch in range(15):
        for data in train_loader:
            img = data[0]
            recon_batch, mu, logvar = model(img)
            loss = vae_loss_function(recon_batch, img, mu, logvar)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # --- 2. Anomaly Detection ---
    model.eval()

    # STEP A: Determine Threshold from TRAINING data only
    with torch.no_grad():
        recon_train, _, _ = model(train_tensor)
        raw_train_errors = torch.mean((train_tensor - recon_train) ** 2, dim=1)

    train_error_df = pd.DataFrame({'error': raw_train_errors.numpy()})
    # Use the 'window' parameter from the experiment
    smoothed_train_errors = train_error_df['error'].rolling(window=window, min_periods=1, center=True).mean().values
    
    # Use the 'quantile' parameter from the experiment
    threshold = pd.Series(smoothed_train_errors).quantile(quantile)
    print(f"Anomaly Threshold determined: {threshold:.6f}")

    # STEP B: Evaluate on TEST data
    test_tensor = torch.FloatTensor(X_test_scaled)
    with torch.no_grad():
        recon_test, _, _ = model(test_tensor)
        raw_test_errors = torch.mean((test_tensor - recon_test) ** 2, dim=1)

    test_error_df = pd.DataFrame({'error': raw_test_errors.numpy()})
    # Use the same 'window' parameter
    smoothed_test_errors = test_error_df['error'].rolling(window=window, min_periods=1, center=True).mean().values
    
    y_pred = (smoothed_test_errors > threshold).astype(int)

    # We only care about the F1 score for the anomaly class (label 1)
    # pos_label=1 ensures we get the score for anomalies, not normal data
    anomaly_f1_score = f1_score(y_test_true, y_pred, pos_label=1)
    
    print(f"--- Anomaly F1-Score for this run: {anomaly_f1_score:.4f} ---")
    
    return anomaly_f1_score