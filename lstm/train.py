# train.py
import numpy as np
import pandas as pd
from preprocess import load_and_prepare_data
from lstm_model import create_lstm_model
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings("ignore")

def create_sequences(data, window_size):
    X, y = [], []
    for i in range(window_size, len(data)):
        X.append(data[i-window_size:i])
        y.append(data[i])
    return np.array(X), np.array(y)

def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    non_zero_idx = y_true != 0
    y_true = y_true[non_zero_idx]
    y_pred = y_pred[non_zero_idx]
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100 if len(y_true) > 0 else np.nan

def train_all_skus(window_size=4):
    df_weekly = load_and_prepare_data()
    results = []

    for sku in df_weekly.columns:
        series = df_weekly[sku].values.reshape(-1, 1)

        # Skip SKU if there’s too little data
        if len(series) < window_size + 10:
            continue

        scaler = MinMaxScaler()
        scaled = scaler.fit_transform(series)

        X, y = create_sequences(scaled, window_size)
        if len(X) < 10:
            continue

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
        X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
        X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

        try:
            model = create_lstm_model((window_size, 1))
            model.fit(X_train, y_train, epochs=30, batch_size=16, verbose=0)

            preds = model.predict(X_test)
            y_test_real = scaler.inverse_transform(y_test.reshape(-1, 1))
            preds_real = scaler.inverse_transform(preds)

            mae = mean_absolute_error(y_test_real, preds_real)
            rmse = np.sqrt(mean_squared_error(y_test_real, preds_real))
            mape = mean_absolute_percentage_error(y_test_real, preds_real)
            accuracy = 100 - mape if mape is not np.nan else np.nan

            results.append({
                'SKU': sku,
                'MAE': round(mae, 2),
                'RMSE': round(rmse, 2),
                'MAPE (%)': round(mape, 2),
                'Accuracy (%)': round(accuracy, 2)
            })
        except Exception as e:
            print(f"⚠️ Failed for SKU {sku}: {e}")
            continue

    result_df = pd.DataFrame(results)
    result_df.to_csv('data/lstm_results_by_sku.csv', index=False)
    print("📊 Results saved to 'data/lstm_results_by_sku.csv'")

if __name__ == '__main__':
    train_all_skus()


# Main Function: train_all_skus(window_size=4)

# Step-by-step:
# Loads cleaned weekly sales data from preprocess.py
# For each SKU:
# Gets time series: e.g., [22, 35, 10, 0, 5, 11, ...]
# Scales data with MinMaxScaler
# Splits data into overlapping sequences:
# X = sequences of 4 weeks
# y = the 5th week's sales
# Splits into training & test sets
# Trains LSTM model for 30 epochs (default)
# Predicts on test set → inverse transforms

# Calculates:
# MAE: Mean Absolute Error
# RMSE: Root Mean Squared Error
# MAPE: % Error
# Accuracy = 100 - MAPE
# Saves results to data/lstm_results_by_sku.csv

# Why This Works:
# The sliding window of 4 steps allows the model to learn patterns over time
# MinMaxScaler helps neural networks converge faster (values between 0–1)
# LSTM retains sequence memory better than standard dense networks