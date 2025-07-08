import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from preprocess import load_and_preprocess
from lstm_autoencoder import create_lstm_autoencoder

# Load and preprocess data
df = load_and_preprocess("data/warehouse_3_dataset.csv")
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df)

# Parameters
sequence_length = 14
forecast_horizon = 1  # only 1 day

# Create sequences and labels
def create_sequences(data, seq_length, horizon):
    X, y = [], []
    for i in range(len(data) - seq_length - horizon + 1):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length:i+seq_length+horizon])
    return np.array(X), np.array(y)

X, y = create_sequences(scaled_data, sequence_length, forecast_horizon)

# Train-test split (70-30)
train_size = int(0.7 * len(X))
X_train, y_train = X[:train_size], y[:train_size]
X_test, y_test = X[train_size:], y[train_size:]

# Model
model = create_lstm_autoencoder(sequence_length, scaled_data.shape[1])
model.fit(X_train, X_train, epochs=30, batch_size=32, validation_split=0.1)

# Predict next 1 day using last available window
last_window = scaled_data[-sequence_length:]
input_seq = np.expand_dims(last_window, axis=0)
predicted_scaled = model.predict(input_seq)[0, -1:]  # Get last timestep

# Convert back to original scale
predicted_actual = scaler.inverse_transform(predicted_scaled)
true_actual = scaler.inverse_transform(y[-1])  # the real next-day sales

# Accuracy
mae = mean_absolute_error(true_actual, predicted_actual)
mse = mean_squared_error(true_actual, predicted_actual)

# MAPE and Accuracy (%)
mape = np.mean(np.abs((true_actual - predicted_actual) / true_actual)) * 100
accuracy_percent = 100 - mape

print(f"  Mean Absolute Percentage Error (MAPE): {mape:.2f}%")
print(f"  Accuracy: {accuracy_percent:.2f}%")

# Show Results
categories = df.columns.tolist()
prediction_date = df.index[-1] + pd.Timedelta(days=1)

print(f"\n📅 Predicted Sales for {prediction_date.date()}:\n")
for category, value in zip(categories, predicted_actual[0]):
    print(f"  {category}: {int(value)} units")

print(f"\n✅ Accuracy Metrics:")
print(f"  Mean Absolute Error (MAE): {mae:.2f}")
print(f"  Mean Squared Error (MSE): {mse:.2f}")

# Optional Plot
plt.figure(figsize=(8, 5))
plt.bar(categories, predicted_actual[0], color='skyblue', label="Predicted")
plt.bar(categories, true_actual[0], color='orange', alpha=0.6, label="Actual")
plt.xticks(rotation=45)
plt.title(f"Sales Prediction vs Actual for {prediction_date.date()}")
plt.ylabel("Units Sold")
plt.legend()
plt.tight_layout()
plt.show()
