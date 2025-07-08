# forecast.py
# Workflow
# Train LSTM on past 4-week sequences →
# Use last available 4 weeks →
# Predict week 5 sales →
# Save prediction
import numpy as np
import pandas as pd
from preprocess import load_and_prepare_data
from lstm_model import create_lstm_model
from sklearn.preprocessing import MinMaxScaler

def create_sequences(data, window_size):
    X = []
    for i in range(window_size, len(data)):
        X.append(data[i-window_size:i])
    return np.array(X)

def forecast_next_week(window_size=4, epochs=30):
    df_weekly = load_and_prepare_data(save_cleaned=False)
    forecast_results = []

    for sku in df_weekly.columns:
        series = df_weekly[sku].values.reshape(-1, 1)

        if len(series) < window_size + 10:
            continue

        # Scale
        scaler = MinMaxScaler()
        scaled = scaler.fit_transform(series)

        # Prepare training data
        X, y = [], []
        for i in range(window_size, len(scaled)):
            X.append(scaled[i-window_size:i])
            y.append(scaled[i])
        X, y = np.array(X), np.array(y)

        if len(X) < 10:
            continue

        # Reshape
        X = X.reshape((X.shape[0], X.shape[1], 1))

        try:
            # Train model
            model = create_lstm_model((window_size, 1))
            model.fit(X, y, epochs=epochs, batch_size=16, verbose=0)

            # Use last `window_size` weeks to forecast next
            last_seq = scaled[-window_size:]
            last_seq = last_seq.reshape((1, window_size, 1))
            forecast_scaled = model.predict(last_seq)
            forecast_units = scaler.inverse_transform(forecast_scaled)[0][0]

            forecast_results.append({
                'SKU': sku,
                'Forecast_Units_Next_Week': round(forecast_units)
            })
        except Exception as e:
            print(f"⚠️ Forecast failed for {sku}: {e}")
            continue

    # Save forecast
    forecast_df = pd.DataFrame(forecast_results)
    forecast_df.to_csv('data/weekly_forecast.csv', index=False)
    print("📈 Weekly forecast saved to: data/weekly_forecast.csv")

if __name__ == '__main__':
    forecast_next_week()
