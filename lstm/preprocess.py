# preprocess.py
# Raw Rows → Aggregated Daily Sales → Weekly Time Series Per SKU
import pandas as pd
import os

def load_and_prepare_data(path='data/warehouse_3_dataset.csv', save_cleaned=True):
    # Load and parse dates
    df = pd.read_csv(path)
    df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)

    # Drop invalid rows
    df = df.dropna(subset=['SKU', 'Units Sold', 'Date'])
    df['Units Sold'] = pd.to_numeric(df['Units Sold'], errors='coerce').fillna(0)

    # ✅ Combine transactions: Aggregate daily sales by SKU
    df_grouped = df.groupby(['Date', 'SKU'])['Units Sold'].sum().reset_index()

    # Pivot for time series format (rows: dates, columns: SKUs)
    df_pivot = df_grouped.pivot(index='Date', columns='SKU', values='Units Sold').fillna(0)

    # Weekly resampling (you can change to 'D' for daily if needed)
    df_weekly = df_pivot.resample('W').sum()

    # Save cleaned data
    if save_cleaned:
        output_path = 'data/cleaned_weekly_sales.csv'
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df_weekly.to_csv(output_path)
        print(f"✅ Cleaned and aggregated weekly data saved at: {output_path}")
        print(f"🔎 Shape of cleaned data: {df_weekly.shape}")
        print(f"📅 Date Range: {df_weekly.index.min()} → {df_weekly.index.max()}")
    
    return df_weekly

if __name__ == '__main__':
    load_and_prepare_data()
# Raw Sales Data → Preprocessing → Weekly Time Series
#                ↓
#         LSTM Training (per SKU)
#                ↓
#          Model Evaluation (MAE, RMSE, etc.)
#                ↓
#          Future Forecasting (per SKU)
