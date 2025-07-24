# data_utils_vae.py

import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def load_and_preprocess_data():
    """
    Loads the MetroPT3(AirCompressor).csv dataset, parses timestamps, and creates anomaly labels.
    """
    try:
        df = pd.read_csv('MetroPT3(AirCompressor).csv')
        # Using the date format from your provided data snippet
        df['timestamp'] = pd.to_datetime(df['timestamp'], format='%d/%m/%Y %H:%M')
    except FileNotFoundError:
        print("\nERROR: 'MetroPT3(AirCompressor).csv' not found in the project directory.")
        exit()
    except ValueError:
        print("\nERROR: Timestamp format in CSV does not match '%d/%m/%Y %H:%M'.")
        print("Switching to ISO 8601 format ('%Y-%m-%d %H:%M:%S')...")
        df = pd.read_csv('MetroPT3(AirCompressor).csv')
        df['timestamp'] = pd.to_datetime(df['timestamp'], format='%Y-%m-%d %H:%M:%S')


    failure_dates = [
        '2020-04-18', '2020-04-29', '2020-05-13', '2020-05-18', '2020-05-25',
        '2020-06-15', '2020-06-22', '2020-07-07', '2020-07-13', '2020-07-27'
    ]
    failure_dates = pd.to_datetime(failure_dates)

    df['anomaly'] = 0
    for failure_date in failure_dates:
        pre_failure_window_start = failure_date - pd.Timedelta(hours=24)
        mask = (df['timestamp'] >= pre_failure_window_start) & (df['timestamp'] < failure_date)
        df.loc[mask, 'anomaly'] = 1

    return df

def get_training_testing_data(df, train_end_date):
    """
    Splits and scales data. VAEs with sigmoid output work best with data scaled 0-1.
    """
    digital_cols = [
        'COMP', 'DV_eletric', 'Towers', 'MPG', 'LPS', 'Pressure_switch', 'Oil_level', 'Caudal_impulses'
    ]

    train_df = df[(df['timestamp'] < train_end_date) & (df['anomaly'] == 0)]
    test_df = df[df['timestamp'] >= train_end_date]

    # VAEs use a sigmoid activation in the final layer, so data MUST be scaled 0-1.
    scaler = MinMaxScaler(feature_range=(0, 1))

    X_train_scaled = scaler.fit_transform(train_df[digital_cols])
    X_test_scaled = scaler.transform(test_df[digital_cols])
    
    y_test_true = test_df['anomaly'].values

    return X_train_scaled, X_test_scaled, y_test_true, test_df['timestamp']