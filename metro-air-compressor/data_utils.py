import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def load_and_preprocess_data():
    df = pd.read_csv('MetroPT3(AirCompressor).csv')
    df['timestamp'] = pd.to_datetime(df['timestamp'], format='%Y-%m-%d %H:%M:%S')

    failure_dates = pd.to_datetime([
        '2020-04-18', '2020-04-29', '2020-05-13', '2020-05-18', '2020-05-25',
        '2020-06-15', '2020-06-22', '2020-07-07', '2020-07-13', '2020-07-27'
    ])
    df['anomaly'] = 0
    for failure in failure_dates:
        start = failure - pd.Timedelta(hours=24)
        mask = (df['timestamp'] >= start) & (df['timestamp'] < failure)
        df.loc[mask, 'anomaly'] = 1

    return df

def create_sequences(X, y, timestamps, seq_len=10):
    """
    Converts flat time series into overlapping sequences.
    """
    Xs, ys, ts = [], [], []
    for i in range(len(X) - seq_len):
        Xs.append(X[i:i+seq_len])
        ys.append(y[i+seq_len-1])  # assign label of last timestep
        ts.append(timestamps[i+seq_len-1])
    return np.array(Xs), np.array(ys), np.array(ts)

def get_training_testing_data(df, train_end_date, seq_len=10):
    digital_cols = [
        'COMP', 'DV_eletric', 'Towers', 'MPG', 'LPS', 'Pressure_switch', 'Oil_level', 'Caudal_impulses'
    ]
    train_df = df[(df['timestamp'] < train_end_date) & (df['anomaly'] == 0)]
    test_df = df[df['timestamp'] >= train_end_date]

    scaler = MinMaxScaler(feature_range=(-1, 1))
    X_train_scaled = scaler.fit_transform(train_df[digital_cols])
    X_test_scaled = scaler.transform(test_df[digital_cols])

    X_train_seq, _, _ = create_sequences(X_train_scaled, train_df['anomaly'].values, train_df['timestamp'].values, seq_len)
    X_test_seq, y_test_seq, ts_seq = create_sequences(X_test_scaled, test_df['anomaly'].values, test_df['timestamp'].values, seq_len)

    return X_train_seq, X_test_seq, y_test_seq, ts_seq
