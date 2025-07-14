# preprocessing.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from config import DATA_PATH, TEST_SIZE, RANDOM_STATE

from imblearn.over_sampling import SMOTE

def load_and_preprocess_data():
    df = pd.read_csv(DATA_PATH)

    # Drop duplicates
    df.drop_duplicates(inplace=True)

    # Separate features and labels
    X = df.drop("Fault_Type", axis=1)
    y = df["Fault_Type"]

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Balance classes using SMOTE
    smote = SMOTE(random_state=RANDOM_STATE)
    X_resampled, y_resampled = smote.fit_resample(X_scaled, y)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_resampled, y_resampled, test_size=TEST_SIZE, 
        random_state=RANDOM_STATE, stratify=y_resampled
    )

    return X_train, X_test, y_train, y_test
