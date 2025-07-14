# preprocessing.py

from imblearn.combine import SMOTEENN
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
from config import DATA_PATH, TEST_SIZE, RANDOM_STATE

def load_and_preprocess_data():
    df = pd.read_csv(DATA_PATH)
    df.drop_duplicates(inplace=True)

    X = df.drop("Fault_Type", axis=1)
    y = df["Fault_Type"]

    # Train-test split before resampling
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )

    # Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Apply SMOTEENN (resampling only training set)
    smote_enn = SMOTEENN(random_state=RANDOM_STATE)
    X_train_resampled, y_train_resampled = smote_enn.fit_resample(X_train_scaled, y_train)

    return X_train_resampled, X_test_scaled, y_train_resampled, y_test
