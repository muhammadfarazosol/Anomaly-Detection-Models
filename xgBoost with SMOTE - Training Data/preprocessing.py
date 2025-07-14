# preprocessing.py

# SMOTE on training dataset only
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
from config import DATA_PATH, TEST_SIZE, RANDOM_STATE

def load_and_preprocess_data():
    df = pd.read_csv(DATA_PATH)
    df.drop_duplicates(inplace=True)

    X = df.drop("Fault_Type", axis=1)
    y = df["Fault_Type"]

    # Split first (no leakage)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )

    # Scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Apply SMOTE on training data only
    smote = SMOTE(random_state=RANDOM_STATE)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)

    return X_train_resampled, X_test_scaled, y_train_resampled, y_test