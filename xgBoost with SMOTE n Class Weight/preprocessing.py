# preprocessing.py

from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
import pandas as pd
import numpy as np
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

    # Compute class weights on original (non-resampled) y_train
    classes = np.unique(y_train)
    weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_train)
    class_weight_dict = {int(c): w for c, w in zip(classes, weights)}

    return X_train_resampled, X_test_scaled, y_train_resampled, y_test, class_weight_dict
