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

# # Updated preprocessing.py without SMOTE 69% accuracy 
# import pandas as pd
# import numpy as np
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# from sklearn.utils.class_weight import compute_class_weight
# from config import DATA_PATH, TEST_SIZE, RANDOM_STATE

# def load_and_preprocess_data():
#     # Load data
#     df = pd.read_csv(DATA_PATH)
#     df.drop_duplicates(inplace=True)

#     # Split features and target
#     X = df.drop("Fault_Type", axis=1)
#     y = df["Fault_Type"]

#     # Split data before any resampling
#     X_train, X_test, y_train, y_test = train_test_split(
#         X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
#     )

#     # Scale features
#     scaler = StandardScaler()
#     X_train_scaled = scaler.fit_transform(X_train)
#     X_test_scaled = scaler.transform(X_test)

#     # Compute class weights
#     class_weights = compute_class_weight(
#     class_weight='balanced',
#     classes=np.array(sorted(y_train.unique())),
#     y=y_train)
#     class_weight_dict = {i: weight for i, weight in enumerate(class_weights)}

#     return X_train_scaled, X_test_scaled, y_train, y_test, class_weight_dict
