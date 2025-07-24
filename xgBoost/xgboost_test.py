# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# from sklearn.metrics import classification_report, confusion_matrix
# import xgboost as xgb

# def main():
#     """
#     Main function to run the fault detection pipeline using XGBoost.
#     """
#     # --- 1. Load and Prepare Data ---
#     try:
#         df = pd.read_csv('Industrial_fault_detection.csv')
#     except FileNotFoundError:
#         print("Error: 'Industrial_fault_detection.csv' not found.")
#         return

#     # Create a binary label: 0 for 'No Fault', 1 for 'Fault' (any type)
#     df['Is_Fault'] = df['Fault_Type'].apply(lambda x: 1 if x > 0 else 0)

#     X = df.drop(['Fault_Type', 'Is_Fault'], axis=1)
#     y = df['Is_Fault']

#     # Split data into training and testing sets
#     X_train, X_test, y_train, y_test = train_test_split(
#         X, y, test_size=0.3, random_state=42, stratify=y
#     )

#     print(f"Training set size: {len(X_train)} samples")
#     print(f"Test set size: {len(X_test)} samples")
    
#     # --- 2. Calculate scale_pos_weight for Imbalance ---
#     # This is the key parameter to handle class imbalance in XGBoost.
#     # It's the ratio of the number of negative class samples to positive class samples.
#     scale_pos_weight = y_train.value_counts()[0] / y_train.value_counts()[1]
#     print(f"\nCalculated scale_pos_weight for handling imbalance: {scale_pos_weight:.2f}")

#     # --- 3. Scale Data ---
#     # Scaling is still good practice for consistency.
#     scaler = StandardScaler()
#     X_train_scaled = scaler.fit_transform(X_train)
#     X_test_scaled = scaler.transform(X_test)

#     # --- 4. Build and Train XGBoost Model ---
#     print("\n--- Training XGBoost Classifier ---")
#     # Instantiate the XGBoost classifier with the calculated scale_pos_weight
#     model = xgb.XGBClassifier(
#         objective='binary:logistic',
#         eval_metric='logloss',
#         scale_pos_weight=scale_pos_weight,
#         use_label_encoder=False,
#         random_state=42
#     )

#     model.fit(X_train_scaled, y_train)
#     print("Training complete.")

#     # --- 5. Evaluate on Test Data ---
#     print("\n--- Evaluating on Test Set ---")
#     y_pred = model.predict(X_test_scaled)
    
#     # --- 6. Report Final Results ---
#     print("\n--- Final Anomaly Detection Performance (XGBoost) ---")
#     class_report = classification_report(y_test, y_pred, target_names=['Normal (0)', 'Fault (1)'])
#     print(class_report)

#     cm = confusion_matrix(y_test, y_pred)
#     plt.figure(figsize=(8, 6))
#     sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
#                 xticklabels=['Predicted Normal', 'Predicted Fault'],
#                 yticklabels=['Actual Normal', 'Actual Fault'])
#     plt.title('Confusion Matrix (XGBoost)')
#     plt.savefig('xgboost_confusion_matrix.png')
#     print("Final confusion matrix saved as 'xgboost_confusion_matrix.png'")

#     # --- 7. Feature Importance ---
#     # See which sensors are most important for detecting faults.
#     plt.figure(figsize=(10, 8))
#     xgb.plot_importance(model, height=0.6)
#     plt.title('Feature Importance (XGBoost)')
#     plt.tight_layout()
#     plt.savefig('xgboost_feature_importance.png')
#     print("Feature importance plot saved as 'xgboost_feature_importance.png'")


# if __name__ == '__main__':
#     main()

# 2nd attempt: Adding Hyperparameter Tuning with GridSearchCV
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn.model_selection import train_test_split, GridSearchCV
# from sklearn.preprocessing import StandardScaler
# from sklearn.metrics import classification_report, confusion_matrix
# import xgboost as xgb

# def main():
#     """
#     Main function to run the fault detection pipeline using XGBoost with Hyperparameter Tuning.
#     """
#     # --- 1. Load and Prepare Data ---
#     try:
#         df = pd.read_csv('Industrial_fault_detection.csv')
#     except FileNotFoundError:
#         print("Error: 'Industrial_fault_detection.csv' not found.")
#         return

#     df['Is_Fault'] = df['Fault_Type'].apply(lambda x: 1 if x > 0 else 0)

#     X = df.drop(['Fault_Type', 'Is_Fault'], axis=1)
#     y = df['Is_Fault']

#     X_train, X_test, y_train, y_test = train_test_split(
#         X, y, test_size=0.3, random_state=42, stratify=y
#     )

#     print(f"Training set size: {len(X_train)} samples")
#     print(f"Test set size: {len(X_test)} samples")
    
#     # --- 2. Calculate scale_pos_weight for Imbalance ---
#     scale_pos_weight = y_train.value_counts()[0] / y_train.value_counts()[1]
#     print(f"\nCalculated scale_pos_weight for handling imbalance: {scale_pos_weight:.2f}")

#     # --- 3. Scale Data ---
#     scaler = StandardScaler()
#     X_train_scaled = scaler.fit_transform(X_train)
#     X_test_scaled = scaler.transform(X_test)

#     # --- 4. Hyperparameter Tuning with GridSearchCV ---
#     print("\n--- Finding Best Hyperparameters with GridSearchCV ---")
    
#     # Define the parameter grid to search
#     param_grid = {
#         'max_depth': [3, 5, 7],
#         'learning_rate': [0.01, 0.1, 0.2],
#         'n_estimators': [100, 200, 300],
#         'gamma': [0, 0.1, 0.2]
#     }

#     # Instantiate the XGBoost classifier
#     model = xgb.XGBClassifier(
#         objective='binary:logistic',
#         eval_metric='logloss',
#         scale_pos_weight=scale_pos_weight,
#         # use_label_encoder=False,
#         random_state=42
#     )
    
#     # Set up GridSearchCV to find the best parameters by maximizing recall
#     # 'cv=3' means 3-fold cross-validation.
#     # 'scoring='recall'' is crucial - it tells the search to prioritize finding faults.
#     grid_search = GridSearchCV(
#         estimator=model, 
#         param_grid=param_grid, 
#         scoring='recall', 
#         cv=3, 
#         verbose=1,
#         n_jobs=-1 # Use all available CPU cores
#     )

#     grid_search.fit(X_train_scaled, y_train)

#     print("\nBest parameters found: ", grid_search.best_params_)
    
#     # Use the best model found by the grid search
#     best_model = grid_search.best_estimator_

#     # --- 5. Evaluate on Test Data with Best Model ---
#     print("\n--- Evaluating on Test Set with Best Model ---")
#     y_pred = best_model.predict(X_test_scaled)
    
#     # --- 6. Report Final Results ---
#     print("\n--- Final Anomaly Detection Performance (Tuned XGBoost) ---")
#     class_report = classification_report(y_test, y_pred, target_names=['Normal (0)', 'Fault (1)'])
#     print(class_report)

#     cm = confusion_matrix(y_test, y_pred)
#     plt.figure(figsize=(8, 6))
#     sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
#                 xticklabels=['Predicted Normal', 'Predicted Fault'],
#                 yticklabels=['Actual Normal', 'Actual Fault'])
#     plt.title('Confusion Matrix (Tuned XGBoost)')
#     plt.savefig('xgboost_tuned_confusion_matrix.png')
#     print("Final confusion matrix saved as 'xgboost_tuned_confusion_matrix.png'")

#     # --- 7. Feature Importance ---
#     plt.figure(figsize=(10, 8))
#     xgb.plot_importance(best_model, height=0.6)
#     plt.title('Feature Importance (Tuned XGBoost)')
#     plt.tight_layout()
#     plt.savefig('xgboost_tuned_feature_importance.png')
#     print("Feature importance plot saved as 'xgboost_tuned_feature_importance.png'")


# if __name__ == '__main__':
#     main()

# With Smote 
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn.model_selection import train_test_split, GridSearchCV
# from sklearn.preprocessing import StandardScaler
# from sklearn.metrics import classification_report, confusion_matrix
# # Import SMOTE from the imblearn library
# from imblearn.over_sampling import SMOTE
# import xgboost as xgb

# def main():
#     """
#     Main function to run the fault detection pipeline using XGBoost with SMOTE for handling class imbalance.
#     """
#     # --- 1. Load and Prepare Data ---
#     try:
#         df = pd.read_csv('Industrial_fault_detection.csv')
#     except FileNotFoundError:
#         print("Error: 'Industrial_fault_detection.csv' not found.")
#         return

#     df['Is_Fault'] = df['Fault_Type'].apply(lambda x: 1 if x > 0 else 0)

#     X = df.drop(['Fault_Type', 'Is_Fault'], axis=1)
#     y = df['Is_Fault']

#     X_train, X_test, y_train, y_test = train_test_split(
#         X, y, test_size=0.3, random_state=42, stratify=y
#     )

#     print(f"Original training set shape: {X_train.shape}")
#     print(f"Original training set distribution:\n{y_train.value_counts()}")
    
#     # --- 2. Scale Data ---
#     # We fit the scaler on the original training data
#     scaler = StandardScaler()
#     X_train_scaled = scaler.fit_transform(X_train)
#     X_test_scaled = scaler.transform(X_test)

#     # --- 3. Apply SMOTE to Generate Synthetic Data ---
#     print("\n--- Applying SMOTE to balance the training data ---")
#     smote = SMOTE(random_state=42)
#     # We apply SMOTE only to the training data
#     X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)
    
#     print(f"Resampled training set shape: {X_train_resampled.shape}")
#     print(f"Resampled training set distribution:\n{y_train_resampled.value_counts()}")


#     # --- 4. Train XGBoost Model on Resampled Data ---
#     print("\n--- Training XGBoost Classifier on SMOTE data ---")
    
#     # Instantiate the XGBoost classifier.
#     # We no longer need scale_pos_weight because SMOTE has balanced the data.
#     model = xgb.XGBClassifier(
#         objective='binary:logistic',
#         eval_metric='logloss',
#         use_label_encoder=False,
#         random_state=42
#     )

#     # We train the model on the new, balanced, resampled data
#     model.fit(X_train_resampled, y_train_resampled)
#     print("Training complete.")
    
#     # --- 5. Evaluate on the Original (imbalanced) Test Data ---
#     print("\n--- Evaluating on the original, unseen test set ---")
#     y_pred = model.predict(X_test_scaled)
    
#     # --- 6. Report Final Results ---
#     print("\n--- Final Anomaly Detection Performance (XGBoost with SMOTE) ---")
#     class_report = classification_report(y_test, y_pred, target_names=['Normal (0)', 'Fault (1)'])
#     print(class_report)

#     cm = confusion_matrix(y_test, y_pred)
#     plt.figure(figsize=(8, 6))
#     sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
#                 xticklabels=['Predicted Normal', 'Predicted Fault'],
#                 yticklabels=['Actual Normal', 'Actual Fault'])
#     plt.title('Confusion Matrix (XGBoost with SMOTE)')
#     plt.savefig('xgboost_smote_confusion_matrix.png')
#     print("Final confusion matrix saved as 'xgboost_smote_confusion_matrix.png'")

#     # --- 7. Feature Importance ---
#     plt.figure(figsize=(10, 8))
#     xgb.plot_importance(model, height=0.6)
#     plt.title('Feature Importance (XGBoost with SMOTE)')
#     plt.tight_layout()
#     plt.savefig('xgboost_smote_feature_importance.png')
#     print("Feature importance plot saved as 'xgboost_smote_feature_importance.png'")


# if __name__ == '__main__':
#     main()

# Random Forest
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn.model_selection import train_test_split, GridSearchCV
# from sklearn.preprocessing import StandardScaler
# from sklearn.metrics import classification_report, confusion_matrix
# # Import the necessary tools from imblearn and sklearn
# from imblearn.over_sampling import SMOTE
# from imblearn.pipeline import Pipeline
# from sklearn.ensemble import RandomForestClassifier


# def main():
#     """
#     Main function to run the fault detection pipeline using a combination of
#     SMOTE and a tuned RandomForestClassifier for optimal performance.
#     """
#     # --- 1. Load and Prepare Data ---
#     try:
#         df = pd.read_csv('Industrial_fault_detection.csv')
#     except FileNotFoundError:
#         print("Error: 'Industrial_fault_detection.csv' not found.")
#         return

#     df['Is_Fault'] = df['Fault_Type'].apply(lambda x: 1 if x > 0 else 0)

#     X = df.drop(['Fault_Type', 'Is_Fault'], axis=1)
#     y = df['Is_Fault']

#     X_train, X_test, y_train, y_test = train_test_split(
#         X, y, test_size=0.3, random_state=42, stratify=y
#     )

#     print(f"Original training set shape: {X_train.shape}")
#     print(f"Original training set distribution:\n{y_train.value_counts()}")
    
#     # --- 2. Scale Data ---
#     # We scale the data before feeding it into the pipeline.
#     scaler = StandardScaler()
#     X_train_scaled = scaler.fit_transform(X_train)
#     X_test_scaled = scaler.transform(X_test)

#     # --- 3. Set up Pipeline and Hyperparameter Grid ---
#     # Create a pipeline that first applies SMOTE and then trains a RandomForestClassifier.
#     # This is the correct way to use SMOTE with cross-validation (GridSearchCV).
#     pipeline = Pipeline([
#         ('smote', SMOTE(random_state=42)),
#         ('classifier', RandomForestClassifier(random_state=42))
#     ])

#     # Define the parameter grid for the RandomForestClassifier.
#     # Note the 'classifier__' prefix, which tells the pipeline which step to apply the parameters to.
#     param_grid = {
#         'classifier__n_estimators': [100, 200],
#         'classifier__max_depth': [10, 20, None],
#         'classifier__min_samples_split': [2, 5],
#         'classifier__min_samples_leaf': [1, 2]
#     }

#     # --- 4. Find Best Model with GridSearchCV ---
#     print("\n--- Finding Best RandomForest Model with GridSearchCV (optimizing for Recall) ---")
    
#     # Set up GridSearchCV to find the best parameters by maximizing recall.
#     grid_search = GridSearchCV(
#         estimator=pipeline, 
#         param_grid=param_grid, 
#         scoring='recall', 
#         cv=3, 
#         verbose=1,
#         n_jobs=-1 # Use all available CPU cores
#     )

#     # Train the grid search on the (unbalanced) training data.
#     # The pipeline will handle applying SMOTE correctly for each fold.
#     grid_search.fit(X_train_scaled, y_train)

#     print("\nBest parameters found: ", grid_search.best_params_)
    
#     # The best model is the one found by the grid search
#     best_model = grid_search.best_estimator_

#     # --- 5. Evaluate the Best Model ---
#     print("\n--- Evaluating on the original, unseen test set ---")
#     y_pred = best_model.predict(X_test_scaled)
    
#     # --- 6. Report Final Results ---
#     print("\n--- Final Anomaly Detection Performance (Tuned RandomForest with SMOTE) ---")
#     class_report = classification_report(y_test, y_pred, target_names=['Normal (0)', 'Fault (1)'])
#     print(class_report)

#     cm = confusion_matrix(y_test, y_pred)
#     plt.figure(figsize=(8, 6))
#     sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
#                 xticklabels=['Predicted Normal', 'Predicted Fault'],
#                 yticklabels=['Actual Normal', 'Actual Fault'])
#     plt.title('Confusion Matrix (Tuned RandomForest with SMOTE)')
#     plt.savefig('randomforest_smote_confusion_matrix.png')
#     print("Final confusion matrix saved as 'randomforest_smote_confusion_matrix.png'")


# if __name__ == '__main__':
#     main()


#  xgboost with new dataset
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# import xgboost as xgb
# # Import necessary tools from sklearn and imblearn
# from sklearn.model_selection import GridSearchCV
# from sklearn.metrics import classification_report, confusion_matrix
# from imblearn.over_sampling import SMOTE
# from imblearn.pipeline import Pipeline
# from scipy.fft import fft

# def create_advanced_features(df, sensor_columns):
#     """
#     Creates basic and advanced time-series features from the raw sensor data.
    
#     Args:
#         df (pd.DataFrame): The input dataframe with a datetime index and sensor columns.
#         sensor_columns (list): The list of sensor column names to create features for.
        
#     Returns:
#         pd.DataFrame: The dataframe with new time-series features.
#     """
#     df_featured = df.copy()
#     window_sizes = [5, 15]
    
#     print("\n--- Starting Advanced Feature Engineering ---")
    
#     # --- Time-Based Features ---
#     df_featured['hour'] = df_featured.index.hour
#     df_featured['dayofweek'] = df_featured.index.dayofweek

#     for col in sensor_columns:
#         # --- Basic Rolling and Lag Features ---
#         for window in window_sizes:
#             df_featured[f'{col}_roll_mean_{window}'] = df_featured[col].rolling(window=window, min_periods=1).mean()
#             df_featured[f'{col}_roll_std_{window}'] = df_featured[col].rolling(window=window, min_periods=1).std()
        
#         for lag in [1, 3]:
#             df_featured[f'{col}_lag_{lag}'] = df_featured[col].shift(lag)
            
#         # --- Fourier Transform Features for Vibration ---
#         if 'vibration' in col:
#             # Calculate FFT
#             fft_vals = fft(df_featured[col].values)
#             # Get the dominant frequency (excluding the DC component at index 0)
#             dominant_freq = np.argmax(np.abs(fft_vals[1:])) + 1
#             df_featured[f'{col}_dominant_freq'] = dominant_freq
    
#     # --- Interaction Features ---
#     if 'temperature' in sensor_columns and 'pressure' in sensor_columns:
#         df_featured['temp_x_pressure'] = df_featured['temperature'] * df_featured['pressure']
#     if 'vibration' in sensor_columns and 'current' in sensor_columns:
#         df_featured['vibration_x_current'] = df_featured['vibration'] * df_featured['current']
            
#     print("Feature engineering complete.")
#     df_featured.fillna(0, inplace=True) # Fill NaNs with 0 after shifting/rolling
    
#     return df_featured

# def main():
#     """
#     Main function to run the entire predictive maintenance pipeline.
#     """
#     # --- 1. Load and Prepare Data ---
#     try:
#         df = pd.read_csv('smart_manufacturing_data.csv')
#     except FileNotFoundError:
#         print("Error: 'smart_manufacturing_data.csv' not found.")
#         return

#     df['timestamp'] = pd.to_datetime(df['timestamp'])
#     df.set_index('timestamp', inplace=True)
    
#     df['Is_Failure'] = (df['failure_type'] != 'Normal').astype(int)
    
#     sensor_columns = ['temperature', 'vibration', 'humidity', 'pressure', 'energy_consumption', 'current', 'voltage']
#     # Ensure all sensor columns exist in the dataframe before proceeding
#     sensor_columns = [col for col in sensor_columns if col in df.columns]
    
#     df_model_data = df[sensor_columns + ['Is_Failure']]

#     # --- 2. Feature Engineering ---
#     df_featured = create_advanced_features(df_model_data, sensor_columns)

#     # --- 3. Define Features (X) and Target (y) ---
#     X = df_featured.drop('Is_Failure', axis=1)
#     y = df_featured['Is_Failure']

#     # --- 4. Chronological Train-Test Split (80/20) ---
#     split_index = int(len(X) * 0.8)
#     X_train, y_train = X.iloc[:split_index], y.iloc[:split_index]
#     X_test, y_test = X.iloc[split_index:], y.iloc[split_index:]

#     print(f"\nTraining set size: {len(X_train)} samples")
#     print(f"Testing set size: {len(X_test)} samples")

#     # --- 5. Set up Pipeline with SMOTE and XGBoost ---
#     if 1 not in y_train.value_counts() or y_train.value_counts()[1] == 0:
#         print("\nWarning: No failure events found in the training data. Skipping model training.")
#         return
        
#     pipeline = Pipeline([
#         ('smote', SMOTE(random_state=42)),
#         ('classifier', xgb.XGBClassifier(
#             objective='binary:logistic',
#             eval_metric='logloss',
#             # use_label_encoder=False,
#             random_state=42
#         ))
#     ])
    
#     param_grid = {
#         'classifier__max_depth': [5, 7],
#         'classifier__learning_rate': [0.1, 0.2],
#         'classifier__n_estimators': [200, 300],
#         'classifier__gamma': [0.1, 0.2]
#     }

#     # --- 6. Hyperparameter Tuning with GridSearchCV on the Pipeline ---
#     print("\n--- Finding Best Hyperparameters with SMOTE + GridSearchCV ---")
    
#     grid_search = GridSearchCV(
#         estimator=pipeline, 
#         param_grid=param_grid, 
#         scoring='recall', 
#         cv=3,
#         verbose=2,
#         n_jobs=-1
#     )

#     grid_search.fit(X_train, y_train)

#     print("\nBest parameters found: ", grid_search.best_params_)
    
#     best_pipeline = grid_search.best_estimator_

#     # --- 7. Evaluate the Best Pipeline ---
#     print("\n--- Evaluating on Test Set with Best Tuned Pipeline ---")
#     y_pred = best_pipeline.predict(X_test)
    
#     print("\n--- Final Performance (Tuned XGBoost with SMOTE and Advanced Features) ---")
#     class_report = classification_report(y_test, y_pred, target_names=['No Failure (0)', 'Failure (1)'], zero_division=0)
#     print(class_report)

#     cm = confusion_matrix(y_test, y_pred)
#     plt.figure(figsize=(8, 6))
#     sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
#                 xticklabels=['Predicted No Failure', 'Predicted Failure'],
#                 yticklabels=['Actual No Failure', 'Actual Failure'])
#     plt.title('Confusion Matrix (Tuned XGBoost with SMOTE & Advanced Features)')
#     plt.savefig('timeseries_xgboost_advanced_features_matrix.png')
#     print("Final confusion matrix saved as 'timeseries_xgboost_advanced_features_matrix.png'")


# if __name__ == '__main__':
#     main()

