# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# from sklearn.metrics import classification_report, confusion_matrix
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import LSTM, Dense, Dropout
# from tensorflow.keras.utils import to_categorical

# def create_sequences(X, y, time_steps=30):
#     """
#     Creates sequences of data for LSTM model.
    
#     Args:
#         X (np.array): The input features.
#         y (np.array): The target labels.
#         time_steps (int): The number of time steps in each sequence.
        
#     Returns:
#         np.array, np.array: The sequences of features and corresponding labels.
#     """
#     Xs, ys = [], []
#     for i in range(len(X) - time_steps):
#         v = X[i:(i + time_steps)]
#         Xs.append(v)
#         ys.append(y[i + time_steps])
#     return np.array(Xs), np.array(ys)

# def build_lstm_model(input_shape):
#     """
#     Builds the LSTM model architecture.
#     """
#     model = Sequential()
#     model.add(LSTM(100, input_shape=input_shape, return_sequences=True))
#     model.add(Dropout(0.2))
#     model.add(LSTM(50))
#     model.add(Dropout(0.2))
#     model.add(Dense(2, activation='softmax')) # 2 classes: No Failure, Failure
    
#     model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
#     return model

# def main():
#     """
#     Main function to run the predictive maintenance pipeline using an LSTM model.
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
#     sensor_columns = [col for col in sensor_columns if col in df.columns]
    
#     df_model_data = df[sensor_columns + ['Is_Failure']]
#     df_model_data.fillna(0, inplace=True)

#     # --- 2. Scale Features ---
#     scaler = StandardScaler()
#     df_model_data[sensor_columns] = scaler.fit_transform(df_model_data[sensor_columns])

#     # --- 3. Create Sequences ---
#     TIME_STEPS = 30 # Use a window of 30 time steps to predict the next step
#     X_sequences, y_sequences = create_sequences(
#         df_model_data[sensor_columns].values,
#         df_model_data['Is_Failure'].values,
#         TIME_STEPS
#     )
    
#     # Convert labels to categorical for softmax
#     y_sequences_cat = to_categorical(y_sequences)

#     # --- 4. Chronological Train-Test Split ---
#     split_index = int(len(X_sequences) * 0.8)
#     X_train, y_train = X_sequences[:split_index], y_sequences_cat[:split_index]
#     X_test, y_test = X_sequences[split_index:], y_sequences_cat[split_index:]
    
#     # Keep the original 0/1 labels for the final report
#     y_test_original = y_sequences[split_index:]

#     print(f"\nTraining sequences: {X_train.shape[0]}")
#     print(f"Testing sequences: {X_test.shape[0]}")

#     # --- 5. Calculate Class Weights for Imbalance ---
#     # This is the standard way to handle imbalance in Keras
#     neg, pos = np.bincount(df_model_data['Is_Failure'])
#     total = neg + pos
#     weight_for_0 = (1 / neg) * (total / 2.0)
#     weight_for_1 = (1 / pos) * (total / 2.0)
#     class_weight = {0: weight_for_0, 1: weight_for_1}
    
#     print(f"\nClass weights: {class_weight}")

#     # --- 6. Build and Train LSTM Model ---
#     print("\n--- Building and Training LSTM Model ---")
#     model = build_lstm_model(input_shape=(X_train.shape[1], X_train.shape[2]))
#     model.summary()

#     history = model.fit(
#         X_train, y_train,
#         epochs=10,
#         batch_size=128,
#         validation_split=0.1,
#         class_weight=class_weight,
#         verbose=1
#     )

#     # --- 7. Evaluate the Model ---
#     print("\n--- Evaluating on the Test Set ---")
#     y_pred_prob = model.predict(X_test)
#     y_pred = np.argmax(y_pred_prob, axis=1) # Get the class with the highest probability
    
#     print("\n--- Final Performance (LSTM) ---")
#     class_report = classification_report(y_test_original, y_pred, target_names=['No Failure (0)', 'Failure (1)'], zero_division=0)
#     print(class_report)

#     cm = confusion_matrix(y_test_original, y_pred)
#     plt.figure(figsize=(8, 6))
#     sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
#                 xticklabels=['Predicted No Failure', 'Predicted Failure'],
#                 yticklabels=['Actual No Failure', 'Actual Failure'])
#     plt.title('Confusion Matrix (LSTM)')
#     plt.savefig('lstm_confusion_matrix.png')
#     print("Final confusion matrix saved as 'lstm_confusion_matrix.png'")


# if __name__ == '__main__':
#     main()

# lstm 
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# from sklearn.metrics import classification_report, confusion_matrix
# from tensorflow.keras.models import Sequential
# # Import Bidirectional wrapper and other necessary layers/callbacks
# from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
# from tensorflow.keras.utils import to_categorical
# from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping

# def create_sequences(X, y, time_steps=30):
#     """
#     Creates sequences of data for LSTM model.
    
#     Args:
#         X (np.array): The input features.
#         y (np.array): The target labels.
#         time_steps (int): The number of time steps in each sequence.
        
#     Returns:
#         np.array, np.array: The sequences of features and corresponding labels.
#     """
#     Xs, ys = [], []
#     for i in range(len(X) - time_steps):
#         v = X[i:(i + time_steps)]
#         Xs.append(v)
#         ys.append(y[i + time_steps])
#     return np.array(Xs), np.array(ys)

# def build_bidirectional_lstm_model(input_shape):
#     """
#     Builds a more advanced and stable Bidirectional LSTM model.
#     """
#     model = Sequential()
#     # The Bidirectional wrapper allows the LSTM to learn from the sequence in both directions
#     model.add(Bidirectional(LSTM(100, return_sequences=True), input_shape=input_shape))
#     model.add(Dropout(0.3)) # Increased dropout for regularization
#     model.add(Bidirectional(LSTM(50)))
#     model.add(Dropout(0.3))
#     model.add(Dense(2, activation='softmax')) # 2 classes: No Failure, Failure
    
#     model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
#     return model

# def main():
#     """
#     Main function to run the predictive maintenance pipeline using an LSTM model.
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
#     sensor_columns = [col for col in sensor_columns if col in df.columns]
    
#     df_model_data = df[sensor_columns + ['Is_Failure']]
#     df_model_data.fillna(0, inplace=True)

#     # --- 2. Scale Features ---
#     scaler = StandardScaler()
#     # Use a copy to avoid SettingWithCopyWarning
#     df_scaled = df_model_data.copy()
#     df_scaled[sensor_columns] = scaler.fit_transform(df_scaled[sensor_columns])

#     # --- 3. Create Sequences ---
#     TIME_STEPS = 30 # Use a window of 30 time steps to predict the next step
#     X_sequences, y_sequences = create_sequences(
#         df_scaled[sensor_columns].values,
#         df_scaled['Is_Failure'].values,
#         TIME_STEPS
#     )
    
#     y_sequences_cat = to_categorical(y_sequences)

#     # --- 4. Chronological Train-Test Split ---
#     split_index = int(len(X_sequences) * 0.8)
#     X_train, y_train = X_sequences[:split_index], y_sequences_cat[:split_index]
#     X_test, y_test = X_sequences[split_index:], y_sequences_cat[split_index:]
    
#     y_test_original = y_sequences[split_index:]

#     print(f"\nTraining sequences: {X_train.shape[0]}")
#     print(f"Testing sequences: {X_test.shape[0]}")

#     # --- 5. Calculate Class Weights for Imbalance ---
#     neg, pos = np.bincount(df_model_data['Is_Failure'])
#     total = neg + pos
#     weight_for_0 = (1 / neg) * (total / 2.0) if neg > 0 else 0
#     weight_for_1 = (1 / pos) * (total / 2.0) if pos > 0 else 0
#     class_weight = {0: weight_for_0, 1: weight_for_1}
    
#     print(f"\nClass weights: {class_weight}")

#     # --- 6. Build and Train Advanced LSTM Model ---
#     print("\n--- Building and Training Bidirectional LSTM Model ---")
#     model = build_bidirectional_lstm_model(input_shape=(X_train.shape[1], X_train.shape[2]))
#     model.summary()

#     # Define callbacks for more stable training
#     # ReduceLROnPlateau will reduce the learning rate if the validation loss stops improving.
#     lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)
#     # EarlyStopping will stop training if there's no improvement after a certain number of epochs.
#     early_stopping = EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True, verbose=1)

#     history = model.fit(
#         X_train, y_train,
#         epochs=30, # Increased epochs to give the model more time to learn
#         batch_size=128,
#         validation_split=0.1,
#         class_weight=class_weight,
#         callbacks=[lr_scheduler, early_stopping],
#         verbose=1
#     )

#     # --- 7. Evaluate the Model ---
#     print("\n--- Evaluating on the Test Set ---")
#     y_pred_prob = model.predict(X_test)
#     y_pred = np.argmax(y_pred_prob, axis=1)
    
#     print("\n--- Final Performance (Bidirectional LSTM) ---")
#     class_report = classification_report(y_test_original, y_pred, target_names=['No Failure (0)', 'Failure (1)'], zero_division=0)
#     print(class_report)

#     cm = confusion_matrix(y_test_original, y_pred)
#     plt.figure(figsize=(8, 6))
#     sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
#                 xticklabels=['Predicted No Failure', 'Predicted Failure'],
#                 yticklabels=['Actual No Failure', 'Actual Failure'])
#     plt.title('Confusion Matrix (Bidirectional LSTM)')
#     plt.savefig('bidirectional_lstm_confusion_matrix.png')
#     print("Final confusion matrix saved as 'bidirectional_lstm_confusion_matrix.png'")


# if __name__ == '__main__':
#     main()

# lstm and auto encoder
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve
# from tensorflow.keras.models import Model
# from tensorflow.keras.layers import Input, LSTM, Dense, RepeatVector, TimeDistributed
# from tensorflow.keras.callbacks import EarlyStopping

# def create_sequences(X, time_steps=30):
#     """
#     Creates sequences of data for the LSTM Autoencoder.
    
#     Args:
#         X (np.array): The input features.
#         time_steps (int): The number of time steps in each sequence.
        
#     Returns:
#         np.array: The sequences of features.
#     """
#     Xs = []
#     for i in range(len(X) - time_steps):
#         v = X[i:(i + time_steps)]
#         Xs.append(v)
#     return np.array(Xs)

# def build_lstm_autoencoder(input_shape):
#     """
#     Builds the LSTM Autoencoder model.
#     """
#     # --- Encoder ---
#     # Takes a sequence of data and compresses it into a single vector.
#     inputs = Input(shape=input_shape)
#     encoder = LSTM(128, activation='relu', return_sequences=False)(inputs)
    
#     # --- Repeat Vector ---
#     # Repeats the compressed vector so it can be fed into the decoder.
#     repeater = RepeatVector(input_shape[0])(encoder)
    
#     # --- Decoder ---
#     # Takes the compressed vector and tries to reconstruct the original sequence.
#     decoder = LSTM(128, activation='relu', return_sequences=True)(repeater)
#     output = TimeDistributed(Dense(input_shape[1]))(decoder)
    
#     model = Model(inputs=inputs, outputs=output)
#     model.compile(optimizer='adam', loss='mse')
#     return model

# def main():
#     """
#     Main function to run the predictive maintenance pipeline using an LSTM Autoencoder.
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
#     sensor_columns = [col for col in sensor_columns if col in df.columns]
    
#     df_model_data = df[sensor_columns + ['Is_Failure']]
#     df_model_data.fillna(0, inplace=True)

#     # --- 2. Split Data Chronologically Before Scaling ---
#     split_index = int(len(df_model_data) * 0.8)
#     train_df = df_model_data.iloc[:split_index]
#     test_df = df_model_data.iloc[split_index:]

#     # --- 3. Scale Features ---
#     # Fit the scaler ONLY on the training data's sensor values
#     scaler = StandardScaler()
#     train_df_scaled = train_df.copy()
#     test_df_scaled = test_df.copy()
    
#     train_df_scaled[sensor_columns] = scaler.fit_transform(train_df[sensor_columns])
#     test_df_scaled[sensor_columns] = scaler.transform(test_df[sensor_columns])

#     # --- 4. Create Sequences for Normal Data Only (for training) ---
#     # We only train the autoencoder on healthy machine data.
#     normal_train_data = train_df_scaled[train_df_scaled['Is_Failure'] == 0]
#     X_train = create_sequences(normal_train_data[sensor_columns].values)
    
#     # Create sequences for the entire test set (normal and failure)
#     X_test = create_sequences(test_df_scaled[sensor_columns].values)
#     y_test = test_df_scaled['Is_Failure'].values[30:] # Align labels with sequences

#     print(f"\nTraining sequences (normal data only): {X_train.shape[0]}")
#     print(f"Testing sequences (mixed data): {X_test.shape[0]}")

#     # --- 5. Build and Train LSTM Autoencoder ---
#     print("\n--- Building and Training LSTM Autoencoder ---")
#     model = build_lstm_autoencoder(input_shape=(X_train.shape[1], X_train.shape[2]))
#     model.summary()

#     early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    
#     history = model.fit(
#         X_train, X_train, # Note: input and target are the same
#         epochs=50,
#         batch_size=128,
#         validation_split=0.1,
#         callbacks=[early_stopping],
#         verbose=1
#     )

#     # --- 6. Determine Optimal Anomaly Threshold ---
#     print("\n--- Finding Optimal Anomaly Threshold ---")
#     reconstructions = model.predict(X_test)
#     test_mse = np.mean(np.power(X_test - reconstructions, 2), axis=(1, 2))
    
#     precision, recall, thresholds = precision_recall_curve(y_test, test_mse)
    
#     f1_scores = np.nan_to_num(2 * (precision * recall) / (precision + recall))
    
#     optimal_idx = np.argmax(f1_scores)
#     optimal_threshold = thresholds[optimal_idx]
    
#     print(f"Optimal Threshold based on max F1-score: {optimal_threshold:.4f}")

#     # --- 7. Evaluate on Test Data with Optimal Threshold ---
#     y_pred = (test_mse > optimal_threshold).astype(int)
    
#     print("\n--- Final Performance (LSTM Autoencoder) ---")
#     class_report = classification_report(y_test, y_pred, target_names=['No Failure (0)', 'Failure (1)'], zero_division=0)
#     print(class_report)

#     cm = confusion_matrix(y_test, y_pred)
#     plt.figure(figsize=(8, 6))
#     sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
#                 xticklabels=['Predicted No Failure', 'Predicted Failure'],
#                 yticklabels=['Actual No Failure', 'Actual Failure'])
#     plt.title('Confusion Matrix (LSTM Autoencoder)')
#     plt.savefig('lstm_autoencoder_confusion_matrix.png')
#     print("Final confusion matrix saved as 'lstm_autoencoder_confusion_matrix.png'")


# if __name__ == '__main__':
#     main()

# new approach code best so far
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve
# from tensorflow.keras.models import Model
# from tensorflow.keras.layers import Input, LSTM, Dense, RepeatVector, TimeDistributed
# from tensorflow.keras.callbacks import EarlyStopping

# def create_sequences(X, time_steps=30):
#     """
#     Creates sequences of data for the LSTM Autoencoder.
    
#     Args:
#         X (np.array): The input features.
#         time_steps (int): The number of time steps in each sequence.
        
#     Returns:
#         np.array: The sequences of features.
#     """
#     Xs = []
#     for i in range(len(X) - time_steps):
#         v = X[i:(i + time_steps)]
#         Xs.append(v)
#     return np.array(Xs)

# def build_lstm_autoencoder(input_shape):
#     """
#     Builds the LSTM Autoencoder model.
#     """
#     # --- Encoder ---
#     inputs = Input(shape=input_shape)
#     encoder = LSTM(128, activation='relu', return_sequences=False)(inputs)
    
#     # --- Repeat Vector ---
#     repeater = RepeatVector(input_shape[0])(encoder)
    
#     # --- Decoder ---
#     decoder = LSTM(128, activation='relu', return_sequences=True)(repeater)
#     output = TimeDistributed(Dense(input_shape[1]))(decoder)
    
#     model = Model(inputs=inputs, outputs=output)
#     model.compile(optimizer='adam', loss='mse')
#     return model

# def main():
#     """
#     Main function to run the predictive maintenance pipeline using an LSTM Autoencoder.
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
#     sensor_columns = [col for col in sensor_columns if col in df.columns]
    
#     df_model_data = df[sensor_columns + ['Is_Failure']]
#     df_model_data.fillna(0, inplace=True)

#     # --- 2. Split Data Chronologically Before Scaling ---
#     split_index = int(len(df_model_data) * 0.8)
#     train_df = df_model_data.iloc[:split_index]
#     test_df = df_model_data.iloc[split_index:]

#     # --- 3. Scale Features ---
#     scaler = StandardScaler()
#     train_df_scaled = train_df.copy()
#     test_df_scaled = test_df.copy()
    
#     train_df_scaled[sensor_columns] = scaler.fit_transform(train_df[sensor_columns])
#     test_df_scaled[sensor_columns] = scaler.transform(test_df[sensor_columns])

#     # --- 4. Create Sequences ---
#     normal_train_data = train_df_scaled[train_df_scaled['Is_Failure'] == 0]
#     X_train = create_sequences(normal_train_data[sensor_columns].values)
    
#     X_test = create_sequences(test_df_scaled[sensor_columns].values)
#     y_test = test_df_scaled['Is_Failure'].values[30:]

#     print(f"\nTraining sequences (normal data only): {X_train.shape[0]}")
#     print(f"Testing sequences (mixed data): {X_test.shape[0]}")

#     # --- 5. Build and Train LSTM Autoencoder ---
#     print("\n--- Building and Training LSTM Autoencoder ---")
#     model = build_lstm_autoencoder(input_shape=(X_train.shape[1], X_train.shape[2]))
#     model.summary()

#     early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    
#     history = model.fit(
#         X_train, X_train,
#         epochs=50,
#         batch_size=128,
#         validation_split=0.1,
#         callbacks=[early_stopping],
#         verbose=1
#     )

#     # --- 6. Analyze Precision-Recall Trade-off ---
#     print("\n--- Analyzing Precision-Recall Trade-off ---")
#     reconstructions = model.predict(X_test)
#     test_mse = np.mean(np.power(X_test - reconstructions, 2), axis=(1, 2))
    
#     precision, recall, thresholds = precision_recall_curve(y_test, test_mse)
    
#     # Plot the Precision-Recall Curve
#     plt.figure(figsize=(10, 7))
#     plt.plot(recall, precision, marker='.', label='Precision-Recall Curve')
#     plt.xlabel('Recall (Sensitivity)')
#     plt.ylabel('Precision (Positive Predictive Value)')
#     plt.title('Precision-Recall Curve')
#     plt.grid(True)
#     plt.legend()
#     plt.savefig('precision_recall_curve.png')
#     print("Precision-Recall curve saved as 'precision_recall_curve.png'")
    
#     # --- 7. Evaluate Different Thresholds ---
    
#     # --- Threshold 1: Maximize F1-Score (Balanced Approach) ---
#     f1_scores = np.nan_to_num(2 * (precision * recall) / (precision + recall))
#     optimal_idx = np.argmax(f1_scores)
#     threshold_f1 = thresholds[optimal_idx]
#     print(f"\n--- Evaluation for Threshold that Maximizes F1-Score: {threshold_f1:.4f} ---")
#     y_pred_f1 = (test_mse > threshold_f1).astype(int)
#     print(classification_report(y_test, y_pred_f1, target_names=['No Failure (0)', 'Failure (1)'], zero_division=0))

#     # --- Threshold 2: Target High Precision (e.g., 80%) ---
#     # Find the threshold where precision is >= 0.80 for the first time
#     try:
#         high_precision_idx = np.where(precision >= 0.80)[0][0]
#         threshold_p80 = thresholds[high_precision_idx]
#         print(f"\n--- Evaluation for Threshold with >= 80% Precision: {threshold_p80:.4f} ---")
#         y_pred_p80 = (test_mse > threshold_p80).astype(int)
#         print(classification_report(y_test, y_pred_p80, target_names=['No Failure (0)', 'Failure (1)'], zero_division=0))
#     except IndexError:
#         print("\nCould not find a threshold that achieves 80% precision.")

#     # --- Threshold 3: Target High Recall (e.g., 90%) ---
#     # Find the threshold for the highest recall that is >= 0.90
#     try:
#         high_recall_idx = np.where(recall >= 0.90)[0][-1]
#         threshold_r90 = thresholds[high_recall_idx]
#         print(f"\n--- Evaluation for Threshold with >= 90% Recall: {threshold_r90:.4f} ---")
#         y_pred_r90 = (test_mse > threshold_r90).astype(int)
#         print(classification_report(y_test, y_pred_r90, target_names=['No Failure (0)', 'Failure (1)'], zero_division=0))
#     except IndexError:
#         print("\nCould not find a threshold that achieves 90% recall.")


# if __name__ == '__main__':
#     main()


# implementation starts now with GPT
# lstm classifier implementation
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns

# from sklearn.preprocessing import StandardScaler
# from sklearn.metrics import classification_report, confusion_matrix

# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import LSTM, Dense, Dropout
# from tensorflow.keras.callbacks import EarlyStopping
# from tensorflow.keras.utils import plot_model
# from tensorflow.keras import backend as K

# # ---------------------------
# # 1. Create labeled sequences
# # ---------------------------
# def create_sequences(data, labels, time_steps=30):
#     X, y = [], []
#     for i in range(len(data) - time_steps):
#         X.append(data[i:i + time_steps])
#         y.append(labels[i + time_steps])  # label is at end of sequence
#     return np.array(X), np.array(y)

# # ---------------------------
# # 2. Load & preprocess data
# # ---------------------------
# df = pd.read_csv('smart_manufacturing_data.csv')
# df['timestamp'] = pd.to_datetime(df['timestamp'])
# df.set_index('timestamp', inplace=True)

# df['Is_Failure'] = (df['failure_type'] != 'Normal').astype(int)

# sensor_cols = ['temperature', 'vibration', 'humidity', 'pressure', 'energy_consumption', 'current', 'voltage']
# sensor_cols = [col for col in sensor_cols if col in df.columns]
# df = df[sensor_cols + ['Is_Failure']].fillna(0)

# # Split chronologically
# split_idx = int(len(df) * 0.8)
# train_df = df.iloc[:split_idx]
# test_df = df.iloc[split_idx:]

# # Scale features
# scaler = StandardScaler()
# train_scaled = train_df.copy()
# test_scaled = test_df.copy()
# train_scaled[sensor_cols] = scaler.fit_transform(train_df[sensor_cols])
# test_scaled[sensor_cols] = scaler.transform(test_df[sensor_cols])

# # ---------------------------
# # 3. Create sequences
# # ---------------------------
# X_train, y_train = create_sequences(train_scaled[sensor_cols].values, train_scaled['Is_Failure'].values)
# X_test, y_test = create_sequences(test_scaled[sensor_cols].values, test_scaled['Is_Failure'].values)

# print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
# print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")

# # ---------------------------
# # 4. LSTM Model
# # ---------------------------
# model = Sequential([
#     LSTM(128, input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=False),
#     Dropout(0.3),
#     Dense(64, activation='relu'),
#     Dropout(0.2),
#     Dense(1, activation='sigmoid')
# ])
# model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# # Compute class weights to handle imbalance
# from sklearn.utils import class_weight
# weights = class_weight.compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
# class_weights = {0: weights[0], 1: weights[1]}
# print(f"Class weights: {class_weights}")

# history = model.fit(
#     X_train, y_train,
#     epochs=50,
#     batch_size=128,
#     validation_split=0.1,
#     class_weight=class_weights,
#     callbacks=[early_stop],
#     verbose=1
# )

# # ---------------------------
# # 5. Evaluation
# # ---------------------------
# y_pred_prob = model.predict(X_test).flatten()
# y_pred = (y_pred_prob > 0.5).astype(int)

# print("\nClassification Report:")
# print(classification_report(y_test, y_pred, target_names=['Normal', 'Failure'], zero_division=0))

# conf_matrix = confusion_matrix(y_test, y_pred)
# sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Normal', 'Failure'], yticklabels=['Normal', 'Failure'])
# plt.title("Confusion Matrix")
# plt.xlabel("Predicted")
# plt.ylabel("Actual")
# plt.savefig("confusion_matrix_lstm_classifier.png")
# plt.show()

# # ---------------------------
# # 6. Plot Training History
# # ---------------------------
# plt.plot(history.history['loss'], label='Training Loss')
# plt.plot(history.history['val_loss'], label='Validation Loss')
# plt.title('Loss Over Epochs')
# plt.xlabel('Epoch')
# plt.ylabel('Loss')
# plt.legend()
# plt.grid(True)
# plt.savefig("training_loss_lstm_classifier.png")
# plt.show()

# # Optional: Save model
# model.save("lstm_failure_classifier.h5")
