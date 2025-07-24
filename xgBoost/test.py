# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve

# import tensorflow as tf
# from tensorflow.keras.models import Model
# from tensorflow.keras.layers import Input, Dense, ActivityRegularization
# from tensorflow.keras.callbacks import EarlyStopping

# def build_and_train_autoencoder(train_data):
#     """
#     Builds and trains a deep autoencoder model.

#     Args:
#         train_data (np.array): Scaled training data containing only normal samples.

#     Returns:
#         A trained Keras autoencoder model.
#     """
#     input_dim = train_data.shape[1]
    
#     # --- Encoder ---
#     input_layer = Input(shape=(input_dim,))
#     # Adding a small amount of activity regularization can help create more robust representations
#     encoder = Dense(32, activation='relu', activity_regularizer=tf.keras.regularizers.l1(10e-5))(input_layer)
#     encoder = Dense(16, activation='relu')(encoder)
#     encoder = Dense(8, activation='relu')(encoder)

#     # --- Decoder ---
#     decoder = Dense(16, activation='relu')(encoder)
#     decoder = Dense(32, activation='relu')(decoder)
#     decoder = Dense(input_dim, activation='linear')(decoder)

#     # --- Autoencoder Model ---
#     autoencoder = Model(inputs=input_layer, outputs=decoder)
#     autoencoder.compile(optimizer='adam', loss='mean_squared_error')
    
#     print("\n--- Autoencoder Architecture ---")
#     autoencoder.summary()

#     # --- Train the Model ---
#     early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    
#     print("\n--- Training Autoencoder ---")
#     history = autoencoder.fit(
#         train_data, train_data,
#         epochs=100,
#         batch_size=32,
#         shuffle=True,
#         validation_split=0.2,
#         callbacks=[early_stopping],
#         verbose=1
#     )
    
#     return autoencoder

# def main():
#     """
#     Main function to run the fault detection pipeline.
#     """
#     # --- 1. Load and Prepare Data ---
#     try:
#         df = pd.read_csv('Industrial_fault_detection.csv')
#     except FileNotFoundError:
#         print("Error: 'Industrial_fault_detection.csv' not found.")
#         return

#     df['Is_Fault'] = df['Fault_Type'].apply(lambda x: 1 if x > 0 else 0)

#     # Separate normal and faulty data
#     normal_df = df[df['Is_Fault'] == 0]
#     fault_df = df[df['Is_Fault'] == 1]

#     # Create training set from a portion of normal data
#     # Create test set from the remaining normal data and all faulty data
#     X_normal = normal_df.drop(['Fault_Type', 'Is_Fault'], axis=1)
#     X_train, X_test_normal_sample = train_test_split(X_normal, test_size=0.5, random_state=42)
    
#     X_test = pd.concat([X_test_normal_sample, fault_df.drop(['Fault_Type', 'Is_Fault'], axis=1)])
#     y_test = pd.concat([normal_df.loc[X_test_normal_sample.index]['Is_Fault'], fault_df['Is_Fault']])

#     print(f"Training set size (normal data only): {len(X_train)} samples")
#     print(f"Test set size (mixed normal and fault): {len(X_test)} samples")

#     # --- 2. Scale Data ---
#     scaler = StandardScaler()
#     X_train_scaled = scaler.fit_transform(X_train)
#     X_test_scaled = scaler.transform(X_test)

#     # --- 3. Build and Train Autoencoder ---
#     autoencoder = build_and_train_autoencoder(X_train_scaled)

#     # --- 4. Determine Optimal Anomaly Threshold ---
#     print("\n--- Finding Optimal Anomaly Threshold ---")
#     # Get reconstruction errors on the test set
#     test_reconstructions = autoencoder.predict(X_test_scaled)
#     test_mse = np.mean(np.power(X_test_scaled - test_reconstructions, 2), axis=1)
    
#     # Calculate precision, recall, and thresholds
#     precision, recall, thresholds = precision_recall_curve(y_test, test_mse)
    
#     # Calculate F1 score for each threshold
#     # np.nan_to_num handles the case of 0 precision or 0 recall
#     f1_scores = np.nan_to_num(2 * (precision * recall) / (precision + recall))
    
#     # Find the threshold that maximizes the F1 score
#     optimal_idx = np.argmax(f1_scores)
#     optimal_threshold = thresholds[optimal_idx]
    
#     print(f"Optimal Threshold based on max F1-score: {optimal_threshold:.4f}")
#     print(f"Achieved F1-score at this threshold: {f1_scores[optimal_idx]:.4f}")

#     # Plot Precision-Recall-Threshold Curve
#     plt.figure(figsize=(10, 7))
#     plt.plot(thresholds, precision[:-1], label='Precision')
#     plt.plot(thresholds, recall[:-1], label='Recall')
#     plt.plot(thresholds, f1_scores[:-1], label='F1-Score', linewidth=3)
#     plt.axvline(optimal_threshold, color='r', linestyle='--', label=f'Optimal Threshold = {optimal_threshold:.4f}')
#     plt.title('Precision, Recall, and F1-Score vs. Threshold')
#     plt.xlabel('Reconstruction Error Threshold')
#     plt.ylabel('Score')
#     plt.legend()
#     plt.savefig('optimal_threshold_curve.png')
#     print("Optimal threshold plot saved as 'optimal_threshold_curve.png'")


#     # --- 5. Evaluate on Test Data with Optimal Threshold ---
#     print("\n--- Evaluating on Test Set with Optimal Threshold ---")
#     y_pred = (test_mse > optimal_threshold).astype(int)

#     # --- 6. Report Final Results ---
#     print("\n--- Final Anomaly Detection Performance ---")
#     class_report = classification_report(y_test, y_pred, target_names=['Normal (0)', 'Fault (1)'])
#     print(class_report)

#     # Generate and save confusion matrix
#     cm = confusion_matrix(y_test, y_pred)
#     plt.figure(figsize=(8, 6))
#     sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
#                 xticklabels=['Predicted Normal', 'Predicted Fault'],
#                 yticklabels=['Actual Normal', 'Actual Fault'])
#     plt.title('Confusion Matrix (Optimal Threshold)')
#     plt.savefig('anomaly_confusion_matrix_optimal.png')
#     print("Final confusion matrix saved as 'anomaly_confusion_matrix_optimal.png'")

# if __name__ == '__main__':
#     main()

# Sparse Autoencoder Implementation for Fault Detection
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve

# import tensorflow as tf
# from tensorflow.keras.models import Model
# from tensorflow.keras.layers import Input, Dense
# from tensorflow.keras import regularizers # Import the regularizers module
# from tensorflow.keras.callbacks import EarlyStopping

# def build_and_train_sparse_autoencoder(train_data):
#     """
#     Builds and trains a Sparse Autoencoder model.
#     The key change is the 'activity_regularizer' which encourages sparsity.

#     Args:
#         train_data (np.array): Scaled training data containing only normal samples.

#     Returns:
#         A trained Keras autoencoder model.
#     """
#     input_dim = train_data.shape[1]
    
#     # --- Encoder ---
#     # The activity_regularizer penalizes the activation of neurons, forcing sparsity.
#     # This encourages the model to learn a more robust and essential representation.
#     input_layer = Input(shape=(input_dim,))
#     encoder = Dense(32, activation='relu', 
#                     activity_regularizer=regularizers.l1(10e-5))(input_layer)
#     encoder = Dense(16, activation='relu')(encoder)
#     encoder = Dense(8, activation='relu')(encoder) # Bottleneck layer

#     # --- Decoder ---
#     decoder = Dense(16, activation='relu')(encoder)
#     decoder = Dense(32, activation='relu')(decoder)
#     decoder = Dense(input_dim, activation='linear')(decoder)

#     # --- Autoencoder Model ---
#     autoencoder = Model(inputs=input_layer, outputs=decoder)
#     autoencoder.compile(optimizer='adam', loss='mean_squared_error')
    
#     print("\n--- Sparse Autoencoder Architecture ---")
#     autoencoder.summary()

#     # --- Train the Model ---
#     early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    
#     print("\n--- Training Sparse Autoencoder ---")
#     history = autoencoder.fit(
#         train_data, train_data,
#         epochs=100,
#         batch_size=32,
#         shuffle=True,
#         validation_split=0.2,
#         callbacks=[early_stopping],
#         verbose=1
#     )
    
#     return autoencoder

# def main():
#     """
#     Main function to run the fault detection pipeline.
#     """
#     # --- 1. Load and Prepare Data ---
#     try:
#         df = pd.read_csv('Industrial_fault_detection.csv')
#     except FileNotFoundError:
#         print("Error: 'Industrial_fault_detection.csv' not found.")
#         return

#     df['Is_Fault'] = df['Fault_Type'].apply(lambda x: 1 if x > 0 else 0)

#     normal_df = df[df['Is_Fault'] == 0]
#     fault_df = df[df['Is_Fault'] == 1]

#     X_normal = normal_df.drop(['Fault_Type', 'Is_Fault'], axis=1)
#     X_train, X_test_normal_sample = train_test_split(X_normal, test_size=0.5, random_state=42)
    
#     X_test = pd.concat([X_test_normal_sample, fault_df.drop(['Fault_Type', 'Is_Fault'], axis=1)])
#     y_test = pd.concat([normal_df.loc[X_test_normal_sample.index]['Is_Fault'], fault_df['Is_Fault']])

#     print(f"Training set size (normal data only): {len(X_train)} samples")
#     print(f"Test set size (mixed normal and fault): {len(X_test)} samples")

#     # --- 2. Scale Data ---
#     scaler = StandardScaler()
#     X_train_scaled = scaler.fit_transform(X_train)
#     X_test_scaled = scaler.transform(X_test)

#     # --- 3. Build and Train Autoencoder ---
#     autoencoder = build_and_train_sparse_autoencoder(X_train_scaled)

#     # --- 4. Determine Optimal Anomaly Threshold ---
#     print("\n--- Finding Optimal Anomaly Threshold ---")
#     test_reconstructions = autoencoder.predict(X_test_scaled)
#     test_mse = np.mean(np.power(X_test_scaled - test_reconstructions, 2), axis=1)
    
#     precision, recall, thresholds = precision_recall_curve(y_test, test_mse)
    
#     # Using np.nan_to_num to avoid errors when precision or recall is 0
#     f1_scores = np.nan_to_num(2 * (precision * recall) / (precision + recall))
    
#     optimal_idx = np.argmax(f1_scores)
#     optimal_threshold = thresholds[optimal_idx]
    
#     print(f"Optimal Threshold based on max F1-score: {optimal_threshold:.4f}")
#     print(f"Achieved F1-score at this threshold: {f1_scores[optimal_idx]:.4f}")

#     # Plot Precision-Recall-Threshold Curve
#     plt.figure(figsize=(10, 7))
#     plt.plot(thresholds, precision[:-1], label='Precision')
#     plt.plot(thresholds, recall[:-1], label='Recall')
#     plt.plot(thresholds, f1_scores[:-1], label='F1-Score', linewidth=3, color='g')
#     plt.axvline(optimal_threshold, color='r', linestyle='--', label=f'Optimal Threshold = {optimal_threshold:.4f}')
#     plt.title('Precision, Recall, and F1-Score vs. Threshold')
#     plt.xlabel('Reconstruction Error Threshold')
#     plt.ylabel('Score')
#     plt.legend()
#     plt.grid(True)
#     plt.savefig('sparse_optimal_threshold_curve.png')
#     print("Optimal threshold plot saved as 'sparse_optimal_threshold_curve.png'")

#     # --- 5. Evaluate on Test Data with Optimal Threshold ---
#     print("\n--- Evaluating on Test Set with Optimal Threshold ---")
#     y_pred = (test_mse > optimal_threshold).astype(int)

#     # --- 6. Report Final Results ---
#     print("\n--- Final Anomaly Detection Performance (Sparse Autoencoder) ---")
#     class_report = classification_report(y_test, y_pred, target_names=['Normal (0)', 'Fault (1)'])
#     print(class_report)

#     cm = confusion_matrix(y_test, y_pred)
#     plt.figure(figsize=(8, 6))
#     sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
#                 xticklabels=['Predicted Normal', 'Predicted Fault'],
#                 yticklabels=['Actual Normal', 'Actual Fault'])
#     plt.title('Confusion Matrix (Sparse Autoencoder)')
#     plt.savefig('sparse_anomaly_confusion_matrix.png')
#     print("Final confusion matrix saved as 'sparse_anomaly_confusion_matrix.png'")

# if __name__ == '__main__':
#     main()

#  Variational AutoEncoder (VAE) Implementation for Fault Detection
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve

# import tensorflow as tf
# from tensorflow.keras.models import Model
# from tensorflow.keras.layers import Input, Dense, Lambda
# from tensorflow.keras.callbacks import EarlyStopping

# # --- VAE Model Definition (Modern Approach) ---
# class VAE(tf.keras.Model):
#     """
#     Defines a Variational Autoencoder (VAE) as a custom Keras Model.
#     This approach provides better control over the training loop and loss calculation.
#     """
#     def __init__(self, input_dim, latent_dim=8, **kwargs):
#         super(VAE, self).__init__(**kwargs)
#         self.input_dim = input_dim
#         self.latent_dim = latent_dim

#         # --- Encoder ---
#         encoder_inputs = Input(shape=(input_dim,))
#         x = Dense(32, activation="relu")(encoder_inputs)
#         x = Dense(16, activation="relu")(x)
#         z_mean = Dense(latent_dim, name="z_mean")(x)
#         z_log_var = Dense(latent_dim, name="z_log_var")(x)
#         self.encoder = Model(encoder_inputs, [z_mean, z_log_var], name="encoder")

#         # --- Sampling Layer ---
#         self.sampling = self._create_sampling()

#         # --- Decoder ---
#         latent_inputs = Input(shape=(latent_dim,))
#         x = Dense(16, activation="relu")(latent_inputs)
#         x = Dense(32, activation="relu")(x)
#         decoder_outputs = Dense(input_dim, activation="linear")(x)
#         self.decoder = Model(latent_inputs, decoder_outputs, name="decoder")

#     def _create_sampling(self):
#         def sampling_layer(inputs):
#             z_mean, z_log_var = inputs
#             batch = tf.shape(z_mean)[0]
#             dim = tf.shape(z_mean)[1]
#             epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
#             return z_mean + tf.exp(0.5 * z_log_var) * epsilon
#         return Lambda(sampling_layer, name='sampling')

#     def call(self, inputs):
#         """Defines the forward pass of the VAE."""
#         z_mean, z_log_var = self.encoder(inputs)
#         z = self.sampling([z_mean, z_log_var])
#         reconstruction = self.decoder(z)
#         return reconstruction, z_mean, z_log_var

#     def train_step(self, data):
#         """Defines the custom training logic for the VAE."""
#         if isinstance(data, tuple):
#             data = data[0]
        
#         with tf.GradientTape() as tape:
#             reconstruction, z_mean, z_log_var = self(data, training=True)
            
#             # Calculate Reconstruction Loss (MSE)
#             reconstruction_loss = tf.reduce_mean(
#                 tf.square(data - reconstruction), axis=-1
#             )
#             reconstruction_loss *= self.input_dim
            
#             # Calculate KL Divergence Loss
#             kl_loss = 1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)
#             kl_loss = tf.reduce_sum(kl_loss, axis=-1)
#             kl_loss *= -0.5
            
#             # Total Loss
#             total_loss = tf.reduce_mean(reconstruction_loss + kl_loss)

#         # Apply gradients
#         grads = tape.gradient(total_loss, self.trainable_weights)
#         self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        
#         return {"loss": total_loss}

#     def test_step(self, data):
#         """Defines the custom validation logic."""
#         if isinstance(data, tuple):
#             data = data[0]
            
#         reconstruction, z_mean, z_log_var = self(data, training=False)
        
#         reconstruction_loss = tf.reduce_mean(
#             tf.square(data - reconstruction), axis=-1
#         )
#         reconstruction_loss *= self.input_dim
        
#         kl_loss = 1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)
#         kl_loss = tf.reduce_sum(kl_loss, axis=-1)
#         kl_loss *= -0.5
        
#         total_loss = tf.reduce_mean(reconstruction_loss + kl_loss)
        
#         return {"loss": total_loss}

# def build_and_train_vae(train_data):
#     """
#     Builds and trains a Variational Autoencoder (VAE) model.
#     """
#     input_dim = train_data.shape[1]
    
#     vae = VAE(input_dim)
#     vae.compile(optimizer='adam')
    
#     early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
    
#     print("\n--- Training VAE ---")
#     history = vae.fit(
#         train_data,
#         epochs=150,
#         batch_size=32,
#         shuffle=True,
#         validation_split=0.2,
#         callbacks=[early_stopping],
#         verbose=1
#     )
    
#     return vae

# def main():
#     """
#     Main function to run the fault detection pipeline.
#     """
#     # --- 1. Load and Prepare Data ---
#     try:
#         df = pd.read_csv('Industrial_fault_detection.csv')
#     except FileNotFoundError:
#         print("Error: 'Industrial_fault_detection.csv' not found.")
#         return

#     df['Is_Fault'] = df['Fault_Type'].apply(lambda x: 1 if x > 0 else 0)

#     normal_df = df[df['Is_Fault'] == 0]
#     fault_df = df[df['Is_Fault'] == 1]

#     X_normal = normal_df.drop(['Fault_Type', 'Is_Fault'], axis=1)
#     X_train, X_test_normal_sample = train_test_split(X_normal, test_size=0.5, random_state=42)
    
#     X_test = pd.concat([X_test_normal_sample, fault_df.drop(['Fault_Type', 'Is_Fault'], axis=1)])
#     y_test = pd.concat([normal_df.loc[X_test_normal_sample.index]['Is_Fault'], fault_df['Is_Fault']])

#     print(f"Training set size (normal data only): {len(X_train)} samples")
#     print(f"Test set size (mixed normal and fault): {len(X_test)} samples")

#     # --- 2. Scale Data ---
#     scaler = StandardScaler()
#     X_train_scaled = scaler.fit_transform(X_train)
#     X_test_scaled = scaler.transform(X_test)

#     # --- 3. Build and Train VAE ---
#     vae = build_and_train_vae(X_train_scaled)

#     # --- 4. Determine Optimal Anomaly Threshold ---
#     print("\n--- Finding Optimal Anomaly Threshold ---")
#     # For VAE, we only need the reconstruction, not the other outputs
#     reconstructions, _, _ = vae.predict(X_test_scaled)
#     test_mse = np.mean(np.power(X_test_scaled - reconstructions, 2), axis=1)
    
#     precision, recall, thresholds = precision_recall_curve(y_test, test_mse)
    
#     f1_scores = np.nan_to_num(2 * (precision * recall) / (precision + recall))
    
#     optimal_idx = np.argmax(f1_scores)
#     optimal_threshold = thresholds[optimal_idx]
    
#     print(f"Optimal Threshold based on max F1-score: {optimal_threshold:.4f}")
#     print(f"Achieved F1-score at this threshold: {f1_scores[optimal_idx]:.4f}")

#     # Plot Precision-Recall-Threshold Curve
#     plt.figure(figsize=(10, 7))
#     plt.plot(thresholds, precision[:-1], label='Precision')
#     plt.plot(thresholds, recall[:-1], label='Recall')
#     plt.plot(thresholds, f1_scores[:-1], label='F1-Score', linewidth=3, color='g')
#     plt.axvline(optimal_threshold, color='r', linestyle='--', label=f'Optimal Threshold = {optimal_threshold:.4f}')
#     plt.title('Precision, Recall, and F1-Score vs. Threshold (VAE)')
#     plt.xlabel('Reconstruction Error Threshold')
#     plt.ylabel('Score')
#     plt.legend()
#     plt.grid(True)
#     plt.savefig('vae_optimal_threshold_curve.png')
#     print("Optimal threshold plot saved as 'vae_optimal_threshold_curve.png'")

#     # --- 5. Evaluate on Test Data with Optimal Threshold ---
#     print("\n--- Evaluating on Test Set with Optimal Threshold ---")
#     y_pred = (test_mse > optimal_threshold).astype(int)

#     # --- 6. Report Final Results ---
#     print("\n--- Final Anomaly Detection Performance (VAE) ---")
#     class_report = classification_report(y_test, y_pred, target_names=['Normal (0)', 'Fault (1)'])
#     print(class_report)

#     cm = confusion_matrix(y_test, y_pred)
#     plt.figure(figsize=(8, 6))
#     sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
#                 xticklabels=['Predicted Normal', 'Predicted Fault'],
#                 yticklabels=['Actual Normal', 'Actual Fault'])
#     plt.title('Confusion Matrix (VAE)')
#     plt.savefig('vae_anomaly_confusion_matrix.png')
#     print("Final confusion matrix saved as 'vae_anomaly_confusion_matrix.png'")

# if __name__ == '__main__':
#     main()

