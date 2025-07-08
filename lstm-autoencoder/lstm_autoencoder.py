from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, RepeatVector, TimeDistributed, Dense

def create_lstm_autoencoder(sequence_length, num_features):
    inputs = Input(shape=(sequence_length, num_features))
    encoded = LSTM(128, activation='relu')(inputs)
    repeated = RepeatVector(sequence_length)(encoded)
    decoded = LSTM(128, activation='relu', return_sequences=True)(repeated)
    outputs = TimeDistributed(Dense(num_features))(decoded)

    model = Model(inputs, outputs)
    model.compile(optimizer='adam', loss='mse')
    return model
