# lstm_model.py

# Function: create_lstm_model(input_shape)
# What it does:
# Builds a simple LSTM network:
# 1 LSTM layer with 64 units → learns time dependencies
# 1 Dense output layer → gives next week's prediction
# Uses mse loss (good for regression)
# Uses adam optimizer (good balance of speed + accuracy)

# Workflow
# Input Sequence (4 weeks) →
# LSTM (64 memory units + tanh gate functions) →
# Dense (1 output unit for next week's forecast)

# LSTM retains memory over time using 3 gates:
# Forget gate (what to discard)
# Input gate (what to keep)
# Output gate (what to emit)
# The tanh activation helps map values between -1 and 1, ideal when paired with MinMaxScaler

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

def create_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(units=64, activation='tanh', input_shape=input_shape))
    model.add(Dense(1))  # One output value
    model.compile(optimizer='adam', loss='mse')
    return model
 