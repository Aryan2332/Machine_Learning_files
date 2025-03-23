import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt


# Load the Airline Passengers dataset (assuming a CSV file with 'Month' and 'Passengers' columns)
def load_airline_data(file_path):
    df = pd.read_csv("/content/airline-passengers.csv")
    # Convert 'Month' to datetime if it's in string format (adjust if needed)
    df['Month'] = pd.to_datetime(df['Month'])
    return df['Passengers'].values  # We are interested in the 'Passengers' column


# Function to create data sequences for RNN
def create_dataset(data, window_size=10):
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i:i + window_size])
        y.append(data[i + window_size])
    return np.array(X), np.array(y)


# Load and preprocess dataset
file_path = 'airline-passenger-data.csv'  # Change this to the actual dataset file path
data = load_airline_data(file_path)

# Normalize data to the range [0, 1]
scaler = MinMaxScaler(feature_range=(0, 1))
data_scaled = scaler.fit_transform(data.reshape(-1, 1))  # Reshaping for scaler

# Create sequences from the data
window_size = 20  # You can adjust this window size based on the data
X, y = create_dataset(data_scaled, window_size)

# Reshape the target variable to ensure it matches the input shape
y = y.reshape(-1, 1)  # Reshaping y to be 2D (samples, 1)

# Split the data into training and test sets
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Reshape data for RNN input (samples, time steps, features)
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

# Build the RNN model
model = Sequential()
model.add(SimpleRNN(50, activation='relu', input_shape=(X_train.shape[1], 1)))
model.add(Dense(1))  # Output layer to predict a single value
model.compile(optimizer=Adam(), loss='mean_squared_error')

# Train the model
model.fit(X_train, y_train, epochs=20, batch_size=32, verbose=1)

# Evaluate the model
train_loss = model.evaluate(X_train, y_train)
test_loss = model.evaluate(X_test, y_test)

print(f"Training Loss: {train_loss}")
print(f"Test Loss: {test_loss}")


# Make predictions
def predict_future(model, data, window_size=20, steps=5):
    """
    Predict the future values based on the trained model.

    Args:
    model: The trained RNN model.
    data: The dataset used for prediction (scaled).
    window_size: The number of previous timesteps to use for prediction.
    steps: The number of future steps to predict.

    Returns:
    Predicted future values.
    """
    # Get the last window of data from the input dataset
    input_sequence = data[-window_size:].reshape((1, window_size, 1))

    future_predictions = []

    for _ in range(steps):
        # Predict the next value
        next_value = model.predict(input_sequence)[0][0]
        future_predictions.append(next_value)

        # Reshape next_value to match the shape of the input_sequence (1, 1, 1)
        next_value_reshaped = np.array(next_value).reshape(1, 1, 1)

        # Append the predicted value to the sequence and update the input for next prediction
        input_sequence = np.append(input_sequence[:, 1:, :], next_value_reshaped, axis=1)

    # Inverse transform the predictions back to the original scale
    future_predictions_rescaled = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))
    return future_predictions_rescaled


# Predict the next 5 months
future_steps = 5
future_predictions = predict_future(model, data_scaled, window_size, future_steps)

print(f"Future predictions (next {future_steps} months): {future_predictions.flatten()}")

# Plot the predictions vs actual data
plt.figure(figsize=(10, 6))
plt.plot(np.arange(len(data)), scaler.inverse_transform(data_scaled), label="Actual Data")
plt.plot(np.arange(len(data), len(data) + future_steps), future_predictions.flatten(), label="Future Predictions",
         linestyle='--')
plt.legend()
plt.title('Airline Passengers: Actual vs Predicted Future Values')
plt.xlabel('Time')
plt.ylabel('Passengers')
plt.show()
