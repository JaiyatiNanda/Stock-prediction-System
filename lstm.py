import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
from keras.models import Sequential
from keras.layers import LSTM, Dense

# Load data
aapl = pd.read_csv('C:/Users/admin/PycharmProjects/web scraping/historical_data.csv')

# Check for missing values in the dataset
print(aapl.isnull().sum())

# Drop rows with missing values
aapl.dropna(inplace=True)

# Prepare data for LSTM
X = aapl['Close'].values.reshape(-1, 1)
scaler = StandardScaler().fit(X)
X_scaled = scaler.transform(X)

def create_dataset(X, time_steps=1):
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        Xs.append(X[i:(i + time_steps), 0])
        ys.append(X[i + time_steps, 0])
    return np.array(Xs), np.array(ys)

TIME_STEPS = 10
X_lstm, y_lstm = create_dataset(X_scaled, TIME_STEPS)

# Split data into train and test set
X_train, X_test, y_train, y_test = train_test_split(X_lstm, y_lstm, test_size=0.3, random_state=101)

# Define LSTM model
model = Sequential()
model.add(LSTM(units=50, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')

# Train the model
history = model.fit(X_train, y_train, epochs=50, batch_size=16, validation_split=0.1, verbose=1)

# Make predictions
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

# Inverse transform predictions to original scale
y_train_pred_inv = scaler.inverse_transform(y_train_pred)
y_test_pred_inv = scaler.inverse_transform(y_test_pred)
y_train_inv = scaler.inverse_transform(y_train.reshape(-1, 1))
y_test_inv = scaler.inverse_transform(y_test.reshape(-1, 1))

# Calculate R-squared score
r2_train = r2_score(y_train_inv, y_train_pred_inv)
r2_test = r2_score(y_test_inv, y_test_pred_inv)

# Calculate Mean Squared Error (MSE)
mse_train = mean_squared_error(y_train_inv, y_train_pred_inv)
mse_test = mean_squared_error(y_test_inv, y_test_pred_inv)

# Print the evaluation metrics
print(f'R-squared score (Training): {r2_train}')
print(f'R-squared score (Testing): {r2_test}')
print(f'Mean Squared Error (Training): {mse_train}')
print(f'Mean Squared Error (Testing): {mse_test}')

# Plot predictions
plt.figure(figsize=(10, 6))
plt.plot(y_train_inv, label='Actual (Training)', color='blue')
plt.plot(y_test_inv, label='Actual (Testing)', color='green')
plt.plot(y_train_pred_inv, label='Predicted (Training)', color='red')
plt.plot(range(len(y_train_inv), len(y_train_inv) + len(y_test_inv)), y_test_pred_inv, label='Predicted (Testing)', color='orange')
plt.title('Actual vs. Predicted Stock Prices (LSTM)')
plt.xlabel('Day')
plt.ylabel('Price')
plt.legend()
plt.show()
