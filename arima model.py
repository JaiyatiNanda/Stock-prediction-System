import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
import math
import pandas as pd

# Load data
aapl = pd.read_csv('historical_data.csv')

# Check for missing values in the dataset
print(aapl.isnull().sum())

# Drop rows with missing values
aapl.dropna(inplace=True)

# Convert 'Date' column to datetime format and set as index
aapl['Date'] = pd.to_datetime(aapl['Date'], format='%b %d, %Y')
aapl.set_index('Date', inplace=True)

# Visualize stock price data
plt.figure(figsize=(10, 6))
plt.plot(aapl['Close'], label='Stock Price')
plt.title('Stock Price of Apple')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.show()

# Select the 'Close' prices
data = aapl['Close']

# Split the data into train and test sets
train_size = int(len(data) * 0.7)
train, test = data[:train_size], data[train_size:]

# Fit the ARIMA model
model = ARIMA(train, order=(5, 1, 0))
model_fit = model.fit()

# Print the model summary
print(model_fit.summary())

# Make predictions on the training data
train_pred = model_fit.predict(start=0, end=len(train)-1, typ='levels')

# Make predictions on the test data
test_pred = model_fit.predict(start=len(train), end=len(train)+len(test)-1, typ='levels')

# Calculate RMSE for training and test sets
rmse_train = math.sqrt(mean_squared_error(train, train_pred))
rmse_test = math.sqrt(mean_squared_error(test, test_pred))

print(f'Root Mean Squared Error (Training): {rmse_train}')
print(f'Root Mean Squared Error (Testing): {rmse_test}')

# Plot the actual and predicted values
plt.figure(figsize=(10, 6))
plt.plot(train, label='Train')
plt.plot(test, label='Test')
plt.plot(train.index, train_pred, label='Train Predictions')
plt.plot(test.index, test_pred, label='Test Predictions')
plt.title('Actual vs Predicted Stock Prices')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.show()