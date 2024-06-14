import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from statsmodels.tsa.arima.model import ARIMA
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
import math

# Load data
aapl = pd.read_csv('historical_data.csv')
aapl.dropna(inplace=True)
aapl['Date'] = pd.to_datetime(aapl['Date'], format='%b %d, %Y')
aapl.set_index('Date', inplace=True)

# Prepare the data for Random Forest and Linear models
data = aapl['Close'].values.reshape(-1, 1)
X = np.arange(len(data)).reshape(-1, 1)  # Features: days as integer
y = data  # Target: stock prices

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

# Train the Random Forest Regressor
rf_model = RandomForestRegressor(n_estimators=100, random_state=101)
rf_model.fit(X_train, y_train.ravel())

# Make predictions with Random Forest
y_train_pred_rf = rf_model.predict(X_train)
y_test_pred_rf = rf_model.predict(X_test)

# Train the Linear Regression model
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

# Make predictions with Linear Regression
y_train_pred_lr = lr_model.predict(X_train)
y_test_pred_lr = lr_model.predict(X_test)

# Train and predict with ARIMA model
train_size = int(len(data) * 0.7)
train_arima, test_arima = data[:train_size], data[train_size:]

# Fit ARIMA model
model_arima = ARIMA(train_arima, order=(5, 1, 0))
model_fit_arima = model_arima.fit()

# Make predictions with ARIMA
train_pred_arima = model_fit_arima.predict(start=0, end=len(train_arima)-1, typ='levels')
test_pred_arima = model_fit_arima.predict(start=len(train_arima), end=len(train_arima)+len(test_arima)-1, typ='levels')

# Prepare the data for LSTM model
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(aapl[['Close']])

def create_sequences(data, seq_length):
    sequences = []
    labels = []
    for i in range(len(data) - seq_length):
        sequences.append(data[i:i + seq_length])
        labels.append(data[i + seq_length])
    return np.array(sequences), np.array(labels)

SEQ_LENGTH = 60
X_lstm, y_lstm = create_sequences(scaled_data, SEQ_LENGTH)

X_train_lstm, X_test_lstm, y_train_lstm, y_test_lstm = train_test_split(X_lstm, y_lstm, test_size=0.3, random_state=101)

# Build the LSTM model
model_lstm = Sequential()
model_lstm.add(LSTM(units=50, return_sequences=True, input_shape=(SEQ_LENGTH, 1)))
model_lstm.add(Dropout(0.2))
model_lstm.add(LSTM(units=50, return_sequences=False))
model_lstm.add(Dropout(0.2))
model_lstm.add(Dense(units=1))

model_lstm.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
model_lstm.fit(X_train_lstm, y_train_lstm, epochs=50, batch_size=32, validation_split=0.2, verbose=1)

# Make predictions with LSTM
y_train_pred_lstm = model_lstm.predict(X_train_lstm)
y_test_pred_lstm = model_lstm.predict(X_test_lstm)

y_train_pred_lstm = scaler.inverse_transform(y_train_pred_lstm)
y_test_pred_lstm = scaler.inverse_transform(y_test_pred_lstm)
y_train_lstm = scaler.inverse_transform(y_train_lstm.reshape(-1, 1))
y_test_lstm = scaler.inverse_transform(y_test_lstm.reshape(-1, 1))

# Plot the results
plt.figure(figsize=(14, 8))

# Actual prices
plt.plot(aapl.index, data, label='Actual Prices', color='blue')

# Predictions of Random Forest
plt.scatter(aapl.index[:len(y_train)], y_train_pred_rf, label='Training Predictions (RF)', color='green')
plt.scatter(aapl.index[len(y_train):], y_test_pred_rf, label='Testing Predictions (RF)', color='red')

# Predictions of ARIMA
plt.plot(aapl.index[:train_size], train_pred_arima, label='Training Predictions (ARIMA)', color='cyan')
plt.plot(aapl.index[train_size:], test_pred_arima, label='Testing Predictions (ARIMA)', color='magenta')

# Predictions of LSTM
plt.scatter(aapl.index[SEQ_LENGTH:SEQ_LENGTH+len(y_train_pred_lstm)], y_train_pred_lstm, label='Training Predictions (LSTM)', color='orange')
plt.scatter(aapl.index[SEQ_LENGTH+len(y_train_pred_lstm):SEQ_LENGTH+len(y_train_pred_lstm)+len(y_test_pred_lstm)], y_test_pred_lstm, label='Testing Predictions (LSTM)', color='purple')

# Predictions of Linear Regression
plt.scatter(aapl.index[:len(y_train)], y_train_pred_lr, label='Training Predictions (LR)', color='yellow')
plt.scatter(aapl.index[len(y_train):], y_test_pred_lr, label='Testing Predictions (LR)', color='brown')

plt.title('Actual vs Predicted Stock Prices')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.show()
