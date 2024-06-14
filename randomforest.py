import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error

# Load data
aapl = pd.read_csv('C:/Users/admin/PycharmProjects/web scraping/historical_data.csv')

# Check for missing values in the dataset
print(aapl.isnull().sum())

# Drop rows with missing values
aapl.dropna(inplace=True)

# split data into train and test set
X = np.array(aapl.index).reshape(-1, 1)
Y = aapl['Close']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=101)

# Standardize features
scaler = StandardScaler().fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define hyperparameters for tuning
param_grid = {
    'n_estimators': [50, 100, 150],  # Number of trees in the forest
    'max_depth': [None, 10, 20],       # Maximum depth of the trees
    'min_samples_split': [2, 5, 10]    # Minimum number of samples required to split a node
}

# Create Random Forest model
rf_model = RandomForestRegressor(random_state=101)

# Perform hyperparameter tuning using GridSearchCV
grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid, cv=5, scoring='r2')
grid_search.fit(X_train_scaled, Y_train)

# Get the best hyperparameters and model
best_params = grid_search.best_params_
best_model = grid_search.best_estimator_

# Make predictions
Y_train_pred = best_model.predict(X_train_scaled)
Y_test_pred = best_model.predict(X_test_scaled)

# Evaluate the model
r2_train = r2_score(Y_train, Y_train_pred)
r2_test = r2_score(Y_test, Y_test_pred)
mse_train = mean_squared_error(Y_train, Y_train_pred)
mse_test = mean_squared_error(Y_test, Y_test_pred)

# Print the evaluation metrics
print("Best Hyperparameters:", best_params)
print(f'R-squared score (Training): {r2_train}')
print(f'R-squared score (Testing): {r2_test}')
print(f'Mean Squared Error (Training): {mse_train}')
print(f'Mean Squared Error (Testing): {mse_test}')

# Plot predictions
plt.figure(figsize=(10, 6))
plt.scatter(X_train, Y_train, color='blue', label='Actual (Training)')
plt.scatter(X_test, Y_test, color='green', label='Actual (Testing)')
plt.plot(X_train, Y_train_pred, color='red', label='Predicted (Training)')
plt.plot(X_test, Y_test_pred, color='orange', label='Predicted (Testing)')
plt.title('Actual vs. Predicted Stock Prices (Random Forest)')
plt.xlabel('Day')
plt.ylabel('Price')
plt.legend()
plt.show()
