import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, train_test_split

# Sample dataset (house prices based on square footage)
data = {
    'SquareFootage': [1500, 1800, 2400, 3000, 3500, 4000, 4500],
    'Price': [200000, 250000, 300000, 350000, 400000, 500000, 600000]
}

# Convert to DataFrame
df = pd.DataFrame(data)

# Display the first few rows of the data
print(df.head())

# Features (X) and Target (y)
X = df[['SquareFootage']]  # Feature(s)
y = df['Price']            # Target variable

# Split data into 80% training and 20% testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Display the shape of the training and testing sets
print(f"Training data: {X_train.shape}, {y_train.shape}")
print(f"Testing data: {X_test.shape}, {y_test.shape}")

# Initialize the Linear Regression model
model = LinearRegression()

# Train the model on the training data
model.fit(X_train, y_train)

# Display the learned coefficients
print(f"Intercept: {model.intercept_}")
print(f"Coefficient: {model.coef_[0]}")
print(4000 * model.coef_[0] + model.intercept_)

# Make predictions on the testing set
y_pred = model.predict(X_test)

# Display the predictions
print("Predicted Prices:", y_pred)
print("Actual Prices:", y_test.values)

# Calculate Mean Squared Error
mse = mean_squared_error(y_test, y_pred)

# Add RMSE and MAE for easier interpretation
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)

# Training metrics
y_train_pred = model.predict(X_train)
train_r2 = r2_score(y_train, y_train_pred)

# Baseline (predict using training mean) for comparison
baseline_pred = np.full_like(y_test, fill_value=y_train.mean(), dtype=float)
baseline_mse = mean_squared_error(y_test, baseline_pred)

# Cross-validation R^2 (more stable estimate)
cv_r2 = cross_val_score(model, X, y, cv=5, scoring='r2')

# Calculate R-squared
r2 = r2_score(y_test, y_pred)

# Display the evaluation metrics
print(f"Mean Squared Error: {mse}")
print(f"Root MSE: {rmse}")
print(f"Mean Absolute Error: {mae}")
print(f"R-squared (test): {r2}")
print(f"R-squared (train): {train_r2}")
print(f"Baseline MSE (predict train mean): {baseline_mse}")
print(f"5-fold CV R^2 scores: {cv_r2}")
print(f"5-fold CV R^2 (mean): {cv_r2.mean()}")

# Plot the data points
plt.scatter(X_test, y_test, color='blue', label='Actual Data')

# Plot the regression line
plt.plot(X_test, y_pred, color='red', label='Regression Line')

# Add labels and title
plt.xlabel('Square Footage')
plt.ylabel('Price')
plt.title('House Prices vs. Square Footage')
plt.legend()

# Show the plot
plt.show()
