from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, train_test_split


# Function to simulate data
def simulate_data(num_samples=100, square_footage_range=(1000, 5000), price_range=(100000, 700000)):
    np.random.seed(42)  # For reproducibility
    square_footage = np.random.randint(square_footage_range[0], square_footage_range[1], num_samples)
    price = (square_footage * 100) + np.random.randint(price_range[0], price_range[1], num_samples)
    return pd.DataFrame({'SquareFootage': square_footage, 'Price': price})

# Generate simulated data
simulated_df = simulate_data()

# Ensure output directory exists and save the simulated data to CSV
output_dir = Path(__file__).resolve().parents[1] / 'data' / 'simulated'
output_dir.mkdir(parents=True, exist_ok=True)
csv_path = output_dir / 'simulated_data.csv'

# Save the simulated data to a CSV file
simulated_df.to_csv(csv_path, index=False)

# Load the simulated data for analysis
df = pd.read_csv(csv_path)

# Display the first few rows of the simulated data
print(df.head())

# Features (X) and Target (y)
X = df[['SquareFootage']]  # Feature(s)
y = df['Price']            # Target variable

# Split data into 80% training and 20% testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the Linear Regression model
model = LinearRegression()

# Train the model on the training data
model.fit(X_train, y_train)

# Make predictions on the testing set
y_pred = model.predict(X_test)

# Calculate evaluation metrics
mse = mean_squared_error(y_test, y_pred)
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

# Plot the data points and the regression line
plt.scatter(X_test, y_test, color='blue', label='Actual Data')
plt.plot(X_test, y_pred, color='red', label='Regression Line')
plt.xlabel('Square Footage')
plt.ylabel('Price')
plt.title('House Prices vs. Square Footage')
plt.legend()
plt.show()
