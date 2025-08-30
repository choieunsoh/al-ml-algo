import numpy as np
import pandas as pd

def generate_simulated_data(num_samples=100, square_footage_range=(1000, 5000), price_range=(150000, 700000)):
    np.random.seed(42)  # For reproducibility
    square_footage = np.random.randint(square_footage_range[0], square_footage_range[1], num_samples)
    price = (square_footage * np.random.uniform(100, 150)) + np.random.normal(0, 20000, num_samples)  # Adding some noise
    return pd.DataFrame({'SquareFootage': square_footage, 'Price': price})

def save_simulated_data(file_path='data/simulated/simulated_data.csv', num_samples=100):
    simulated_data = generate_simulated_data(num_samples)
    simulated_data.to_csv(file_path, index=False)

if __name__ == "__main__":
    save_simulated_data()