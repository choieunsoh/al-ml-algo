import numpy as np
import pandas as pd

def simulate_data(num_samples=100, square_footage_range=(1000, 5000), price_range=(150000, 700000)):
    np.random.seed(42)  # For reproducibility
    square_footage = np.random.randint(square_footage_range[0], square_footage_range[1], num_samples)
    price = (square_footage * np.random.uniform(100, 200, num_samples)).astype(int) + np.random.randint(price_range[0], price_range[1], num_samples)
    
    simulated_data = pd.DataFrame({
        'SquareFootage': square_footage,
        'Price': price
    })
    
    return simulated_data

def test_simulate_data():
    simulated_data = simulate_data(num_samples=10)
    
    assert len(simulated_data) == 10, "Number of samples should be 10"
    assert 'SquareFootage' in simulated_data.columns, "Column 'SquareFootage' should be present"
    assert 'Price' in simulated_data.columns, "Column 'Price' should be present"
    assert simulated_data['SquareFootage'].min() >= 1000, "Minimum square footage should be at least 1000"
    assert simulated_data['SquareFootage'].max() <= 5000, "Maximum square footage should be at most 5000"
    assert simulated_data['Price'].min() >= 150000, "Minimum price should be at least 150000"
    assert simulated_data['Price'].max() <= 700000, "Maximum price should be at most 700000"

if __name__ == "__main__":
    test_simulate_data()