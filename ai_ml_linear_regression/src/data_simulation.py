import numpy as np
import pandas as pd

def generate_simulated_data(num_samples=100, sqft_mean=2500, sqft_std=500, price_per_sqft=150):
    """
    Generate a simulated dataset of house prices based on square footage.
    
    Parameters:
    - num_samples: Number of samples to generate.
    - sqft_mean: Mean square footage of the houses.
    - sqft_std: Standard deviation of the square footage.
    - price_per_sqft: Price per square foot.
    
    Returns:
    - DataFrame containing simulated square footage and prices.
    """
    # Generate random square footage data
    square_footage = np.random.normal(loc=sqft_mean, scale=sqft_std, size=num_samples).astype(int)
    
    # Ensure square footage is positive
    square_footage = np.clip(square_footage, a_min=500, a_max=None)
    
    # Calculate prices based on square footage
    prices = square_footage * price_per_sqft
    
    # Create a DataFrame
    simulated_data = pd.DataFrame({
        'SquareFootage': square_footage,
        'Price': prices
    })
    
    return simulated_data

def save_simulated_data(file_path='data/simulated/simulated_data.csv', num_samples=100):
    """
    Generate and save simulated data to a CSV file.
    
    Parameters:
    - file_path: Path to save the simulated data.
    - num_samples: Number of samples to generate.
    """
    simulated_data = generate_simulated_data(num_samples)
    simulated_data.to_csv(file_path, index=False)
    print(f"Simulated data saved to {file_path}")

# Example usage
if __name__ == "__main__":
    save_simulated_data()