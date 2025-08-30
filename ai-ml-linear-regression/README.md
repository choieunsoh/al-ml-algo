# ai-ml-linear-regression Project

## Overview

This project implements a linear regression model to analyze house prices based on square footage. It includes data simulation, exploratory data analysis, feature engineering, model training, and evaluation.

## Project Structure

- **data/**
  - **raw/**: Original sample dataset (`sample_data.csv`).
  - **simulated/**: Simulated dataset for testing and analysis (`simulated_data.csv`).
- **notebooks/**: Jupyter notebooks for exploratory data analysis (`exploration.ipynb`).
- **src/**: Source code for the project.
  - **01.linear-regression.py**: Script to run the linear regression model.
  - **data_simulation.py**: Functions to generate simulated data (`generate_simulated_data`).
  - **features.py**: Functions for feature engineering (`create_features`).
  - **models.py**: Model definitions and training routines (`LinearRegressionModel`).
  - **linear_regression.py**: Training and evaluation helpers (`train_linear_regression`).
- **tests/**: Unit tests for the project.
  - **test_data_simulation.py**: Tests for data simulation functions.
- **requirements.txt**: Lists project dependencies.
- **setup.py**: Packaging information for the project.

## Setup Instructions

1. Clone the repository:
   ```sh
   git clone <repository-url>
   cd ai-ml-linear-regression
   ```
2. (Recommended) Create and activate a virtual environment:
   ```sh
   python -m venv .venv
   .venv\Scripts\activate
   ```
3. Install the required packages:
   ```sh
   pip install -r requirements.txt
   ```

## Usage

- **Run the linear regression model:**
  ```sh
  python src/linear_regression.py
  ```
- **Generate simulated data:**
  Use the `generate_simulated_data` function in `src/data_simulation.py`.

- **Feature engineering:**
  Use the `create_features` function in `src/features.py`.

- **Exploratory Data Analysis:**
  Open and run `notebooks/exploration.ipynb` in Jupyter Notebook.

## Running Tests

To run unit tests:

```sh
pip install pytest
pytest tests/
```

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or features. Ensure your changes include relevant unit tests.

## License

This project is licensed
