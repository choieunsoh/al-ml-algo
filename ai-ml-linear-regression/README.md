# ai-ml-linear-regression Project

## Overview
This project implements a linear regression model to analyze house prices based on square footage. It includes data simulation, exploratory data analysis, and model evaluation.

## Project Structure
- **data/**: Contains datasets used in the project.
  - **raw/**: Original sample dataset.
  - **simulated/**: Simulated dataset for testing and analysis.
- **notebooks/**: Jupyter notebooks for exploratory data analysis.
- **src/**: Source code for the project.
  - **01.linear-regression.py**: Implementation of the linear regression model.
  - **data_simulation.py**: Functions to generate simulated data.
  - **features.py**: Functions for feature engineering.
  - **models.py**: Model definitions and training routines.
- **tests/**: Unit tests for the project.
  - **test_data_simulation.py**: Tests for data simulation functions.
- **requirements.txt**: Lists project dependencies.
- **setup.py**: Packaging information for the project.

## Setup Instructions
1. Clone the repository:
   ```
   git clone <repository-url>
   cd ai-ml-linear-regression
   ```
2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

## Usage
- To run the linear regression model, execute the `01.linear-regression.py` script in the `src` directory.
- Use the Jupyter notebook in the `notebooks` directory for exploratory data analysis and visualizations.

## Contributing
Contributions are welcome! Please open an issue or submit a pull request for any improvements or features.

## License
This project is licensed under the MIT License.