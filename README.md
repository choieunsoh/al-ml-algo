# AI/ML Linear Regression Project

This repository contains code and resources for linear regression analysis, including data simulation, feature engineering, model implementation, and exploratory analysis using Jupyter notebooks.

## Project Structure

```
linear_regression.py                # Standalone script for linear regression
ai-ml-linear-regression/
├── README.md                       # Project documentation
├── requirements.txt                # Python dependencies
├── setup.py                        # Package setup
├── data/
│   ├── raw/
│   │   └── sample_data.csv         # Raw sample data
│   └── simulated/
│       └── simulated_data.csv      # Simulated data for experiments
├── notebooks/
│   └── exploration.ipynb           # Jupyter notebook for data exploration
├── src/
│   ├── data_simulation.py          # Data simulation utilities
│   ├── features.py                 # Feature engineering scripts
│   ├── linear_regression.py        # Linear regression implementation
│   └── models.py                   # Model definitions
└── tests/
    └── test_data_simulation.py     # Unit tests for data simulation
```

## Getting Started

1. **Install dependencies:**
   ```bash
   pip install -r ai-ml-linear-regression/requirements.txt
   ```
2. **Run the main script:**
   ```bash
   python ai-ml-linear-regression/src/linear_regression.py
   ```
3. **Explore the notebook:**
   Open `ai-ml-linear-regression/notebooks/exploration.ipynb` in Jupyter Lab or Notebook.

## Testing

Run unit tests with:

```bash
python -m unittest discover ai-ml-linear-regression/tests
```

## Data

- Place raw data in `ai-ml-linear-regression/data/raw/`.
- Simulated data is generated in `ai-ml-linear-regression/data/simulated/`.

## License

MIT License
