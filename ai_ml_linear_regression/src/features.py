import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, PolynomialFeatures

def scale_features(X):
    scaler = StandardScaler()
    return scaler.fit_transform(X)

def generate_polynomial_features(X, degree=2):
    poly = PolynomialFeatures(degree)
    return poly.fit_transform(X)

def create_feature_dataframe(X, additional_features=None):
    feature_df = pd.DataFrame(X, columns=['SquareFootage'])
    if additional_features is not None:
        for key, value in additional_features.items():
            feature_df[key] = value
    return feature_df