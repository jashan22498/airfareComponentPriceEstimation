"""
Model Pipeline Definition, Preprocessing, and Evaluation for Airfare Price Estimation.
"""
from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    mean_absolute_percentage_error,
    explained_variance_score,
    max_error
)

CATEGORICAL_FEATURES = [
    "airline",
    "source_city",
    "departure_time",
    "stops",
    "arrival_time",
    "destination_city",
    "class",
]

NUMERICAL_FEATURES = [
    "duration",
    "days_left",
]

FEATURE_COLUMNS = CATEGORICAL_FEATURES + NUMERICAL_FEATURES


def create_preprocessor() -> ColumnTransformer:
    """
    Constructs the ColumnTransformer pipeline for categorical encoding and numerical scaling.
    """
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERICAL_FEATURES),
            (
                "cat",
                OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False),
                CATEGORICAL_FEATURES,
            ),
        ],
        remainder="drop",
    )
    return preprocessor


def get_feature_names(preprocessor: ColumnTransformer) -> List[str]:
    """
    Extracts human-readable feature names from fitted ColumnTransformer.
    """
    feature_names = list(NUMERICAL_FEATURES)
    cat_encoder = preprocessor.named_transformers_["cat"]
    encoded_cat_names = cat_encoder.get_feature_names_out(CATEGORICAL_FEATURES)
    feature_names.extend(list(encoded_cat_names))
    return feature_names


def build_models() -> Dict[str, Any]:
    """
    Instantiates both Linear Regression and Random Forest Regressor.
    """
    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest Regressor": RandomForestRegressor(
            n_estimators=45,
            max_depth=20,
            min_samples_split=4,
            min_samples_leaf=2,
            n_jobs=-1,
            random_state=42,
        ),
    }
    return models


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Calculates comprehensive regression performance metrics.
    """
    mae = float(mean_absolute_error(y_true, y_pred))
    mse = float(mean_squared_error(y_true, y_pred))
    rmse = float(np.sqrt(mse))
    r2 = float(r2_score(y_true, y_pred))
    mape = float(mean_absolute_percentage_error(y_true, y_pred)) * 100.0
    evs = float(explained_variance_score(y_true, y_pred))
    max_err = float(max_error(y_true, y_pred))

    return {
        "MAE": round(mae, 2),
        "MSE": round(mse, 2),
        "RMSE": round(rmse, 2),
        "R2": round(r2, 4),
        "MAPE (%)": round(mape, 2),
        "Explained Variance": round(evs, 4),
        "Max Error": round(max_err, 2),
    }
