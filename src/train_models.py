"""
Model Training and Evaluation Script for Airfare Component Price Estimation.
Trains Linear Regression & Random Forest Regressor, evaluates metrics, and saves artifacts.
"""
import os
import json
import time
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.data_loader import load_raw_dataset, get_route_stats
from src.model_pipeline import (
    CATEGORICAL_FEATURES,
    NUMERICAL_FEATURES,
    FEATURE_COLUMNS,
    create_preprocessor,
    get_feature_names,
    build_models,
    evaluate_predictions,
)


def train_and_evaluate(dataset_path: str = "Clean_Dataset.csv", output_dir: str = "models"):
    """
    Executes full pipeline: data loading, preprocessing, model fitting, metric evaluation,
    feature importance extraction, and artifact serialization.
    """
    os.makedirs(output_dir, exist_ok=True)
    print("=" * 60)
    print("Starting Airfare Component Price Estimation Training Pipeline")
    print("=" * 60)

    # 1. Load Data
    t0 = time.time()
    df = load_raw_dataset(dataset_path)
    print(f"Loaded {len(df):,} rows from {dataset_path} in {time.time() - t0:.2f}s")

    # 2. Prepare Features & Target
    X = df[FEATURE_COLUMNS]
    y = df["price"]

    # 3. Train-Test Split (80/20)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )
    print(f"Train split: {len(X_train):,} samples | Test split: {len(X_test):,} samples")

    # 4. Fit Preprocessor
    print("Fitting ColumnTransformer (StandardScaler + OneHotEncoder)...")
    preprocessor = create_preprocessor()
    t_prep = time.time()
    preprocessor.fit(X_train)
    X_train_trans = preprocessor.transform(X_train)
    X_test_trans = preprocessor.transform(X_test)
    feature_names = get_feature_names(preprocessor)
    print(f"Preprocessor fitted in {time.time() - t_prep:.2f}s. Transformed features: {len(feature_names)}")

    # 5. Build and Fit Models
    models = build_models()
    trained_models = {}
    metrics_summary = {}
    test_preds = {}

    for name, model in models.items():
        print(f"\nTraining {name}...")
        t_start = time.time()
        model.fit(X_train_trans, y_train)
        fit_duration = time.time() - t_start

        # Inference timing
        t_inf_start = time.time()
        y_test_pred = model.predict(X_test_trans)
        inf_duration = time.time() - t_inf_start
        latency_per_sample_ms = (inf_duration / len(X_test)) * 1000.0

        # Evaluate
        y_train_pred = model.predict(X_train_trans)
        test_eval = evaluate_predictions(y_test.values, y_test_pred)
        train_eval = evaluate_predictions(y_train.values, y_train_pred)

        test_preds[name] = y_test_pred

        metrics_summary[name] = {
            "test": test_eval,
            "train": train_eval,
            "training_time_sec": round(fit_duration, 2),
            "inference_time_sec": round(inf_duration, 3),
            "latency_ms_per_sample": round(latency_per_sample_ms, 4),
        }

        print(f"  Finished {name} in {fit_duration:.2f}s")
        print(f"  Test R²: {test_eval['R2']} | Test MAE: ₹{test_eval['MAE']:,} | Test RMSE: ₹{test_eval['RMSE']:,}")

        # Save model file
        clean_model_filename = name.lower().replace(" ", "_") + ".joblib"
        joblib.dump(model, os.path.join(output_dir, clean_model_filename), compress=3)

    # 6. Save Preprocessor
    joblib.dump(preprocessor, os.path.join(output_dir, "preprocessor.joblib"), compress=3)

    # 7. Extract Feature Importances
    print("\nExtracting feature importances & regression coefficients...")
    rf_model = models["Random Forest Regressor"]
    lr_model = models["Linear Regression"]

    rf_importances = [
        {"feature": feat, "importance": round(float(imp), 6)}
        for feat, imp in sorted(
            zip(feature_names, rf_model.feature_importances_),
            key=lambda x: x[1],
            reverse=True,
        )
    ]

    lr_coefficients = [
        {"feature": feat, "coefficient": round(float(coef), 2), "abs_coef": round(abs(float(coef)), 2)}
        for feat, coef in sorted(
            zip(feature_names, lr_model.coef_),
            key=lambda x: abs(x[1]),
            reverse=True,
        )
    ]

    feature_data = {
        "feature_names": feature_names,
        "rf_importances": rf_importances,
        "lr_coefficients": lr_coefficients,
        "lr_intercept": round(float(lr_model.intercept_), 2),
    }

    with open(os.path.join(output_dir, "feature_importance.json"), "w", encoding="utf-8") as f:
        json.dump(feature_data, f, indent=2)

    # 8. Create sample test prediction dataframe for fast, rich interactive visualization
    print("Generating representative test prediction sample (6,000 cases)...")
    sample_indices = np.random.RandomState(42).choice(len(X_test), size=min(6000, len(X_test)), replace=False)
    
    sample_df = X_test.iloc[sample_indices].copy()
    sample_df["actual_price"] = y_test.iloc[sample_indices].values
    sample_df["pred_linear_regression"] = np.round(test_preds["Linear Regression"][sample_indices], 2)
    sample_df["pred_random_forest"] = np.round(test_preds["Random Forest Regressor"][sample_indices], 2)
    
    sample_df["residual_linear"] = np.round(sample_df["pred_linear_regression"] - sample_df["actual_price"], 2)
    sample_df["residual_rf"] = np.round(sample_df["pred_random_forest"] - sample_df["actual_price"], 2)
    sample_df["abs_err_linear"] = np.abs(sample_df["residual_linear"])
    sample_df["abs_err_rf"] = np.abs(sample_df["residual_rf"])
    
    sample_df.to_parquet(os.path.join(output_dir, "test_samples.parquet"), index=False)

    # 9. Save Route Statistics for Fast Lookup
    route_stats_df = get_route_stats(df)
    route_stats_df.to_json(os.path.join(output_dir, "route_stats.json"), orient="records", indent=2)

    # 10. Save Metrics Summary
    with open(os.path.join(output_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics_summary, f, indent=2)

    print("\n" + "=" * 60)
    print("All artifacts successfully trained, evaluated, and saved to:", output_dir)
    print("=" * 60)


if __name__ == "__main__":
    train_and_evaluate()
