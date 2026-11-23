"""
Interactive Airfare Estimation and Component Decomposition Engine.
"""
from typing import Dict, Any, List, Optional
import os
import json
import joblib
import numpy as np
import pandas as pd


class FareEstimator:
    """
    Handles live airfare inference, component price decomposition, and booking insights.
    """

    def __init__(self, models_dir: str = "models"):
        self.models_dir = models_dir
        self.lr_model = None
        self.rf_model = None
        self.preprocessor = None
        self.metrics = None
        self.feature_data = None
        self.route_stats = None
        self._load_artifacts()

    def _load_artifacts(self):
        lr_path = os.path.join(self.models_dir, "linear_regression.joblib")
        rf_path = os.path.join(self.models_dir, "random_forest_regressor.joblib")
        if not os.path.exists(rf_path):
            rf_path = os.path.join(self.models_dir, "random_forest.joblib")
        prep_path = os.path.join(self.models_dir, "preprocessor.joblib")

        if os.path.exists(lr_path) and os.path.exists(rf_path) and os.path.exists(prep_path):
            self.lr_model = joblib.load(lr_path)
            self.rf_model = joblib.load(rf_path)
            self.preprocessor = joblib.load(prep_path)

        metrics_path = os.path.join(self.models_dir, "metrics.json")
        if os.path.exists(metrics_path):
            with open(metrics_path, "r", encoding="utf-8") as f:
                self.metrics = json.load(f)

        feat_path = os.path.join(self.models_dir, "feature_importance.json")
        if os.path.exists(feat_path):
            with open(feat_path, "r", encoding="utf-8") as f:
                self.feature_data = json.load(f)

        route_path = os.path.join(self.models_dir, "route_stats.json")
        if os.path.exists(route_path):
            with open(route_path, "r", encoding="utf-8") as f:
                self.route_stats = json.load(f)

    def is_ready(self) -> bool:
        return self.rf_model is not None and self.lr_model is not None and self.preprocessor is not None

    def predict_fare(
        self,
        airline: str,
        source_city: str,
        departure_time: str,
        stops: str,
        arrival_time: str,
        destination_city: str,
        flight_class: str,
        duration: float,
        days_left: int,
    ) -> Dict[str, Any]:
        """
        Computes airfare estimates for both models and produces pricing component breakdown.
        """
        input_dict = {
            "airline": airline,
            "source_city": source_city,
            "departure_time": departure_time,
            "stops": stops,
            "arrival_time": arrival_time,
            "destination_city": destination_city,
            "class": flight_class,
            "duration": float(duration),
            "days_left": int(days_left),
        }
        df_input = pd.DataFrame([input_dict])
        input_trans = self.preprocessor.transform(df_input)

        pred_lr = float(self.lr_model.predict(input_trans)[0])
        pred_rf = float(self.rf_model.predict(input_trans)[0])

        # Floor at minimum logical fare
        pred_lr_clean = max(1100.0, pred_lr)
        pred_rf_clean = max(1100.0, pred_rf)

        # Baseline comparison (same flight booked 35 days in advance)
        baseline_dict = dict(input_dict)
        baseline_dict["days_left"] = 35
        base_trans = self.preprocessor.transform(pd.DataFrame([baseline_dict]))
        base_rf = float(self.rf_model.predict(base_trans)[0])

        # Urgency surcharge
        urgency_surcharge = max(0.0, pred_rf_clean - base_rf) if days_left < 20 else 0.0

        # Class premium (compare Economy vs Business on same parameters)
        econ_dict = dict(input_dict)
        econ_dict["class"] = "Economy"
        econ_trans = self.preprocessor.transform(pd.DataFrame([econ_dict]))
        econ_fare = float(self.rf_model.predict(econ_trans)[0])
        
        biz_dict = dict(input_dict)
        biz_dict["class"] = "Business"
        biz_trans = self.preprocessor.transform(pd.DataFrame([biz_dict]))
        biz_fare = float(self.rf_model.predict(biz_trans)[0])
        class_premium = max(0.0, biz_fare - econ_fare) if flight_class == "Business" else 0.0

        # Model discrepancy
        diff = pred_rf_clean - pred_lr_clean
        diff_pct = (diff / pred_lr_clean) * 100.0 if pred_lr_clean > 0 else 0.0

        # Confidence bounds using RF model test RMSE
        rf_rmse = self.metrics.get("Random Forest Regressor", {}).get("test", {}).get("RMSE", 2860.0) if self.metrics else 2860.0
        conf_lower = max(1100.0, pred_rf_clean - rf_rmse)
        conf_upper = pred_rf_clean + rf_rmse

        # Recommendation advice
        if days_left <= 3:
            advice = "🚨 Extreme Surge Window: Booking 1-3 days prior incurs maximum dynamic pricing markup. Book immediately if non-negotiable."
            urgency_level = "High Risk / Premium"
        elif days_left <= 10:
            advice = "⚠️ Last-Minute Escalation: Fares are climbing sharply. Delaying further will likely increase ticket cost."
            urgency_level = "Elevated"
        elif days_left <= 25:
            advice = "✅ Standard Market Fare: You are in the moderate booking window. Fares are reasonably stable."
            urgency_level = "Moderate"
        else:
            advice = "🎉 Optimal Booking Window: Fares are at or near minimum baseline advance pricing. Excellent time to lock in."
            urgency_level = "Low / Value Deal"

        return {
            "inputs": input_dict,
            "pred_rf": round(pred_rf_clean, 2),
            "pred_lr": round(pred_lr_clean, 2),
            "conf_lower": round(conf_lower, 2),
            "conf_upper": round(conf_upper, 2),
            "model_diff": round(diff, 2),
            "model_diff_pct": round(diff_pct, 2),
            "base_fare_advance": round(base_rf, 2),
            "urgency_surcharge": round(urgency_surcharge, 2),
            "class_premium": round(class_premium, 2),
            "economy_equivalent": round(econ_fare, 2),
            "business_equivalent": round(biz_fare, 2),
            "advice": advice,
            "urgency_level": urgency_level,
        }

    def simulate_days_left_curve(
        self,
        airline: str,
        source_city: str,
        departure_time: str,
        stops: str,
        arrival_time: str,
        destination_city: str,
        flight_class: str,
        duration: float,
    ) -> pd.DataFrame:
        """
        Simulates how fare changes for days_left from 1 to 49 days.
        """
        days_range = list(range(1, 50))
        records = []
        for d in days_range:
            records.append({
                "airline": airline,
                "source_city": source_city,
                "departure_time": departure_time,
                "stops": stops,
                "arrival_time": arrival_time,
                "destination_city": destination_city,
                "class": flight_class,
                "duration": float(duration),
                "days_left": int(d),
            })
        df_sim = pd.DataFrame(records)
        sim_trans = self.preprocessor.transform(df_sim)
        sim_rf = np.maximum(1100.0, self.rf_model.predict(sim_trans))
        sim_lr = np.maximum(1100.0, self.lr_model.predict(sim_trans))

        result = pd.DataFrame({
            "days_left": days_range,
            "Random Forest (Recommended)": np.round(sim_rf, 2),
            "Linear Regression (Baseline)": np.round(sim_lr, 2),
        })
        return result

    def simulate_airline_comparison(
        self,
        source_city: str,
        departure_time: str,
        stops: str,
        arrival_time: str,
        destination_city: str,
        flight_class: str,
        duration: float,
        days_left: int,
        all_airlines: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Compares fares across all airlines for the identical itinerary.
        """
        if all_airlines is None:
            all_airlines = ["AirAsia", "GO_FIRST", "Indigo", "SpiceJet", "Air_India", "Vistara"]

        records = []
        for air in all_airlines:
            records.append({
                "airline": air,
                "source_city": source_city,
                "departure_time": departure_time,
                "stops": stops,
                "arrival_time": arrival_time,
                "destination_city": destination_city,
                "class": flight_class,
                "duration": float(duration),
                "days_left": int(days_left),
            })
        df_sim = pd.DataFrame(records)
        sim_trans = self.preprocessor.transform(df_sim)
        rf_fares = np.maximum(1100.0, self.rf_model.predict(sim_trans))
        lr_fares = np.maximum(1100.0, self.lr_model.predict(sim_trans))

        res_df = pd.DataFrame({
            "Airline": all_airlines,
            "RF Estimated Fare (₹)": np.round(rf_fares, 2),
            "LR Estimated Fare (₹)": np.round(lr_fares, 2),
        }).sort_values(by="RF Estimated Fare (₹)")

        min_fare = res_df["RF Estimated Fare (₹)"].min()
        res_df["Difference from Lowest (₹)"] = res_df["RF Estimated Fare (₹)"] - min_fare
        return res_df
