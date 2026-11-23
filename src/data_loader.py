"""
Data loading and preprocessing utilities for Airfare Component Price Estimation.
"""
from typing import Dict, Any, Tuple, List
import pandas as pd
import numpy as np


def load_raw_dataset(csv_path: str = "Clean_Dataset.csv") -> pd.DataFrame:
    """
    Loads and standardizes the flight airfare dataset.
    """
    df = pd.read_csv(csv_path)
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])
    
    # Ensure stops format is clean and consistent
    df["stops"] = df["stops"].astype(str).str.strip().str.lower()
    
    # Create route identifier column
    df["route"] = df["source_city"] + " → " + df["destination_city"]
    
    return df


def get_dataset_overview(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Computes top-level summary metrics for executive dashboard.
    """
    return {
        "total_records": len(df),
        "num_airlines": df["airline"].nunique(),
        "num_source_cities": df["source_city"].nunique(),
        "num_dest_cities": df["destination_city"].nunique(),
        "num_routes": df["route"].nunique(),
        "min_fare": float(df["price"].min()),
        "max_fare": float(df["price"].max()),
        "mean_fare": float(df["price"].mean()),
        "median_fare": float(df["price"].median()),
        "economy_mean_fare": float(df[df["class"] == "Economy"]["price"].mean()),
        "business_mean_fare": float(df[df["class"] == "Business"]["price"].mean()),
        "avg_duration": float(df["duration"].mean()),
        "avg_days_left": float(df["days_left"].mean()),
    }


def get_unique_options(df: pd.DataFrame) -> Dict[str, List[str]]:
    """
    Returns unique values sorted appropriately for UI selectors.
    """
    airlines = sorted(df["airline"].unique().tolist())
    source_cities = sorted(df["source_city"].unique().tolist())
    destination_cities = sorted(df["destination_city"].unique().tolist())
    departure_times = ["Early_Morning", "Morning", "Afternoon", "Evening", "Night", "Late_Night"]
    arrival_times = ["Early_Morning", "Morning", "Afternoon", "Evening", "Night", "Late_Night"]
    stops = ["zero", "one", "two_or_more"]
    classes = ["Economy", "Business"]

    return {
        "airlines": airlines,
        "source_cities": source_cities,
        "destination_cities": destination_cities,
        "departure_times": departure_times,
        "arrival_times": arrival_times,
        "stops": stops,
        "classes": classes,
    }


def get_route_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregates route level statistics: avg price, min price, max price, avg duration, count.
    """
    route_agg = df.groupby(["source_city", "destination_city", "class"]).agg(
        avg_price=("price", "mean"),
        min_price=("price", "min"),
        max_price=("price", "max"),
        avg_duration=("duration", "mean"),
        flight_count=("price", "count")
    ).reset_index()
    
    route_agg["avg_price"] = route_agg["avg_price"].round(2)
    route_agg["avg_duration"] = route_agg["avg_duration"].round(2)
    return route_agg


def get_typical_duration(df: pd.DataFrame, source: str, destination: str, stops: str = "zero") -> float:
    """
    Estimates typical flight duration for given source, destination and stops.
    """
    subset = df[(df["source_city"] == source) & (df["destination_city"] == destination)]
    if stops in subset["stops"].values:
        sub_stops = subset[subset["stops"] == stops]
        if not sub_stops.empty:
            return round(float(sub_stops["duration"].median()), 2)
    if not subset.empty:
        return round(float(subset["duration"].median()), 2)
    return 2.50
