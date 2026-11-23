# ✈️ AeroPrice AI: Airfare Component Price Estimation

### Smart India Hackathon (SIH 2026) Problem Bank — SIH26056
**Topic S.No. 17** | **Course:** Supervised Learning (INT395)  
**Environment:** Python 3.15.2 | Scikit-Learn | Plotly | Streamlit

---

## 📌 Project Overview & Deliverables Scope

In modern commercial aviation, airline ticket fares fluctuate dynamically according to revenue management algorithms. These yield optimization models account for lead time, cabin class, route distances, departure times, and carrier tier. This project delivers an end-to-end, production-grade **Supervised Regression Machine Learning Pipeline** and an **Interactive Full-Stack Web Dashboard** to predict airfares and decompose pricing components.

### Expected Deliverables Completed:
- ✅ **Regression Pipeline**: Automated feature extraction, scaling, and transformation.
- ✅ **Categorical Encoding**: `OneHotEncoder(drop='first')` for high-dimensional categorical features.
- ✅ **Model Comparison**: Empirical benchmark between **Linear Regression** and **Random Forest Regressor**.
- ✅ **Metrics Evaluation**: Test & Train metrics for $R^2$, MAE, RMSE, MSE, MAPE, and Max Error.
- ✅ **Feature Importance Analysis**: MDI Gini importance and standardized regression coefficients.
- ✅ **Interactive Fare Estimator**: Consumer pricing calculator with real-time what-if scenario simulator.

---

## 📊 Dataset Description (`Clean_Dataset.csv`)

The dataset contains **300,153 verified flight records** across 6 major Indian Tier-1 metropolitan airports:
- **Airlines (6)**: Vistara, Air India, Indigo, GO FIRST, AirAsia, SpiceJet
- **Cities (6)**: Delhi, Mumbai, Bangalore, Kolkata, Hyderabad, Chennai
- **Cabin Classes (2)**: Economy, Business
- **Booking Window (`days_left`)**: 1 to 49 days before flight departure
- **Stops (3)**: Zero (Non-stop), One-stop, Two or more stops
- **Time Windows**: Early Morning, Morning, Afternoon, Evening, Night, Late Night
- **Target Variable**: Ticket `price` in Indian Rupees (₹), ranging from ₹1,105 to ₹1,23,071.

---

## 🤖 Model Benchmark & Evaluation Results

Evaluated on a held-out test partition of **60,031 samples (20% split)**:

| Metric / Attribute | Linear Regression (Baseline) | Random Forest Regressor (Ensemble) | Performance Advantage |
| :--- | :---: | :---: | :---: |
| **$R^2$ Score** | **0.9113** | **0.9844** | 🏆 **Random Forest (+7.31% variance)** |
| **Mean Absolute Error (MAE)** | **₹4,553.29** | **₹1,298.68** | 🏆 **Random Forest (71.5% error reduction)** |
| **Root Mean Squared Error (RMSE)** | **₹6,761.71** | **₹2,834.40** | 🏆 **Random Forest (58.1% error reduction)** |
| **Mean Absolute % Error (MAPE)** | **46.24%** | **9.09%** | 🏆 **Random Forest (<10% error margin)** |
| **Explained Variance** | **0.9113** | **0.9844** | 🏆 **Random Forest** |
| **Training Duration** | **0.22 s** | **6.35 s** | ⚡ Linear Regression (Sub-second fit) |
| **Inference Latency** | **<0.001 ms** | **~0.002 ms** | ⚡ Both models ultra-low latency |

### 💡 Core Engineering Takeaways:
1. **Non-Linear Surge Pricing**: Fares spike non-linearly inside the 15-day window. Linear Regression forces a straight-line slope, under-predicting last-minute flights and over-predicting advance flights. Random Forest captures non-linear decision splits cleanly.
2. **Feature Leverage**: Cabin Class (`class_Economy`) is the single most dominant factor (~88.8% feature importance), followed by flight `duration` (~5.6%) and `days_left` (~1.5%).

---

## 📁 Repository Structure

```text
Airfare Component Price Estimation/
├── Clean_Dataset.csv                  # 300,153 flight records
├── app.py                             # Full-Stack Streamlit Dashboard
├── requirements.txt                   # Environment package versions
├── README.md                          # Comprehensive documentation
├── src/
│   ├── __init__.py
│   ├── data_loader.py                 # Dataset loading, cleaning, route metrics
│   ├── model_pipeline.py              # Scikit-learn Pipeline & evaluation functions
│   ├── train_models.py                # Standalone training & artifact generator
│   └── fare_calculator.py             # Live inference, decomposition & simulation engine
└── models/                            # Serialized artifacts for instant loading
    ├── linear_regression.joblib       # Fitted Linear Regression model
    ├── random_forest.joblib           # Fitted Random Forest Regressor
    ├── preprocessor.joblib            # Fitted ColumnTransformer (OHE + StandardScaler)
    ├── metrics.json                   # Detailed metric scorecards
    ├── feature_importance.json        # Top features & coefficients
    ├── route_stats.json               # Aggregated city-pair statistics
    └── test_samples.parquet           # Sample test predictions for interactive plots
```

---

## 🚀 How to Run the Dashboard

### 1. Launch the Web Dashboard
Run the Streamlit application using Python 3.15.2:
```bash
python -m streamlit run app.py
```
Open your browser at `http://localhost:8501`.

### 2. (Optional) Re-train and Update Models
To re-run the training pipeline from scratch and re-generate all artifacts:
```bash
python -m src.train_models
```

---

## 🖥️ Dashboard Features

1. **Executive Overview**: Project background, SIH problem mapping, high-level KPIs, and architecture workflow.
2. **Exploratory Data Analysis (EDA)**: Interactive distribution plots, airline boxplots, class violin charts, route heatmaps, and departure window analysis.
3. **Model Benchmark**: Side-by-side metric tables, actual vs. predicted scatter plots with 45° reference lines, residual histograms, and homoscedasticity diagnostics.
4. **Feature Importance**: Bar charts of Random Forest MDI importance and Linear Regression coefficients with domain explainability.
5. **Interactive Fare Estimator**: Consumer-friendly search form with instant live predictions, confidence intervals, price build-up waterfall, and smart booking recommendations.
6. **Dynamic What-If Simulator**:
   - **Days-to-Departure Curve**: Visualizes price progression across 1–49 days for the exact itinerary.
   - **Carrier Rate Matrix**: Side-by-side fare quotes for all 6 airlines for the identical flight.
