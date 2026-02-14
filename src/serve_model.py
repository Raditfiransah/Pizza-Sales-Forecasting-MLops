"""
Pizza Sales Forecasting API
============================

FastAPI service for serving the trained XGBoost model.
Accepts a target date and returns estimated revenue prediction.

Usage:
    uvicorn src.serve_model:app --reload --port 8000

Endpoints:
    GET  /              → Health check & model info
    POST /predict       → Predict revenue for a specific date
    POST /predict/batch → Predict revenue for multiple dates
    GET  /model/info    → Model performance metrics
    GET  /history       → Recent historical data

Author: MLOps Pipeline
Date: 2026-02-14
"""

import json
import joblib
import numpy as np
import pandas as pd
import holidays
from pathlib import Path
from datetime import date, datetime, timedelta
from typing import Optional, List
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


# ============================================================
# GLOBAL STATE
# ============================================================

BASE_DIR = Path(__file__).parent.parent
MODEL_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data"

# These will be loaded on startup
model = None
feature_names = None
historical_df = None
us_holidays = None
training_results = None


# ============================================================
# PYDANTIC SCHEMAS
# ============================================================

class PredictionRequest(BaseModel):
    """Request body for a single prediction."""
    target_date: date = Field(..., description="Target date for prediction (YYYY-MM-DD)")
    
    model_config = {"json_schema_extra": {"examples": [{"target_date": "2015-11-01"}]}}

class BatchPredictionRequest(BaseModel):
    """Request body for batch predictions."""
    start_date: date = Field(..., description="Start date (YYYY-MM-DD)")
    end_date: date = Field(..., description="End date (YYYY-MM-DD)")
    
    model_config = {"json_schema_extra": {"examples": [{"start_date": "2015-11-01", "end_date": "2015-11-07"}]}}

class PredictionResponse(BaseModel):
    """Response for a single prediction."""
    date: str
    day_name: str
    predicted_revenue: float
    is_weekend: bool
    is_holiday: bool
    holiday_name: Optional[str] = None
    confidence_note: str

class BatchPredictionResponse(BaseModel):
    """Response for batch predictions."""
    predictions: List[PredictionResponse]
    total_predicted_revenue: float
    avg_daily_revenue: float
    num_days: int

class ModelInfoResponse(BaseModel):
    """Model information and metrics."""
    model_type: str
    num_features: int
    training_timestamp: str
    metrics: dict
    feature_names: List[str]

class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    model_loaded: bool
    historical_data_rows: int
    api_version: str


# ============================================================
# FEATURE ENGINEERING (mirrors train_model.py logic)
# ============================================================

def compute_features_for_date(target_date: date, history: pd.DataFrame) -> pd.Series:
    """
    Compute all 54 features for a single target date.
    
    Uses historical data to calculate lag and rolling features.
    This mirrors the AdvancedFeatureEngineer from train_model.py.
    """
    target_dt = pd.Timestamp(target_date)
    
    # Get recent history sorted by date
    hist = history[history['date'] < target_dt].sort_values('date').copy()
    
    if len(hist) < 30:
        raise ValueError(f"Not enough historical data before {target_date}. "
                        f"Need at least 30 days, got {len(hist)}")
    
    features = {}
    
    # --- Lag Features ---
    for lag in [1, 2, 3, 7, 14, 30]:
        row = hist.iloc[-lag] if lag <= len(hist) else None
        features[f'total_revenue_lag_{lag}'] = row['total_revenue'] if row is not None else 0
    
    for col in ['num_orders', 'total_quantity']:
        for lag in [1, 7]:
            row = hist.iloc[-lag] if lag <= len(hist) else None
            features[f'{col}_lag_{lag}'] = row[col] if row is not None else 0
    
    # --- Rolling Features ---
    recent = hist.tail(30)
    revenue_series = recent['total_revenue']
    
    for window in [7, 14, 30]:
        window_data = revenue_series.tail(window)
        features[f'total_revenue_ma_{window}'] = window_data.mean()
        features[f'total_revenue_std_{window}'] = window_data.std() if len(window_data) >= 2 else 0
        features[f'total_revenue_min_{window}'] = window_data.min()
        features[f'total_revenue_max_{window}'] = window_data.max()
    
    for col in ['num_orders', 'total_quantity']:
        features[f'{col}_ma_7'] = recent[col].tail(7).mean()
    
    # --- Cyclical Features ---
    dow = target_dt.dayofweek  # Monday=0
    month = target_dt.month
    woy = target_dt.isocalendar()[1]
    dom = target_dt.day
    
    features['dow_sin'] = np.sin(2 * np.pi * dow / 7)
    features['dow_cos'] = np.cos(2 * np.pi * dow / 7)
    features['month_sin'] = np.sin(2 * np.pi * month / 12)
    features['month_cos'] = np.cos(2 * np.pi * month / 12)
    features['woy_sin'] = np.sin(2 * np.pi * woy / 52)
    features['woy_cos'] = np.cos(2 * np.pi * woy / 52)
    features['dom_sin'] = np.sin(2 * np.pi * dom / 31)
    features['dom_cos'] = np.cos(2 * np.pi * dom / 31)
    
    # --- Business Features ---
    features['is_weekend'] = 1 if dow >= 5 else 0
    features['is_payday'] = 1 if (dom >= 25 or dom <= 3) else 0
    features['is_month_start'] = 1 if dom <= 7 else 0
    features['is_month_end'] = 1 if dom >= 25 else 0
    features['is_friday'] = 1 if dow == 4 else 0
    features['is_midweek'] = 1 if dow in [1, 2] else 0
    
    # --- Holiday Features ---
    features['is_holiday'] = 1 if target_dt in us_holidays else 0
    features['is_day_before_holiday'] = 1 if (target_dt + timedelta(days=1)) in us_holidays else 0
    features['is_day_after_holiday'] = 1 if (target_dt - timedelta(days=1)) in us_holidays else 0
    features['is_holiday_weekend'] = 1 if (features['is_holiday'] and features['is_weekend']) else 0
    
    # --- Trend Features ---
    if len(hist) >= 2:
        features['revenue_diff_1'] = hist.iloc[-1]['total_revenue'] - hist.iloc[-2]['total_revenue']
    else:
        features['revenue_diff_1'] = 0
    
    if len(hist) >= 2 and hist.iloc[-2]['total_revenue'] != 0:
        features['revenue_pct_change'] = (
            (hist.iloc[-1]['total_revenue'] - hist.iloc[-2]['total_revenue']) 
            / hist.iloc[-2]['total_revenue']
        )
    else:
        features['revenue_pct_change'] = 0
    
    if len(hist) >= 8:
        features['revenue_wow_change'] = hist.iloc[-1]['total_revenue'] - hist.iloc[-8]['total_revenue']
    else:
        features['revenue_wow_change'] = 0
    
    ma7_val = revenue_series.tail(7).mean()
    features['revenue_vs_ma7_ratio'] = (
        hist.iloc[-1]['total_revenue'] / ma7_val if ma7_val != 0 else 1
    )
    
    # --- Original Features (use averages from history for the same day-of-week) ---
    same_dow = hist[hist['date'].dt.dayofweek == dow]
    if len(same_dow) > 0:
        features['num_orders'] = same_dow['num_orders'].mean()
        features['total_quantity'] = same_dow['total_quantity'].mean()
        features['avg_order_value'] = same_dow['avg_order_value'].mean()
        features['avg_items_per_order'] = same_dow['avg_items_per_order'].mean()
    else:
        features['num_orders'] = hist['num_orders'].mean()
        features['total_quantity'] = hist['total_quantity'].mean()
        features['avg_order_value'] = hist['avg_order_value'].mean()
        features['avg_items_per_order'] = hist['avg_items_per_order'].mean()
    
    features['day_of_week'] = dow
    features['month'] = month
    features['week_of_year'] = woy
    features['day_of_month'] = dom
    
    return pd.Series(features)


def predict_single_date(target_date: date) -> PredictionResponse:
    """Generate prediction for a single date."""
    target_dt = pd.Timestamp(target_date)
    
    # Compute features
    features = compute_features_for_date(target_date, historical_df)
    
    # Ensure correct feature order
    feature_vector = features[feature_names].values.reshape(1, -1)
    feature_df = pd.DataFrame(feature_vector, columns=feature_names)
    
    # Predict
    predicted_revenue = float(model.predict(feature_df)[0])
    
    # Get holiday info
    holiday_name = us_holidays.get(target_dt)
    is_holiday = target_dt in us_holidays
    
    # Confidence note based on how much historical data we have
    days_from_last = (target_dt - historical_df['date'].max()).days
    if days_from_last <= 7:
        confidence = "High — prediction is close to available historical data"
    elif days_from_last <= 30:
        confidence = "Medium — prediction is within a month of historical data"
    else:
        confidence = "Low — prediction is far from available historical data, use with caution"
    
    return PredictionResponse(
        date=str(target_date),
        day_name=target_dt.strftime('%A'),
        predicted_revenue=round(predicted_revenue, 2),
        is_weekend=target_dt.dayofweek >= 5,
        is_holiday=is_holiday,
        holiday_name=holiday_name,
        confidence_note=confidence
    )


# ============================================================
# FASTAPI APP
# ============================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model and data on startup."""
    global model, feature_names, historical_df, us_holidays, training_results
    
    print("🔄 Loading model and data...")
    
    # Load the best model (XGBoost)
    model_path = MODEL_DIR / "xgboost_model.pkl"
    if not model_path.exists():
        raise RuntimeError(f"Model not found at {model_path}. Run train_model.py first!")
    model = joblib.load(model_path)
    print(f"  ✓ Model loaded: {model_path}")
    
    # Load feature names
    feat_path = MODEL_DIR / "feature_names.json"
    with open(feat_path) as f:
        feature_names = json.load(f)
    print(f"  ✓ Features loaded: {len(feature_names)} features")
    
    # Load historical data (training + monitoring for full history)
    train_path = DATA_DIR / "pizza_sales_train.csv"
    monitoring_path = DATA_DIR / "pizza_sales_monitoring.csv"
    
    dfs = [pd.read_csv(train_path)]
    if monitoring_path.exists():
        dfs.append(pd.read_csv(monitoring_path))
    
    historical_df = pd.concat(dfs, ignore_index=True)
    historical_df['date'] = pd.to_datetime(historical_df['date'])
    historical_df = historical_df.sort_values('date').reset_index(drop=True)
    print(f"  ✓ Historical data: {len(historical_df)} days "
          f"({historical_df['date'].min().date()} → {historical_df['date'].max().date()})")
    
    # Load holidays
    us_holidays = holidays.US(years=range(2014, 2028))
    print(f"  ✓ US holidays loaded")
    
    # Load training results
    results_path = MODEL_DIR / "training_results.json"
    if results_path.exists():
        with open(results_path) as f:
            training_results = json.load(f)
    
    print("✅ API ready to serve predictions!\n")
    
    yield  # App runs here
    
    print("👋 Shutting down API...")


app = FastAPI(
    title="🍕 Pizza Sales Forecasting API",
    description=(
        "Predict daily pizza sales revenue using a trained XGBoost model.\n\n"
        "The model uses 54 engineered features including lag values, rolling statistics, "
        "cyclical encoding, business context, and US holidays to forecast revenue."
    ),
    version="1.0.0",
    lifespan=lifespan
)


# ─── Endpoints ───

@app.get("/", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Health check — verify the API and model are running."""
    return HealthResponse(
        status="healthy",
        model_loaded=model is not None,
        historical_data_rows=len(historical_df) if historical_df is not None else 0,
        api_version="1.0.0"
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Predictions"])
async def predict(request: PredictionRequest):
    """
    Predict revenue for a specific date.
    
    Send a date and get back the estimated daily revenue.
    The model uses historical data to compute lag and trend features automatically.
    
    **Example:**
    ```json
    {"date": "2015-11-01"}
    ```
    """
    try:
        result = predict_single_date(request.target_date)
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["Predictions"])
async def predict_batch(request: BatchPredictionRequest):
    """
    Predict revenue for a date range.
    
    Send a start and end date to get predictions for every day in that range.
    
    **Example:**
    ```json
    {"start_date": "2015-11-01", "end_date": "2015-11-07"}
    ```
    """
    if request.end_date < request.start_date:
        raise HTTPException(status_code=400, detail="end_date must be after start_date")
    
    num_days = (request.end_date - request.start_date).days + 1
    if num_days > 90:
        raise HTTPException(status_code=400, detail="Maximum 90 days per batch request")
    
    predictions = []
    current = request.start_date
    
    while current <= request.end_date:
        try:
            pred = predict_single_date(current)
            predictions.append(pred)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=f"Failed for {current}: {str(e)}")
        current += timedelta(days=1)
    
    total = sum(p.predicted_revenue for p in predictions)
    
    return BatchPredictionResponse(
        predictions=predictions,
        total_predicted_revenue=round(total, 2),
        avg_daily_revenue=round(total / len(predictions), 2),
        num_days=len(predictions)
    )


@app.get("/model/info", response_model=ModelInfoResponse, tags=["Model"])
async def model_info():
    """Get information about the current model, including performance metrics."""
    if training_results is None:
        raise HTTPException(status_code=404, detail="Training results not found")
    
    xgb_metrics = training_results.get("models", {}).get("xgboost", {})
    
    return ModelInfoResponse(
        model_type="XGBRegressor",
        num_features=training_results.get("num_features", len(feature_names)),
        training_timestamp=training_results.get("timestamp", "unknown"),
        metrics={
            "MAE": f"${xgb_metrics.get('mae', 0):,.2f}",
            "RMSE": f"${xgb_metrics.get('rmse', 0):,.2f}",
            "R²": f"{xgb_metrics.get('r2', 0):.4f}",
            "MAPE": f"{xgb_metrics.get('mape', 0):.2f}%"
        },
        feature_names=feature_names
    )


@app.get("/history", tags=["Data"])
async def get_history(last_n: int = 7):
    """Get the most recent N days of historical data."""
    if last_n > 60:
        raise HTTPException(status_code=400, detail="Maximum 60 days")
    
    recent = historical_df.tail(last_n)
    records = recent[['date', 'total_revenue', 'num_orders', 'total_quantity']].copy()
    records['date'] = records['date'].dt.strftime('%Y-%m-%d')
    
    return {
        "last_n_days": last_n,
        "data": records.to_dict(orient='records')
    }


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.serve_model:app", host="0.0.0.0", port=8000, reload=True)
