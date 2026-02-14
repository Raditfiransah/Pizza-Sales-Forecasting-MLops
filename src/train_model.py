"""
Model Training Pipeline for Pizza Sales Forecasting
====================================================

Advanced Feature Engineering + MLflow Tracking with:
- Time-Series Lag Features (t-1, t-2, t-3, t-7, t-14, t-30)
- Rolling Window Features (7-day, 14-day, 30-day MA & Std)
- Cyclical Encoding (Sine/Cosine for day_of_week & month)
- Business Context Features (Payday, Weekend, Holiday)
- US Public Holidays (dataset is US-based pizza sales)
- Model Comparison: Random Forest & XGBoost
- MLflow Experiment Tracking

Author: MLOps Pipeline
Date: 2026-02-14
"""

import pandas as pd
import numpy as np
from pathlib import Path
import joblib
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Machine Learning
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error
from sklearn.model_selection import TimeSeriesSplit
import xgboost as xgb

# MLflow
import mlflow
import mlflow.sklearn
import mlflow.xgboost

# Holidays
import holidays

# Visualization
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns


# ============================================================
# FEATURE ENGINEERING
# ============================================================

class AdvancedFeatureEngineer:
    """
    Advanced Feature Engineering for time-series pizza sales forecasting.
    """
    
    def __init__(self):
        self.feature_names = []
        self.us_holidays = holidays.US(years=range(2014, 2017))
    
    def create_lag_features(self, df: pd.DataFrame, target_col: str) -> pd.DataFrame:
        """Create lag features: t-1, t-2, t-3, t-7, t-14, t-30."""
        df = df.copy()
        lags = [1, 2, 3, 7, 14, 30]
        
        for lag in lags:
            col_name = f'{target_col}_lag_{lag}'
            df[col_name] = df[target_col].shift(lag)
            self.feature_names.append(col_name)
        
        for secondary_col in ['num_orders', 'total_quantity']:
            for lag in [1, 7]:
                col_name = f'{secondary_col}_lag_{lag}'
                df[col_name] = df[secondary_col].shift(lag)
                self.feature_names.append(col_name)
        
        return df
    
    def create_rolling_features(self, df: pd.DataFrame, target_col: str) -> pd.DataFrame:
        """Create rolling window statistics (MA, Std, Min, Max)."""
        df = df.copy()
        windows = [7, 14, 30]
        
        for window in windows:
            shifted = df[target_col].shift(1)
            
            col_mean = f'{target_col}_ma_{window}'
            df[col_mean] = shifted.rolling(window=window, min_periods=1).mean()
            self.feature_names.append(col_mean)
            
            col_std = f'{target_col}_std_{window}'
            df[col_std] = shifted.rolling(window=window, min_periods=2).std()
            self.feature_names.append(col_std)
            
            col_min = f'{target_col}_min_{window}'
            df[col_min] = shifted.rolling(window=window, min_periods=1).min()
            self.feature_names.append(col_min)
            
            col_max = f'{target_col}_max_{window}'
            df[col_max] = shifted.rolling(window=window, min_periods=1).max()
            self.feature_names.append(col_max)
        
        for secondary_col in ['num_orders', 'total_quantity']:
            shifted_s = df[secondary_col].shift(1)
            col_name = f'{secondary_col}_ma_7'
            df[col_name] = shifted_s.rolling(window=7, min_periods=1).mean()
            self.feature_names.append(col_name)
        
        return df
    
    def create_cyclical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Cyclical encoding using Sine/Cosine transformation."""
        df = df.copy()
        
        df['dow_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['dow_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        df['woy_sin'] = np.sin(2 * np.pi * df['week_of_year'] / 52)
        df['woy_cos'] = np.cos(2 * np.pi * df['week_of_year'] / 52)
        
        df['day_of_month'] = pd.to_datetime(df['date']).dt.day
        df['dom_sin'] = np.sin(2 * np.pi * df['day_of_month'] / 31)
        df['dom_cos'] = np.cos(2 * np.pi * df['day_of_month'] / 31)
        
        cyclical_features = [
            'dow_sin', 'dow_cos', 'month_sin', 'month_cos',
            'woy_sin', 'woy_cos', 'dom_sin', 'dom_cos'
        ]
        self.feature_names.extend(cyclical_features)
        return df
    
    def create_business_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Business context features: weekend, payday, TGIF."""
        df = df.copy()
        df['day_of_month'] = pd.to_datetime(df['date']).dt.day
        
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        df['is_payday'] = ((df['day_of_month'] >= 25) | (df['day_of_month'] <= 3)).astype(int)
        df['is_month_start'] = (df['day_of_month'] <= 7).astype(int)
        df['is_month_end'] = (df['day_of_month'] >= 25).astype(int)
        df['is_friday'] = (df['day_of_week'] == 4).astype(int)
        df['is_midweek'] = ((df['day_of_week'] >= 1) & (df['day_of_week'] <= 2)).astype(int)
        
        business_features = [
            'is_weekend', 'is_payday', 'is_month_start',
            'is_month_end', 'is_friday', 'is_midweek'
        ]
        self.feature_names.extend(business_features)
        return df
    
    def create_holiday_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """US public holiday features."""
        df = df.copy()
        dates = pd.to_datetime(df['date'])
        
        df['is_holiday'] = dates.apply(lambda x: 1 if x in self.us_holidays else 0)
        df['is_day_before_holiday'] = dates.apply(
            lambda x: 1 if (x + pd.Timedelta(days=1)) in self.us_holidays else 0
        )
        df['is_day_after_holiday'] = dates.apply(
            lambda x: 1 if (x - pd.Timedelta(days=1)) in self.us_holidays else 0
        )
        df['is_holiday_weekend'] = ((df['is_holiday'] == 1) & (df['is_weekend'] == 1)).astype(int)
        
        holiday_features = [
            'is_holiday', 'is_day_before_holiday',
            'is_day_after_holiday', 'is_holiday_weekend'
        ]
        self.feature_names.extend(holiday_features)
        return df
    
    def create_trend_features(self, df: pd.DataFrame, target_col: str) -> pd.DataFrame:
        """Trend and momentum features."""
        df = df.copy()
        
        df['revenue_diff_1'] = df[target_col].shift(1) - df[target_col].shift(2)
        self.feature_names.append('revenue_diff_1')
        
        df['revenue_pct_change'] = df[target_col].shift(1).pct_change()
        self.feature_names.append('revenue_pct_change')
        
        df['revenue_wow_change'] = df[target_col].shift(1) - df[target_col].shift(8)
        self.feature_names.append('revenue_wow_change')
        
        ma7 = df[target_col].shift(1).rolling(window=7, min_periods=1).mean()
        df['revenue_vs_ma7_ratio'] = df[target_col].shift(1) / ma7
        self.feature_names.append('revenue_vs_ma7_ratio')
        
        return df
    
    def fit_transform(self, df: pd.DataFrame, target_col: str = 'total_revenue') -> pd.DataFrame:
        """Apply ALL feature engineering steps."""
        print("\n" + "="*70)
        print("🔧 ADVANCED FEATURE ENGINEERING")
        print("="*70)
        
        self.feature_names = []
        df = df.sort_values('date').reset_index(drop=True)
        
        print("  ✓ Creating lag features (t-1, t-2, t-3, t-7, t-14, t-30)...")
        df = self.create_lag_features(df, target_col)
        
        print("  ✓ Creating rolling statistics (7, 14, 30 day windows)...")
        df = self.create_rolling_features(df, target_col)
        
        print("  ✓ Creating cyclical features (sin/cos encoding)...")
        df = self.create_cyclical_features(df)
        
        print("  ✓ Creating business context features (payday, weekend, TGIF)...")
        df = self.create_business_features(df)
        
        print("  ✓ Creating holiday features (US public holidays)...")
        df = self.create_holiday_features(df)
        
        print("  ✓ Creating trend & momentum features...")
        df = self.create_trend_features(df, target_col)
        
        # Keep original numeric features
        original_features = ['day_of_week', 'month', 'week_of_year',
                             'num_orders', 'total_quantity', 'avg_order_value',
                             'avg_items_per_order', 'day_of_month']
        for feat in original_features:
            if feat not in self.feature_names and feat in df.columns:
                self.feature_names.append(feat)
        
        self.feature_names = list(dict.fromkeys(self.feature_names))
        
        initial_rows = len(df)
        df = df.dropna(subset=self.feature_names)
        dropped = initial_rows - len(df)
        
        print(f"\n  📊 Total features: {len(self.feature_names)}")
        print(f"  📊 Rows: {len(df)} (dropped {dropped} due to lag initialization)")
        
        return df


# ============================================================
# MODEL TRAINER WITH MLFLOW
# ============================================================

class ModelTrainer:
    """Train and compare ML models with MLflow tracking."""
    
    def __init__(self, target_col='total_revenue', experiment_name='pizza_sales_forecasting'):
        self.target_col = target_col
        self.models = {}
        self.results = {}
        self.feature_importance = {}
        self.experiment_name = experiment_name
        
        # Setup MLflow
        mlflow.set_experiment(experiment_name)
        print(f"\n  📋 MLflow experiment: '{experiment_name}'")
    
    def prepare_data(self, df, feature_names):
        """Separate features, target, and dates."""
        X = df[feature_names].copy()
        y = df[self.target_col].copy()
        dates = df['date'].copy()
        return X, y, dates
    
    def _log_metrics(self, mae, rmse, r2, mape, prefix="val"):
        """Log metrics to MLflow."""
        mlflow.log_metric(f"{prefix}_mae", mae)
        mlflow.log_metric(f"{prefix}_rmse", rmse)
        mlflow.log_metric(f"{prefix}_r2", r2)
        mlflow.log_metric(f"{prefix}_mape", mape)
    
    def _log_feature_importance_plot(self, importance_df, model_name, plots_dir):
        """Generate and log feature importance plot."""
        plt.figure(figsize=(10, 8))
        top = importance_df.head(15)
        colors = {'random_forest': '#2ecc71', 'xgboost': '#3498db'}
        
        plt.barh(range(len(top)), top['importance'], color=colors.get(model_name, '#3498db'))
        plt.yticks(range(len(top)), top['feature'], fontsize=9)
        plt.xlabel('Importance', fontsize=12)
        plt.title(f'{model_name.upper()} — Top 15 Features', fontsize=14, fontweight='bold')
        plt.gca().invert_yaxis()
        plt.grid(True, alpha=0.3, axis='x')
        plt.tight_layout()
        
        plot_path = plots_dir / f'{model_name}_feature_importance.png'
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        mlflow.log_artifact(str(plot_path))
        return plot_path
    
    def _log_prediction_plot(self, res, model_name, plots_dir):
        """Generate and log prediction plot."""
        fig, axes = plt.subplots(2, 1, figsize=(14, 10))
        color = {'random_forest': '#2ecc71', 'xgboost': '#3498db'}.get(model_name, '#3498db')
        
        # Actual vs Predicted
        ax1 = axes[0]
        x_axis = res['dates'].values if res['dates'] is not None else range(len(res['actual']))
        ax1.plot(x_axis, res['actual'], label='Actual', color='#2c3e50', linewidth=2, alpha=0.8)
        ax1.plot(x_axis, res['predictions'], label='Predicted', color=color, linewidth=2, alpha=0.7, linestyle='--')
        ax1.set_title(f"{model_name.upper()} — R²={res['r2']:.4f} | RMSE=${res['rmse']:,.0f}",
                      fontsize=13, fontweight='bold')
        ax1.set_ylabel('Revenue ($)')
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)
        
        # Residuals
        ax2 = axes[1]
        residuals = res['actual'] - res['predictions']
        ax2.scatter(x_axis, residuals, alpha=0.5, color='purple', s=30)
        ax2.axhline(y=0, color='red', linestyle='--', linewidth=2)
        ax2.set_ylabel('Residuals ($)')
        ax2.set_title(f'{model_name.upper()} — Residual Plot', fontsize=13, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_path = plots_dir / f'{model_name}_predictions.png'
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        mlflow.log_artifact(str(plot_path))
        return plot_path
    
    def train_random_forest(self, X_train, y_train, X_val, y_val, dates_val, 
                            feature_names, plots_dir):
        """Train Random Forest with MLflow tracking."""
        print("\n" + "="*70)
        print("🌲 TRAINING RANDOM FOREST")
        print("="*70)
        
        params = {
            'n_estimators': 300,
            'max_depth': 12,
            'min_samples_split': 5,
            'min_samples_leaf': 3,
            'max_features': 0.7,
            'random_state': 42,
        }
        
        with mlflow.start_run(run_name="random_forest"):
            # Log params
            mlflow.log_params(params)
            mlflow.log_param("model_type", "RandomForestRegressor")
            mlflow.log_param("num_features", len(feature_names))
            mlflow.log_param("train_samples", len(X_train))
            mlflow.log_param("val_samples", len(X_val))
            
            # Train
            model = RandomForestRegressor(**params, n_jobs=-1)
            model.fit(X_train, y_train)
            
            # Evaluate
            preds = model.predict(X_val)
            mae = mean_absolute_error(y_val, preds)
            rmse = np.sqrt(mean_squared_error(y_val, preds))
            r2 = r2_score(y_val, preds)
            mape = mean_absolute_percentage_error(y_val, preds) * 100
            
            # Log metrics
            self._log_metrics(mae, rmse, r2, mape)
            
            # Store results
            self.results['random_forest'] = {
                'mae': mae, 'rmse': rmse, 'r2': r2, 'mape': mape,
                'predictions': preds, 'actual': y_val.values, 'dates': dates_val
            }
            self.models['random_forest'] = model
            
            # Feature importance
            imp_df = pd.DataFrame({
                'feature': X_train.columns,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            self.feature_importance['random_forest'] = imp_df
            
            # Log plots
            plots_dir.mkdir(exist_ok=True, parents=True)
            self._log_feature_importance_plot(imp_df, 'random_forest', plots_dir)
            self._log_prediction_plot(self.results['random_forest'], 'random_forest', plots_dir)
            
            # Log model
            mlflow.sklearn.log_model(model, "model")
            
            print(f"  ✓ Training complete!")
            print(f"\n  📈 RF:  MAE=${mae:,.2f} | RMSE=${rmse:,.2f} | R²={r2:.4f} | MAPE={mape:.2f}%")
        
        return model
    
    def train_xgboost(self, X_train, y_train, X_val, y_val, dates_val,
                      feature_names, plots_dir):
        """Train XGBoost with MLflow tracking."""
        print("\n" + "="*70)
        print("🚀 TRAINING XGBOOST")
        print("="*70)
        
        params = {
            'n_estimators': 500,
            'max_depth': 6,
            'learning_rate': 0.03,
            'subsample': 0.8,
            'colsample_bytree': 0.7,
            'min_child_weight': 3,
            'gamma': 0.1,
            'reg_alpha': 0.3,
            'reg_lambda': 1.5,
            'random_state': 42,
        }
        
        with mlflow.start_run(run_name="xgboost"):
            # Log params
            mlflow.log_params(params)
            mlflow.log_param("model_type", "XGBRegressor")
            mlflow.log_param("early_stopping_rounds", 30)
            mlflow.log_param("num_features", len(feature_names))
            mlflow.log_param("train_samples", len(X_train))
            mlflow.log_param("val_samples", len(X_val))
            
            # Train
            model = xgb.XGBRegressor(
                **params,
                early_stopping_rounds=30,
                n_jobs=-1,
                verbosity=0
            )
            model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
            
            best_iter = model.best_iteration
            mlflow.log_metric("best_iteration", best_iter)
            print(f"  ✓ Best iteration: {best_iter}")
            
            # Evaluate
            preds = model.predict(X_val)
            mae = mean_absolute_error(y_val, preds)
            rmse = np.sqrt(mean_squared_error(y_val, preds))
            r2 = r2_score(y_val, preds)
            mape = mean_absolute_percentage_error(y_val, preds) * 100
            
            # Log metrics
            self._log_metrics(mae, rmse, r2, mape)
            
            # Store results
            self.results['xgboost'] = {
                'mae': mae, 'rmse': rmse, 'r2': r2, 'mape': mape,
                'predictions': preds, 'actual': y_val.values, 'dates': dates_val
            }
            self.models['xgboost'] = model
            
            # Feature importance
            imp_df = pd.DataFrame({
                'feature': X_train.columns,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            self.feature_importance['xgboost'] = imp_df
            
            # Log plots
            plots_dir.mkdir(exist_ok=True, parents=True)
            self._log_feature_importance_plot(imp_df, 'xgboost', plots_dir)
            self._log_prediction_plot(self.results['xgboost'], 'xgboost', plots_dir)
            
            # Log model
            mlflow.xgboost.log_model(model, "model")
            
            print(f"  ✓ Training complete!")
            print(f"\n  📈 XGB: MAE=${mae:,.2f} | RMSE=${rmse:,.2f} | R²={r2:.4f} | MAPE={mape:.2f}%")
        
        return model
    
    def display_comparison(self):
        """Display final model comparison."""
        print("\n" + "="*70)
        print("🏆 MODEL PERFORMANCE COMPARISON")
        print("="*70)
        
        comparison = {}
        for name, res in self.results.items():
            comparison[name] = {
                'MAE ($)': f"${res['mae']:,.2f}",
                'RMSE ($)': f"${res['rmse']:,.2f}",
                'R²': f"{res['r2']:.4f}",
                'MAPE (%)': f"{res['mape']:.2f}%"
            }
        
        comp_df = pd.DataFrame(comparison).T
        print(f"\n{comp_df.to_string()}")
        
        best_r2 = max(self.results.items(), key=lambda x: x[1]['r2'])
        best_rmse = min(self.results.items(), key=lambda x: x[1]['rmse'])
        
        print(f"\n  🥇 Best R²:   {best_r2[0].upper()} ({best_r2[1]['r2']:.4f})")
        print(f"  🥇 Best RMSE: {best_rmse[0].upper()} (${best_rmse[1]['rmse']:,.2f})")
        print("="*70)
    
    def plot_comparison(self, plots_dir):
        """Generate comparison plots."""
        plots_dir.mkdir(exist_ok=True, parents=True)
        
        # Side-by-side prediction comparison
        fig, axes = plt.subplots(len(self.results), 1, figsize=(16, 5*len(self.results)))
        if len(self.results) == 1:
            axes = [axes]
        
        colors = {'random_forest': '#2ecc71', 'xgboost': '#3498db'}
        
        for idx, (name, res) in enumerate(self.results.items()):
            ax = axes[idx]
            x_axis = res['dates'].values if res['dates'] is not None else range(len(res['actual']))
            ax.plot(x_axis, res['actual'], label='Actual', color='#2c3e50', linewidth=2, alpha=0.8)
            ax.plot(x_axis, res['predictions'], label='Predicted',
                   color=colors.get(name, '#3498db'), linewidth=2, alpha=0.7, linestyle='--')
            ax.set_title(f"{name.upper()} — R²={res['r2']:.4f} | RMSE=${res['rmse']:,.0f}",
                        fontsize=13, fontweight='bold')
            ax.set_ylabel('Revenue ($)')
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        path = plots_dir / 'model_predictions_comparison.png'
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Saved: {path}")


def save_artifacts(trainer, feature_engineer, output_dir):
    """Save models and feature names locally."""
    output_dir.mkdir(exist_ok=True, parents=True)
    
    for name, model in trainer.models.items():
        path = output_dir / f'{name}_model.pkl'
        joblib.dump(model, path)
        print(f"  ✓ Saved model: {path}")
    
    feat_path = output_dir / 'feature_names.json'
    with open(feat_path, 'w') as f:
        json.dump(feature_engineer.feature_names, f, indent=2)
    print(f"  ✓ Saved features: {feat_path}")
    
    results_summary = {}
    for name, res in trainer.results.items():
        results_summary[name] = {
            'mae': float(res['mae']),
            'rmse': float(res['rmse']),
            'r2': float(res['r2']),
            'mape': float(res['mape'])
        }
    
    results_path = output_dir / 'training_results.json'
    with open(results_path, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'target': trainer.target_col,
            'num_features': len(feature_engineer.feature_names),
            'models': results_summary
        }, f, indent=2)
    print(f"  ✓ Saved results: {results_path}")


# ============================================================
# MAIN PIPELINE
# ============================================================

def main():
    print("="*70)
    print("🍕 PIZZA SALES FORECASTING — TRAINING PIPELINE + MLFLOW")
    print("="*70)
    
    # Paths
    base_dir = Path(__file__).parent.parent
    data_dir = base_dir / 'data'
    model_dir = base_dir / 'models'
    plots_dir = base_dir / 'plots'
    
    # Set MLflow tracking URI to local
    mlflow.set_tracking_uri(str(base_dir / 'mlruns'))
    
    # ─── Load Data ───
    print("\n📊 Loading training data...")
    train_df = pd.read_csv(data_dir / 'pizza_sales_train.csv')
    train_df['date'] = pd.to_datetime(train_df['date'])
    print(f"  Samples: {len(train_df)}")
    print(f"  Date range: {train_df['date'].min().date()} → {train_df['date'].max().date()}")
    
    # ─── Feature Engineering ───
    fe = AdvancedFeatureEngineer()
    train_df = fe.fit_transform(train_df, target_col='total_revenue')
    
    # ─── Prepare Data ───
    trainer = ModelTrainer(target_col='total_revenue', experiment_name='pizza_sales_forecasting')
    X, y, dates = trainer.prepare_data(train_df, fe.feature_names)
    
    print(f"\n  Features: {X.shape[1]} | Samples: {X.shape[0]}")
    
    # ─── Train/Val Split (chronological) ───
    split_idx = int(len(X) * 0.8)
    X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_val = y.iloc[:split_idx], y.iloc[split_idx:]
    dates_val = dates.iloc[split_idx:]
    
    print(f"  Train: {len(X_train)} | Val: {len(X_val)}")
    
    # ─── Train Models ───
    
    # 1. Random Forest
    trainer.train_random_forest(X_train, y_train, X_val, y_val, dates_val, 
                                fe.feature_names, plots_dir)
    
    # 2. XGBoost
    trainer.train_xgboost(X_train, y_train, X_val, y_val, dates_val,
                          fe.feature_names, plots_dir)
    
    # ─── Comparison ───
    trainer.display_comparison()
    
    # ─── Comparison Plot ───
    print("\n📊 Generating comparison visualization...")
    trainer.plot_comparison(plots_dir)
    
    # ─── Save Locally ───
    print("\n💾 Saving models and artifacts...")
    save_artifacts(trainer, fe, model_dir)
    
    # ─── Summary ───
    print("\n" + "="*70)
    print("✅ TRAINING COMPLETE!")
    print("="*70)
    print(f"\n  📁 Models: {model_dir}")
    print(f"  📊 Plots: {plots_dir}")
    print(f"  📋 MLflow: {base_dir / 'mlruns'}")
    print(f"\n  💡 View MLflow UI: mlflow ui --backend-store-uri {base_dir / 'mlruns'}")
    print(f"\n  Next: Run model on monitoring data to track real-world performance")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
