"""
Time-Series Data Splitter for Pizza Sales Forecasting
======================================================

This script splits the daily aggregated data into training and monitoring sets
using a time-series approach:
- 80% earliest data → Training set
- 20% latest data → Monitoring set

This ensures that the model is evaluated on "future" data it hasn't seen during training.

Author: MLOps Pipeline
Date: 2026-02-14
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime


def load_daily_data(filepath: str) -> pd.DataFrame:
    """
    Load daily aggregated pizza sales data.
    
    Parameters:
    -----------
    filepath : str
        Path to the daily CSV file
        
    Returns:
    --------
    pd.DataFrame
        DataFrame sorted by date
    """
    print(f"Loading daily data from {filepath}...")
    df = pd.read_csv(filepath)
    
    # Convert date to datetime
    df['date'] = pd.to_datetime(df['date'])
    
    # Sort by date to ensure chronological order
    df = df.sort_values('date').reset_index(drop=True)
    
    print(f"Loaded {len(df)} days of data")
    print(f"Date range: {df['date'].min().date()} to {df['date'].max().date()}")
    
    return df


def time_series_split(df: pd.DataFrame, train_ratio: float = 0.8) -> tuple:
    """
    Split data chronologically for time series forecasting and monitoring.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame sorted by date
    train_ratio : float
        Ratio of data to use for training (default: 0.8 for 80%)
        
    Returns:
    --------
    tuple
        (train_df, monitoring_df)
    """
    print(f"\nSplitting data with {train_ratio*100}% for training...")
    
    # Calculate split index
    split_idx = int(len(df) * train_ratio)
    
    # Split data chronologically
    train_df = df.iloc[:split_idx].copy()
    monitoring_df = df.iloc[split_idx:].copy()
    
    print(f"\nTraining set:")
    print(f"  - Records: {len(train_df)}")
    print(f"  - Date range: {train_df['date'].min().date()} to {train_df['date'].max().date()}")
    print(f"  - Days: {(train_df['date'].max() - train_df['date'].min()).days + 1}")
    
    print(f"\nMonitoring set:")
    print(f"  - Records: {len(monitoring_df)}")
    print(f"  - Date range: {monitoring_df['date'].min().date()} to {monitoring_df['date'].max().date()}")
    print(f"  - Days: {(monitoring_df['date'].max() - monitoring_df['date'].min()).days + 1}")
    
    # Verify no data leakage
    assert train_df['date'].max() < monitoring_df['date'].min(), "Data leakage detected!"
    print("\n✓ Verified: No data leakage (training data is strictly before monitoring data)")
    
    return train_df, monitoring_df


def display_split_statistics(train_df: pd.DataFrame, monitoring_df: pd.DataFrame):
    """
    Display statistics comparing train and monitoring sets.
    
    Parameters:
    -----------
    train_df : pd.DataFrame
        Training dataset
    monitoring_df : pd.DataFrame
        Monitoring dataset
    """
    print("\n" + "="*70)
    print("DATASET COMPARISON - TRAINING vs MONITORING")
    print("="*70)
    
    metrics = ['total_revenue', 'num_orders', 'total_quantity']
    
    for metric in metrics:
        print(f"\n{metric.upper().replace('_', ' ')}:")
        print(f"  Training   - Mean: {train_df[metric].mean():,.2f} | Std: {train_df[metric].std():,.2f}")
        print(f"  Monitoring - Mean: {monitoring_df[metric].mean():,.2f} | Std: {monitoring_df[metric].std():,.2f}")
        
        # Check for distribution shift
        train_mean = train_df[metric].mean()
        monitoring_mean = monitoring_df[metric].mean()
        pct_diff = ((monitoring_mean - train_mean) / train_mean) * 100
        
        if abs(pct_diff) > 10:
            print(f"  ⚠️  WARNING: {abs(pct_diff):.1f}% difference detected - possible distribution shift!")
        else:
            print(f"  ✓ Difference: {pct_diff:+.1f}% (acceptable)")
    
    print("\n" + "="*70)


def save_split_data(train_df: pd.DataFrame, monitoring_df: pd.DataFrame, output_dir: Path):
    """
    Save train and monitoring sets to separate CSV files.
    
    Parameters:
    -----------
    train_df : pd.DataFrame
        Training dataset
    monitoring_df : pd.DataFrame
        Monitoring dataset
    output_dir : Path
        Directory to save the files
    """
    # Create output directory if it doesn't exist
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Save training set
    train_path = output_dir / 'pizza_sales_train.csv'
    train_df.to_csv(train_path, index=False)
    print(f"\n✓ Training data saved to: {train_path}")
    print(f"  Shape: {train_df.shape[0]} rows × {train_df.shape[1]} columns")
    
    # Save monitoring set
    monitoring_path = output_dir / 'pizza_sales_monitoring.csv'
    monitoring_df.to_csv(monitoring_path, index=False)
    print(f"\n✓ Monitoring data saved to: {monitoring_path}")
    print(f"  Shape: {monitoring_df.shape[0]} rows × {monitoring_df.shape[1]} columns")
    
    # Save split info
    info = {
        'split_date': datetime.now().isoformat(),
        'train_records': len(train_df),
        'monitoring_records': len(monitoring_df),
        'train_start': train_df['date'].min().isoformat(),
        'train_end': train_df['date'].max().isoformat(),
        'monitoring_start': monitoring_df['date'].min().isoformat(),
        'monitoring_end': monitoring_df['date'].max().isoformat(),
        'train_ratio': len(train_df) / (len(train_df) + len(monitoring_df))
    }
    
    info_path = output_dir / 'split_info.txt'
    with open(info_path, 'w') as f:
        f.write("TIME-SERIES DATA SPLIT INFORMATION\n")
        f.write("="*50 + "\n\n")
        for key, value in info.items():
            f.write(f"{key}: {value}\n")
    
    print(f"\n✓ Split information saved to: {info_path}")


def create_visualization_preview(train_df: pd.DataFrame, monitoring_df: pd.DataFrame):
    """
    Display a simple text-based visualization of the split.
    
    Parameters:
    -----------
    train_df : pd.DataFrame
        Training dataset
    monitoring_df : pd.DataFrame
        Monitoring dataset
    """
    print("\n" + "="*70)
    print("TIME-SERIES SPLIT VISUALIZATION")
    print("="*70)
    
    total_days = len(train_df) + len(monitoring_df)
    train_width = int((len(train_df) / total_days) * 50)
    monitoring_width = 50 - train_width
    
    print("\nTimeline:")
    print("  " + "─" * 66)
    print("  │" + "█" * train_width + "░" * monitoring_width + "│")
    print("  " + "─" * 66)
    print("  " + train_df['date'].min().strftime('%Y-%m-%d') + 
          " " * 40 + monitoring_df['date'].max().strftime('%Y-%m-%d'))
    print()
    print("  █ Training Data (80%)")
    print("  ░ Monitoring Data (20%)")
    print("\n" + "="*70)


def main():
    """
    Main execution function for time-series data splitting.
    """
    # Define file paths
    base_dir = Path(__file__).parent.parent
    data_dir = base_dir / 'data'
    input_file = data_dir / 'pizza_sales_daily_core.csv'
    
    print("="*70)
    print("TIME-SERIES DATA SPLITTER")
    print("Pizza Sales Forecasting MLOps Pipeline")
    print("="*70)
    
    # Step 1: Load daily data
    df = load_daily_data(input_file)
    
    # Step 2: Perform time-series split
    train_df, monitoring_df = time_series_split(df, train_ratio=0.8)
    
    # Step 3: Display statistics
    display_split_statistics(train_df, monitoring_df)
    
    # Step 4: Create visualization
    create_visualization_preview(train_df, monitoring_df)
    
    # Step 5: Save split datasets
    save_split_data(train_df, monitoring_df, data_dir)
    
    print("\n" + "="*70)
    print("DATA SPLIT COMPLETE!")
    print("="*70)
    print("\nNext steps:")
    print("  1. Use pizza_sales_train.csv to train your forecasting model")
    print("  2. Use pizza_sales_monitoring.csv to monitor model performance over time")
    print("  3. Ensure you don't use monitoring data during training (no data leakage!)")
    print("\nReminder:")
    print("  - Training data represents the 'past' the model learns from")
    print("  - Monitoring data represents the 'future' used for performance tracking")
    print("  - Never shuffle time-series data!")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
