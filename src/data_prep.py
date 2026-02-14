"""
Data Preparation Script for Pizza Sales Forecasting
=====================================================

This script transforms transaction-level pizza sales data into daily aggregated format
suitable for time series forecasting models.

Author: Data Engineering Pipeline
Date: 2026-02-14
"""

import pandas as pd
import numpy as np
from pathlib import Path


def load_transaction_data(filepath: str) -> pd.DataFrame:
    """
    Load transaction-level pizza sales data from CSV file.
    
    Parameters:
    -----------
    filepath : str
        Path to the pizza sales CSV file
        
    Returns:
    --------
    pd.DataFrame
        DataFrame containing transaction-level data
    """
    print(f"Loading data from {filepath}...")
    df = pd.read_csv(filepath)
    
    # Convert order_date to datetime
    # The CSV has mixed formats, so use 'mixed' format with dayfirst=True
    df['order_date'] = pd.to_datetime(df['order_date'], format='mixed', dayfirst=True)
    
    print(f"Loaded {len(df):,} transactions")
    print(f"Date range: {df['order_date'].min()} to {df['order_date'].max()}")
    
    return df


def aggregate_daily_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate transaction data to daily level.
    
    This function groups transaction data by date and calculates:
    - Total daily revenue (sum of total_price)
    - Number of orders per day (count of unique order_id)
    - Total quantity of pizzas sold per day (sum of quantity)
    - Number of transactions per day (count of rows)
    
    Parameters:
    -----------
    df : pd.DataFrame
        Transaction-level DataFrame
        
    Returns:
    --------
    pd.DataFrame
        Daily aggregated DataFrame with metrics
    """
    print("\nAggregating data to daily level...")
    
    # Group by date and calculate aggregations
    daily_df = df.groupby('order_date').agg({
        'total_price': 'sum',           # Total daily revenue
        'order_id': 'nunique',          # Number of unique orders
        'quantity': 'sum',              # Total pizzas sold
        'pizza_id': 'count'             # Number of transactions
    }).reset_index()
    
    # Rename columns for clarity
    daily_df.columns = [
        'date',
        'total_revenue',
        'num_orders',
        'total_quantity',
        'num_transactions'
    ]
    
    # Add additional features
    daily_df['avg_order_value'] = daily_df['total_revenue'] / daily_df['num_orders']
    daily_df['avg_items_per_order'] = daily_df['total_quantity'] / daily_df['num_orders']
    
    # Add day of week (0 = Monday, 6 = Sunday)
    daily_df['day_of_week'] = daily_df['date'].dt.dayofweek
    daily_df['day_name'] = daily_df['date'].dt.day_name()
    
    # Add month and year for seasonality analysis
    daily_df['month'] = daily_df['date'].dt.month
    daily_df['year'] = daily_df['date'].dt.year
    daily_df['week_of_year'] = daily_df['date'].dt.isocalendar().week
    
    print(f"Created {len(daily_df)} daily records")
    
    return daily_df


def get_category_sales(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate daily sales by pizza category.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Transaction-level DataFrame
        
    Returns:
    --------
    pd.DataFrame
        Daily sales by category
    """
    print("\nCalculating category-level aggregations...")
    
    category_daily = df.groupby(['order_date', 'pizza_category']).agg({
        'total_price': 'sum',
        'quantity': 'sum'
    }).reset_index()
    
    category_daily.columns = ['date', 'category', 'revenue', 'quantity']
    
    # Pivot to wide format
    category_pivot = category_daily.pivot_table(
        index='date',
        columns='category',
        values=['revenue', 'quantity'],
        fill_value=0
    )
    
    # Flatten column names
    category_pivot.columns = [f'{col[1]}_{col[0]}' for col in category_pivot.columns]
    category_pivot = category_pivot.reset_index()
    
    return category_pivot


def get_size_distribution(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate daily distribution of pizza sizes.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Transaction-level DataFrame
        
    Returns:
    --------
    pd.DataFrame
        Daily pizza size distribution
    """
    print("\nCalculating size distribution...")
    
    size_daily = df.groupby(['order_date', 'pizza_size']).agg({
        'quantity': 'sum'
    }).reset_index()
    
    size_pivot = size_daily.pivot_table(
        index='order_date',
        columns='pizza_size',
        values='quantity',
        fill_value=0
    ).reset_index()
    
    size_pivot.columns = ['date'] + [f'size_{col}_qty' for col in size_pivot.columns[1:]]
    
    return size_pivot


def display_summary_statistics(daily_df: pd.DataFrame):
    """
    Display summary statistics of the aggregated data.
    
    Parameters:
    -----------
    daily_df : pd.DataFrame
        Daily aggregated DataFrame
    """
    print("\n" + "="*60)
    print("SUMMARY STATISTICS - DAILY AGGREGATED DATA")
    print("="*60)
    
    print(f"\nTotal Days: {len(daily_df)}")
    print(f"\nRevenue Statistics:")
    print(f"  Total Revenue: ${daily_df['total_revenue'].sum():,.2f}")
    print(f"  Average Daily Revenue: ${daily_df['total_revenue'].mean():,.2f}")
    print(f"  Min Daily Revenue: ${daily_df['total_revenue'].min():,.2f}")
    print(f"  Max Daily Revenue: ${daily_df['total_revenue'].max():,.2f}")
    
    print(f"\nOrder Statistics:")
    print(f"  Total Orders: {daily_df['num_orders'].sum():,}")
    print(f"  Average Daily Orders: {daily_df['num_orders'].mean():.2f}")
    print(f"  Min Daily Orders: {daily_df['num_orders'].min()}")
    print(f"  Max Daily Orders: {daily_df['num_orders'].max()}")
    
    print(f"\nQuantity Statistics:")
    print(f"  Total Pizzas Sold: {daily_df['total_quantity'].sum():,}")
    print(f"  Average Daily Quantity: {daily_df['total_quantity'].mean():.2f}")
    
    print(f"\nAverage Order Value: ${daily_df['avg_order_value'].mean():.2f}")
    print(f"Average Items per Order: {daily_df['avg_items_per_order'].mean():.2f}")
    
    # Day of week analysis
    print(f"\n\nRevenue by Day of Week:")
    dow_revenue = daily_df.groupby('day_name')['total_revenue'].mean().sort_values(ascending=False)
    for day, revenue in dow_revenue.items():
        print(f"  {day}: ${revenue:,.2f}")
    
    print("\n" + "="*60)


def main():
    """
    Main execution function for data preparation pipeline.
    """
    # Define file paths
    base_dir = Path(__file__).parent.parent
    input_file = base_dir / 'data' / 'pizza_sales.csv'
    output_dir = base_dir / 'data'
    
    # Create output directory if it doesn't exist
    output_dir.mkdir(exist_ok=True)
    
    # Step 1: Load transaction data
    df_transactions = load_transaction_data(input_file)
    
    # Step 2: Aggregate to daily level
    df_daily = aggregate_daily_data(df_transactions)
    
    # Step 3: Get category-level aggregations
    df_category = get_category_sales(df_transactions)
    
    # Step 4: Get size distribution
    df_size = get_size_distribution(df_transactions)
    
    # Step 5: Merge all features
    df_final = df_daily.merge(df_category, on='date', how='left')
    df_final = df_final.merge(df_size, on='date', how='left')
    
    # Fill any NaN values with 0
    df_final = df_final.fillna(0)
    
    # Step 6: Display summary statistics
    display_summary_statistics(df_daily)
    
    # Step 7: Save aggregated data
    output_file = output_dir / 'pizza_sales_daily.csv'
    df_final.to_csv(output_file, index=False)
    print(f"\n✓ Daily aggregated data saved to: {output_file}")
    print(f"  Shape: {df_final.shape[0]} rows × {df_final.shape[1]} columns")
    
    # Also save a simple version with just core metrics
    core_output_file = output_dir / 'pizza_sales_daily_core.csv'
    df_daily.to_csv(core_output_file, index=False)
    print(f"\n✓ Core daily metrics saved to: {core_output_file}")
    print(f"  Shape: {df_daily.shape[0]} rows × {df_daily.shape[1]} columns")
    
    # Preview the data
    print("\n" + "="*60)
    print("PREVIEW OF DAILY AGGREGATED DATA")
    print("="*60)
    print(df_daily.head(5).to_string(index=False))
    

if __name__ == "__main__":
    main()