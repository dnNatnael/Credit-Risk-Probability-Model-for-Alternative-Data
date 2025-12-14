"""
Example: Proxy Target Variable Engineering

This example demonstrates how to create a proxy credit risk target variable
using RFM metrics and K-Means clustering.
"""

import sys
import os
import pandas as pd
import numpy as np

# Add parent directory to path to import src modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_processing import (
    calculate_rfm_metrics,
    segment_customers_with_kmeans,
    identify_high_risk_cluster,
    create_proxy_target_variable,
    map_column_names
)


def create_sample_transaction_data(
    n_transactions: int = 1000,
    n_customers: int = 100,
    start_date: str = '2023-01-01',
    end_date: str = '2024-12-31'
) -> pd.DataFrame:
    """
    Create sample transactional data for demonstration.
    
    Parameters:
    -----------
    n_transactions : int
        Number of transactions to generate
    n_customers : int
        Number of unique customers
    start_date : str
        Start date for transactions
    end_date : str
        End date for transactions
        
    Returns:
    --------
    pd.DataFrame
        Sample transactional data
    """
    np.random.seed(42)
    
    # Generate customer IDs
    customer_ids = [f'CUST_{i:04d}' for i in range(1, n_customers + 1)]
    
    # Create transactions
    transactions = []
    for i in range(n_transactions):
        # Random customer
        customer_id = np.random.choice(customer_ids)
        
        # Random transaction date
        date_range = pd.date_range(start=start_date, end=end_date, freq='D')
        transaction_date = np.random.choice(date_range)
        
        # Transaction amount (some customers have higher/lower amounts)
        base_amount = np.random.lognormal(mean=3, sigma=1)
        transaction_amount = max(0.01, base_amount)
        
        transactions.append({
            'CustomerId': customer_id,
            'TransactionDate': transaction_date,
            'TransactionAmount': transaction_amount
        })
    
    df = pd.DataFrame(transactions)
    
    # Sort by date
    df = df.sort_values('TransactionDate').reset_index(drop=True)
    
    return df


def example_1_basic_rfm_calculation():
    """Example 1: Basic RFM metrics calculation"""
    print("=" * 80)
    print("EXAMPLE 1: Basic RFM Metrics Calculation")
    print("=" * 80)
    
    # Create sample data
    df = create_sample_transaction_data(n_transactions=500, n_customers=50)
    print(f"\nSample data shape: {df.shape}")
    print(f"\nFirst few rows:")
    print(df.head())
    
    # Calculate RFM metrics
    rfm_df = calculate_rfm_metrics(
        df=df,
        customer_id_col='CustomerId',
        transaction_date_col='TransactionDate',
        transaction_amount_col='TransactionAmount'
    )
    
    print(f"\nRFM metrics shape: {rfm_df.shape}")
    print(f"\nRFM metrics summary:")
    print(rfm_df.describe())
    print(f"\nFirst few RFM records:")
    print(rfm_df.head(10))


def example_2_customer_segmentation():
    """Example 2: Customer segmentation using K-Means"""
    print("\n" + "=" * 80)
    print("EXAMPLE 2: Customer Segmentation with K-Means")
    print("=" * 80)
    
    # Create sample data
    df = create_sample_transaction_data(n_transactions=1000, n_customers=100)
    
    # Calculate RFM metrics
    rfm_df = calculate_rfm_metrics(
        df=df,
        customer_id_col='CustomerId',
        transaction_date_col='TransactionDate',
        transaction_amount_col='TransactionAmount'
    )
    
    # Segment customers
    rfm_with_clusters, kmeans_model, scaler = segment_customers_with_kmeans(
        rfm_df=rfm_df,
        n_clusters=3,
        random_state=42
    )
    
    print(f"\nClustered RFM data shape: {rfm_with_clusters.shape}")
    print(f"\nCluster distribution:")
    print(rfm_with_clusters['Cluster'].value_counts().sort_index())
    
    # Identify high-risk cluster
    high_risk_cluster_id, cluster_summary = identify_high_risk_cluster(
        rfm_with_clusters=rfm_with_clusters
    )
    
    print(f"\nHigh-risk cluster ID: {high_risk_cluster_id}")
    print(f"\nCluster summary:")
    print(cluster_summary)


def example_3_complete_pipeline():
    """Example 3: Complete pipeline - create proxy target variable"""
    print("\n" + "=" * 80)
    print("EXAMPLE 3: Complete Pipeline - Create Proxy Target Variable")
    print("=" * 80)
    
    # Create sample data
    df = create_sample_transaction_data(n_transactions=1000, n_customers=100)
    print(f"\nOriginal data shape: {df.shape}")
    
    # Create proxy target variable
    df_with_target, rfm_summary, metadata = create_proxy_target_variable(
        df=df,
        customer_id_col='CustomerId',
        transaction_date_col='TransactionDate',
        transaction_amount_col='TransactionAmount',
        n_clusters=3,
        random_state=42
    )
    
    print(f"\nData with target shape: {df_with_target.shape}")
    print(f"\nTarget variable distribution:")
    print(df_with_target['is_high_risk'].value_counts())
    print(f"\nTarget variable percentage:")
    print(df_with_target['is_high_risk'].value_counts(normalize=True) * 100)
    
    print(f"\nMetadata:")
    print(f"  High-risk cluster ID: {metadata['high_risk_cluster_id']}")
    print(f"  High-risk customers: {metadata['target_distribution']['high_risk_count']}")
    print(f"  Low-risk customers: {metadata['target_distribution']['low_risk_count']}")
    print(f"  High-risk percentage: {metadata['target_distribution']['high_risk_pct']:.2f}%")
    
    print(f"\nSample records with target variable:")
    print(df_with_target[['CustomerId', 'TransactionDate', 'TransactionAmount', 'is_high_risk']].head(10))
    
    print(f"\nRFM summary (first 10 customers):")
    print(rfm_summary.head(10))


def example_4_with_actual_data_format():
    """Example 4: Using actual data format (TransactionStartTime, Amount)"""
    print("\n" + "=" * 80)
    print("EXAMPLE 4: Using Actual Data Format")
    print("=" * 80)
    
    # Create sample data matching actual format
    np.random.seed(42)
    n_transactions = 1000
    n_customers = 100
    
    customer_ids = [f'CUST_{i:04d}' for i in range(1, n_customers + 1)]
    transactions = []
    
    for i in range(n_transactions):
        customer_id = np.random.choice(customer_ids)
        date_range = pd.date_range(start='2023-01-01', end='2024-12-31', freq='D')
        transaction_date = np.random.choice(date_range)
        base_amount = np.random.lognormal(mean=3, sigma=1)
        transaction_amount = max(0.01, base_amount)
        
        transactions.append({
            'CustomerId': customer_id,
            'TransactionStartTime': transaction_date,  # Different column name
            'Amount': transaction_amount  # Different column name
        })
    
    df = pd.DataFrame(transactions)
    print(f"\nSample data shape: {df.shape}")
    print(f"\nColumn names: {df.columns.tolist()}")
    print(f"\nFirst few rows:")
    print(df.head())
    
    # Use column mapping or specify explicitly
    df_with_target, rfm_summary, metadata = create_proxy_target_variable(
        df=df,
        customer_id_col='CustomerId',
        transaction_date_col='TransactionStartTime',  # Specify explicitly
        transaction_amount_col='Amount',  # Specify explicitly
        n_clusters=3,
        random_state=42
    )
    
    print(f"\nData with target shape: {df_with_target.shape}")
    print(f"\nTarget variable distribution:")
    print(df_with_target['is_high_risk'].value_counts())
    
    print(f"\nHigh-risk percentage: {metadata['target_distribution']['high_risk_pct']:.2f}%")
    
    print(f"\nSample records with target variable:")
    print(df_with_target[['CustomerId', 'TransactionStartTime', 'Amount', 'is_high_risk']].head(10))


if __name__ == '__main__':
    # Run all examples
    example_1_basic_rfm_calculation()
    example_2_customer_segmentation()
    example_3_complete_pipeline()
    example_4_with_actual_data_format()
    
    print("\n" + "=" * 80)
    print("All examples completed successfully!")
    print("=" * 80)

