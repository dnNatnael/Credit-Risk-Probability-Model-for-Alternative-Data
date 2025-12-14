"""
Proxy Target Variable Engineering for Credit Risk Modeling

This module creates a proxy credit risk target variable (is_high_risk) using
customer engagement patterns from transactional data. It implements:

1. RFM (Recency, Frequency, Monetary) metrics calculation
2. K-Means clustering for customer segmentation
3. High-risk label identification based on least engaged customers
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from typing import Optional, Tuple
import warnings
warnings.filterwarnings('ignore')


def calculate_rfm_metrics(
    df: pd.DataFrame,
    customer_id_col: str = 'CustomerId',
    transaction_date_col: str = 'TransactionDate',
    transaction_amount_col: str = 'TransactionAmount',
    snapshot_date: Optional[pd.Timestamp] = None
) -> pd.DataFrame:
    """
    Calculate RFM (Recency, Frequency, Monetary) metrics for each customer.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input DataFrame with transactional data containing:
        - customer_id_col: Customer identifier
        - transaction_date_col: Transaction date/timestamp
        - transaction_amount_col: Transaction amount
    customer_id_col : str
        Column name for customer ID (default: 'CustomerId')
    transaction_date_col : str
        Column name for transaction date (default: 'TransactionDate')
    transaction_amount_col : str
        Column name for transaction amount (default: 'TransactionAmount')
    snapshot_date : pd.Timestamp, optional
        Reference date for recency calculation. If None, uses max transaction date + 1 day
        
    Returns:
    --------
    pd.DataFrame
        Customer-level DataFrame with columns:
        - customer_id_col: Customer identifier
        - Recency: Number of days since most recent transaction
        - Frequency: Total number of transactions
        - Monetary: Total transaction amount
    """
    # Make a copy to avoid modifying original
    df_work = df.copy()
    
    # Ensure transaction date is datetime
    if not pd.api.types.is_datetime64_any_dtype(df_work[transaction_date_col]):
        df_work[transaction_date_col] = pd.to_datetime(
            df_work[transaction_date_col], 
            errors='coerce'
        )
    
    # Ensure transaction amount is numeric
    if not pd.api.types.is_numeric_dtype(df_work[transaction_amount_col]):
        df_work[transaction_amount_col] = pd.to_numeric(
            df_work[transaction_amount_col], 
            errors='coerce'
        )
    
    # Define snapshot date (reference point for recency)
    if snapshot_date is None:
        snapshot_date = df_work[transaction_date_col].max() + pd.Timedelta(days=1)
    
    # Calculate RFM metrics per customer
    rfm_df = df_work.groupby(customer_id_col).agg({
        transaction_date_col: [
            ('Recency', lambda x: (snapshot_date - x.max()).days)
        ],
        transaction_amount_col: [
            ('Frequency', 'count'),
            ('Monetary', 'sum')
        ]
    }).reset_index()
    
    # Flatten column names
    rfm_df.columns = [customer_id_col, 'Recency', 'Frequency', 'Monetary']
    
    # Handle edge cases
    # Recency should be non-negative
    rfm_df['Recency'] = rfm_df['Recency'].clip(lower=0)
    # Frequency and Monetary should be non-negative
    rfm_df['Frequency'] = rfm_df['Frequency'].clip(lower=0)
    rfm_df['Monetary'] = rfm_df['Monetary'].clip(lower=0)
    
    return rfm_df


def segment_customers_with_kmeans(
    rfm_df: pd.DataFrame,
    n_clusters: int = 3,
    random_state: int = 42,
    customer_id_col: str = 'CustomerId'
) -> Tuple[pd.DataFrame, KMeans, StandardScaler]:
    """
    Segment customers using K-Means clustering on RFM features.
    
    Parameters:
    -----------
    rfm_df : pd.DataFrame
        DataFrame with RFM metrics (Recency, Frequency, Monetary)
    n_clusters : int
        Number of clusters for K-Means (default: 3)
    random_state : int
        Random state for reproducibility (default: 42)
    customer_id_col : str
        Column name for customer ID (default: 'CustomerId')
        
    Returns:
    --------
    tuple
        (rfm_with_clusters, kmeans_model, scaler)
        - rfm_with_clusters: RFM DataFrame with added 'Cluster' column
        - kmeans_model: Fitted KMeans model
        - scaler: Fitted StandardScaler used for preprocessing
    """
    # Extract RFM features
    rfm_features = ['Recency', 'Frequency', 'Monetary']
    X = rfm_df[rfm_features].values
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Apply K-Means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    clusters = kmeans.fit_predict(X_scaled)
    
    # Add cluster labels to DataFrame
    rfm_with_clusters = rfm_df.copy()
    rfm_with_clusters['Cluster'] = clusters
    
    return rfm_with_clusters, kmeans, scaler


def identify_high_risk_cluster(
    rfm_with_clusters: pd.DataFrame,
    customer_id_col: str = 'CustomerId'
) -> Tuple[int, pd.DataFrame]:
    """
    Identify the least engaged (high-risk) cluster based on RFM centroids.
    
    The high-risk cluster is characterized by:
    - High Recency (long time since last transaction)
    - Low Frequency
    - Low Monetary value
    
    Parameters:
    -----------
    rfm_with_clusters : pd.DataFrame
        DataFrame with RFM metrics and Cluster labels
    customer_id_col : str
        Column name for customer ID (default: 'CustomerId')
        
    Returns:
    --------
    tuple
        (high_risk_cluster_id, cluster_summary)
        - high_risk_cluster_id: Cluster ID identified as high-risk
        - cluster_summary: DataFrame with cluster statistics
    """
    # Calculate cluster centroids (mean RFM values per cluster)
    cluster_summary = rfm_with_clusters.groupby('Cluster').agg({
        'Recency': 'mean',
        'Frequency': 'mean',
        'Monetary': 'mean',
        customer_id_col: 'count'
    }).rename(columns={customer_id_col: 'CustomerCount'})
    
    # Calculate a risk score for each cluster
    # Higher recency = higher risk, Lower frequency = higher risk, Lower monetary = higher risk
    # Normalize each metric to 0-1 scale for comparison
    cluster_summary['Recency_norm'] = (
        (cluster_summary['Recency'] - cluster_summary['Recency'].min()) /
        (cluster_summary['Recency'].max() - cluster_summary['Recency'].min() + 1e-10)
    )
    cluster_summary['Frequency_norm'] = (
        1 - (cluster_summary['Frequency'] - cluster_summary['Frequency'].min()) /
        (cluster_summary['Frequency'].max() - cluster_summary['Frequency'].min() + 1e-10)
    )
    cluster_summary['Monetary_norm'] = (
        1 - (cluster_summary['Monetary'] - cluster_summary['Monetary'].min()) /
        (cluster_summary['Monetary'].max() - cluster_summary['Monetary'].min() + 1e-10)
    )
    
    # Risk score: weighted average (equal weights)
    cluster_summary['RiskScore'] = (
        cluster_summary['Recency_norm'] * 0.4 +
        cluster_summary['Frequency_norm'] * 0.3 +
        cluster_summary['Monetary_norm'] * 0.3
    )
    
    # Identify high-risk cluster (highest risk score)
    high_risk_cluster_id = cluster_summary['RiskScore'].idxmax()
    
    return high_risk_cluster_id, cluster_summary


def create_proxy_target_variable(
    df: pd.DataFrame,
    customer_id_col: str = 'CustomerId',
    transaction_date_col: str = 'TransactionDate',
    transaction_amount_col: str = 'TransactionAmount',
    snapshot_date: Optional[pd.Timestamp] = None,
    n_clusters: int = 3,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, dict]:
    """
    Create a proxy credit risk target variable (is_high_risk) from transactional data.
    
    This function implements the complete pipeline:
    1. Calculate RFM metrics
    2. Segment customers using K-Means
    3. Identify high-risk cluster
    4. Create binary target variable
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input DataFrame with transactional data
    customer_id_col : str
        Column name for customer ID (default: 'CustomerId')
    transaction_date_col : str
        Column name for transaction date (default: 'TransactionDate')
    transaction_amount_col : str
        Column name for transaction amount (default: 'TransactionAmount')
    snapshot_date : pd.Timestamp, optional
        Reference date for recency calculation
    n_clusters : int
        Number of clusters for K-Means (default: 3)
    random_state : int
        Random state for reproducibility (default: 42)
        
    Returns:
    --------
    tuple
        (df_with_target, rfm_summary, metadata)
        - df_with_target: Original DataFrame with added 'is_high_risk' column
        - rfm_summary: Customer-level DataFrame with RFM metrics and cluster labels
        - metadata: Dictionary with clustering information and statistics
    """
    # Determine snapshot date if not provided
    if snapshot_date is None:
        # Ensure transaction date is datetime
        df_work = df.copy()
        if not pd.api.types.is_datetime64_any_dtype(df_work[transaction_date_col]):
            df_work[transaction_date_col] = pd.to_datetime(
                df_work[transaction_date_col], 
                errors='coerce'
            )
        snapshot_date = df_work[transaction_date_col].max() + pd.Timedelta(days=1)
    
    # Step 1: Calculate RFM metrics
    rfm_df = calculate_rfm_metrics(
        df=df,
        customer_id_col=customer_id_col,
        transaction_date_col=transaction_date_col,
        transaction_amount_col=transaction_amount_col,
        snapshot_date=snapshot_date
    )
    
    # Step 2: Segment customers using K-Means
    rfm_with_clusters, kmeans_model, scaler = segment_customers_with_kmeans(
        rfm_df=rfm_df,
        n_clusters=n_clusters,
        random_state=random_state,
        customer_id_col=customer_id_col
    )
    
    # Step 3: Identify high-risk cluster
    high_risk_cluster_id, cluster_summary = identify_high_risk_cluster(
        rfm_with_clusters=rfm_with_clusters,
        customer_id_col=customer_id_col
    )
    
    # Step 4: Create binary target variable
    rfm_with_clusters['is_high_risk'] = (
        rfm_with_clusters['Cluster'] == high_risk_cluster_id
    ).astype(int)
    
    # Step 5: Merge target variable back to original dataset
    df_with_target = df.merge(
        rfm_with_clusters[[customer_id_col, 'is_high_risk']],
        on=customer_id_col,
        how='left'
    )
    
    # Fill any missing values (shouldn't happen, but safety check)
    df_with_target['is_high_risk'] = df_with_target['is_high_risk'].fillna(0).astype(int)
    
    # Prepare metadata
    metadata = {
        'high_risk_cluster_id': int(high_risk_cluster_id),
        'n_clusters': n_clusters,
        'random_state': random_state,
        'snapshot_date': str(snapshot_date),
        'cluster_summary': cluster_summary.to_dict(),
        'target_distribution': {
            'high_risk_count': int(rfm_with_clusters['is_high_risk'].sum()),
            'low_risk_count': int((rfm_with_clusters['is_high_risk'] == 0).sum()),
            'high_risk_pct': float(rfm_with_clusters['is_high_risk'].mean() * 100)
        },
        'rfm_statistics': {
            'recency_mean': float(rfm_df['Recency'].mean()),
            'recency_std': float(rfm_df['Recency'].std()),
            'frequency_mean': float(rfm_df['Frequency'].mean()),
            'frequency_std': float(rfm_df['Frequency'].std()),
            'monetary_mean': float(rfm_df['Monetary'].mean()),
            'monetary_std': float(rfm_df['Monetary'].std())
        }
    }
    
    return df_with_target, rfm_with_clusters, metadata


def map_column_names(
    df: pd.DataFrame,
    transaction_date_col: Optional[str] = None,
    transaction_amount_col: Optional[str] = None
) -> dict:
    """
    Map common column name variations to standard names.
    
    This helper function identifies transaction date and amount columns
    from common naming conventions.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input DataFrame
    transaction_date_col : str, optional
        Explicit column name for transaction date. If None, will auto-detect.
    transaction_amount_col : str, optional
        Explicit column name for transaction amount. If None, will auto-detect.
        
    Returns:
    --------
    dict
        Dictionary with mapped column names:
        {
            'transaction_date_col': 'TransactionDate' or detected name,
            'transaction_amount_col': 'TransactionAmount' or detected name
        }
    """
    mapping = {}
    
    # Auto-detect transaction date column
    if transaction_date_col is None:
        date_candidates = [
            'TransactionDate', 'TransactionStartTime', 'Date', 
            'Timestamp', 'transaction_date', 'transaction_start_time'
        ]
        for candidate in date_candidates:
            if candidate in df.columns:
                mapping['transaction_date_col'] = candidate
                break
        else:
            # Try to find datetime columns
            datetime_cols = df.select_dtypes(include=['datetime64']).columns
            if len(datetime_cols) > 0:
                mapping['transaction_date_col'] = datetime_cols[0]
            else:
                raise ValueError(
                    "Could not auto-detect transaction date column. "
                    "Please specify transaction_date_col parameter."
                )
    else:
        mapping['transaction_date_col'] = transaction_date_col
    
    # Auto-detect transaction amount column
    if transaction_amount_col is None:
        amount_candidates = [
            'TransactionAmount', 'Amount', 'transaction_amount', 
            'amount', 'Value', 'value'
        ]
        for candidate in amount_candidates:
            if candidate in df.columns:
                mapping['transaction_amount_col'] = candidate
                break
        else:
            # Try to find numeric columns that might be amounts
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            # Exclude likely non-amount columns
            exclude = ['CustomerId', 'AccountId', 'SubscriptionId', 'ProviderId', 
                      'ProductId', 'ChannelId', 'PricingStrategy', 'FraudResult']
            amount_cols = [col for col in numeric_cols if col not in exclude]
            if len(amount_cols) > 0:
                mapping['transaction_amount_col'] = amount_cols[0]
            else:
                raise ValueError(
                    "Could not auto-detect transaction amount column. "
                    "Please specify transaction_amount_col parameter."
                )
    else:
        mapping['transaction_amount_col'] = transaction_amount_col
    
    return mapping

