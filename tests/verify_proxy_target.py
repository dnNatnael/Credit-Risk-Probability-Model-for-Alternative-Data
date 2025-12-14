"""Verification script for proxy target variable output."""
from pathlib import Path
import pandas as pd
import json

# Get project root
project_root = Path(__file__).parent.parent


def verify_proxy_target():
    """Verify the proxy target variable output files."""
    # Paths
    data_path = project_root / 'data' / 'processed' / 'data_with_target.csv'
    metadata_path = project_root / 'data' / 'processed' / 'target_metadata.json'
    
    if not data_path.exists():
        print(f"Error: Data file not found at {data_path}")
        print("Please run scripts/create_proxy_target.py first.")
        return
    
    if not metadata_path.exists():
        print(f"Error: Metadata file not found at {metadata_path}")
        print("Please run scripts/create_proxy_target.py first.")
        return
    
    # Load data
    df = pd.read_csv(data_path, nrows=10)
    print('Sample data with target:')
    print(df[['CustomerId', 'TransactionStartTime', 'Amount', 'is_high_risk']].head())
    
    # Load full data for distribution
    df_full = pd.read_csv(data_path)
    print('\nTarget distribution (full dataset):')
    print(df_full['is_high_risk'].value_counts())
    print(f"\nPercentage:")
    print(df_full['is_high_risk'].value_counts(normalize=True) * 100)
    
    # Load metadata
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    print('\nMetadata summary:')
    print(f"High-risk cluster: {metadata['high_risk_cluster_id']}")
    print(f"High-risk %: {metadata['target_distribution']['high_risk_pct']:.2f}%")
    print(f"Total customers: {metadata['target_distribution']['high_risk_count'] + metadata['target_distribution']['low_risk_count']}")
    print(f"\nRFM Statistics:")
    print(f"  Recency - Mean: {metadata['rfm_statistics']['recency_mean']:.2f}, Std: {metadata['rfm_statistics']['recency_std']:.2f}")
    print(f"  Frequency - Mean: {metadata['rfm_statistics']['frequency_mean']:.2f}, Std: {metadata['rfm_statistics']['frequency_std']:.2f}")
    print(f"  Monetary - Mean: {metadata['rfm_statistics']['monetary_mean']:.2f}, Std: {metadata['rfm_statistics']['monetary_std']:.2f}")


if __name__ == '__main__':
    verify_proxy_target()

