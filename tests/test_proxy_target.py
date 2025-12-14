"""Test script for proxy target variable creation."""
from pathlib import Path
import sys
import pandas as pd

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data_processing import create_proxy_target_variable


def test_proxy_target_creation():
    """Test proxy target variable creation with sample data."""
    # Load sample data
    data_path = project_root / 'data' / 'raw' / 'data.csv'
    if not data_path.exists():
        print(f"Warning: Data file not found at {data_path}")
        print("Skipping test.")
        return
    
    df = pd.read_csv(data_path, nrows=1000)
    print(f"Loaded {len(df)} transactions")
    
    # Create proxy target
    result = create_proxy_target_variable(
        df, 
        transaction_date_col='TransactionStartTime', 
        transaction_amount_col='Amount'
    )
    
    df_with_target = result[0]
    rfm_summary = result[1]
    metadata = result[2]
    
    # Assertions
    assert 'is_high_risk' in df_with_target.columns, "Target variable 'is_high_risk' not found"
    assert df_with_target['is_high_risk'].isin([0, 1]).all(), "Target variable must be binary (0 or 1)"
    assert len(df_with_target) == len(df), "Output should have same number of rows as input"
    
    print(f"\n✓ Test passed!")
    print(f"  High-risk: {df_with_target['is_high_risk'].sum()}")
    print(f"  Low-risk: {(df_with_target['is_high_risk']==0).sum()}")
    print(f"  High-risk cluster ID: {metadata['high_risk_cluster_id']}")
    print(f"\nSample output:")
    print(df_with_target[['CustomerId', 'Amount', 'is_high_risk']].head())


if __name__ == '__main__':
    test_proxy_target_creation()

