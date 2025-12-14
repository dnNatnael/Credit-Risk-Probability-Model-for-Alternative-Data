"""
Script to create proxy target variable from transactional data.

This script processes the raw transactional data and creates a binary
proxy credit risk target variable (is_high_risk) using RFM metrics
and K-Means clustering.

Usage:
    python scripts/create_proxy_target.py [--input INPUT_FILE] [--output OUTPUT_FILE]
"""

import argparse
import pandas as pd
import json
from pathlib import Path
import sys

# Add project root to path (from scripts folder, go up one level)
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data_processing import create_proxy_target_variable


def main():
    """Main function to create proxy target variable."""
    parser = argparse.ArgumentParser(
        description='Create proxy credit risk target variable from transactional data'
    )
    parser.add_argument(
        '--input',
        type=str,
        default='data/raw/data.csv',
        help='Input CSV file path (default: data/raw/data.csv)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='data/processed/data_with_target.csv',
        help='Output CSV file path (default: data/processed/data_with_target.csv)'
    )
    parser.add_argument(
        '--rfm-output',
        type=str,
        default='data/processed/rfm_summary.csv',
        help='Output path for RFM summary (default: data/processed/rfm_summary.csv)'
    )
    parser.add_argument(
        '--metadata-output',
        type=str,
        default='data/processed/target_metadata.json',
        help='Output path for metadata JSON (default: data/processed/target_metadata.json)'
    )
    parser.add_argument(
        '--n-clusters',
        type=int,
        default=3,
        help='Number of clusters for K-Means (default: 3)'
    )
    parser.add_argument(
        '--random-state',
        type=int,
        default=42,
        help='Random state for reproducibility (default: 42)'
    )
    
    args = parser.parse_args()
    
    # Resolve paths relative to project root
    project_root = Path(__file__).parent.parent
    input_path = project_root / args.input if not Path(args.input).is_absolute() else Path(args.input)
    output_path = project_root / args.output if not Path(args.output).is_absolute() else Path(args.output)
    rfm_output_path = project_root / args.rfm_output if not Path(args.rfm_output).is_absolute() else Path(args.rfm_output)
    metadata_output_path = project_root / args.metadata_output if not Path(args.metadata_output).is_absolute() else Path(args.metadata_output)
    
    # Load data
    print(f"Loading data from {input_path}...")
    try:
        df = pd.read_csv(input_path)
        print(f"Loaded {len(df):,} transactions")
        print(f"Columns: {df.columns.tolist()}")
    except FileNotFoundError:
        print(f"Error: Input file '{input_path}' not found.")
        return 1
    except Exception as e:
        print(f"Error loading data: {e}")
        return 1
    
    # Map column names to standard format
    # Check for common column name variations
    customer_id_col = 'CustomerId'
    
    # Auto-detect transaction date column
    if 'TransactionStartTime' in df.columns:
        transaction_date_col = 'TransactionStartTime'
    elif 'TransactionDate' in df.columns:
        transaction_date_col = 'TransactionDate'
    else:
        # Try to find datetime columns
        datetime_cols = df.select_dtypes(include=['datetime64']).columns
        if len(datetime_cols) > 0:
            transaction_date_col = datetime_cols[0]
            print(f"Auto-detected transaction date column: {transaction_date_col}")
        else:
            print("Error: Could not find transaction date column.")
            print("Please ensure your data has a column named 'TransactionDate' or 'TransactionStartTime'")
            return 1
    
    # Auto-detect transaction amount column
    if 'Amount' in df.columns:
        transaction_amount_col = 'Amount'
    elif 'TransactionAmount' in df.columns:
        transaction_amount_col = 'TransactionAmount'
    else:
        # Try to find numeric columns
        numeric_cols = df.select_dtypes(include=['number']).columns
        exclude = ['CustomerId', 'AccountId', 'SubscriptionId', 'ProviderId', 
                  'ProductId', 'ChannelId', 'PricingStrategy', 'FraudResult']
        amount_cols = [col for col in numeric_cols if col not in exclude]
        if len(amount_cols) > 0:
            transaction_amount_col = amount_cols[0]
            print(f"Auto-detected transaction amount column: {transaction_amount_col}")
        else:
            print("Error: Could not find transaction amount column.")
            print("Please ensure your data has a column named 'Amount' or 'TransactionAmount'")
            return 1
    
    print(f"\nUsing columns:")
    print(f"  Customer ID: {customer_id_col}")
    print(f"  Transaction Date: {transaction_date_col}")
    print(f"  Transaction Amount: {transaction_amount_col}")
    
    # Create proxy target variable
    print(f"\nCreating proxy target variable...")
    print(f"  Number of clusters: {args.n_clusters}")
    print(f"  Random state: {args.random_state}")
    
    try:
        df_with_target, rfm_summary, metadata = create_proxy_target_variable(
            df=df,
            customer_id_col=customer_id_col,
            transaction_date_col=transaction_date_col,
            transaction_amount_col=transaction_amount_col,
            n_clusters=args.n_clusters,
            random_state=args.random_state
        )
        
        print(f"\n✓ Successfully created proxy target variable")
        print(f"\nTarget variable distribution:")
        target_dist = df_with_target['is_high_risk'].value_counts()
        print(f"  High-risk (1): {target_dist.get(1, 0):,} ({target_dist.get(1, 0)/len(df_with_target)*100:.2f}%)")
        print(f"  Low-risk (0): {target_dist.get(0, 0):,} ({target_dist.get(0, 0)/len(df_with_target)*100:.2f}%)")
        
        print(f"\nHigh-risk cluster ID: {metadata['high_risk_cluster_id']}")
        print(f"Number of unique customers: {rfm_summary['CustomerId'].nunique():,}")
        
    except Exception as e:
        print(f"Error creating proxy target: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # Create output directory if it doesn't exist
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rfm_output_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save results
    print(f"\nSaving results...")
    try:
        df_with_target.to_csv(output_path, index=False)
        print(f"  ✓ Saved data with target to: {output_path}")
        
        rfm_summary.to_csv(rfm_output_path, index=False)
        print(f"  ✓ Saved RFM summary to: {rfm_output_path}")
        
        with open(metadata_output_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        print(f"  ✓ Saved metadata to: {metadata_output_path}")
        
    except Exception as e:
        print(f"Error saving results: {e}")
        return 1
    
    print(f"\n✓ All done! Proxy target variable created successfully.")
    return 0


if __name__ == '__main__':
    exit(main())

