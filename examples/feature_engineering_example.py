"""
Example: Feature Engineering Pipeline Usage

This script demonstrates how to use the feature engineering pipeline
to transform raw transactional data into a model-ready dataset.

Usage:
    python examples/feature_engineering_example.py
"""

import sys
import os
import pandas as pd
import numpy as np

# Add parent directory to path to import src modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.feature_engineering import (
    CustomerAggregationTransformer,
    TemporalFeatureExtractor,
    WOETransformer,
    create_feature_engineering_pipeline,
    build_complete_pipeline
)


def create_sample_data(n_samples: int = 1000, n_customers: int = 100) -> pd.DataFrame:
    """
    Create sample transactional data for demonstration.
    
    Parameters:
    -----------
    n_samples : int
        Number of transactions to generate
    n_customers : int
        Number of unique customers
        
    Returns:
    --------
    pd.DataFrame
        Sample transactional DataFrame
    """
    np.random.seed(42)
    # Generate customer IDs
    customer_ids = [f'CustomerId_{i}' for i in np.random.randint(1, n_customers + 1, n_samples)]
    # Generate timestamps
    start_date = pd.Timestamp('2018-01-01')
    end_date = pd.Timestamp('2019-12-31')
    timestamps = pd.date_range(start_date, end_date, periods=n_samples)
    timestamps = np.random.choice(timestamps, n_samples)
    # Generate transaction amounts (with some negative values for refunds)
    amounts = np.random.lognormal(mean=6, sigma=1.5, size=n_samples)
    # Add some negative transactions (refunds)
    refund_mask = np.random.random(n_samples) < 0.1
    amounts[refund_mask] = -amounts[refund_mask] * 0.5
    # Generate categorical features
    product_categories = np.random.choice(
        ['airtime', 'financial_services', 'utility_bill', 'entertainment'],
        size=n_samples,
        p=[0.3, 0.4, 0.2, 0.1]
    )
    channel_ids = np.random.choice(
        ['ChannelId_1', 'ChannelId_2', 'ChannelId_3'],
        size=n_samples,
        p=[0.3, 0.3, 0.4]
    )
    currency_codes = np.random.choice(['UGX', 'USD', 'EUR'], size=n_samples, p=[0.8, 0.15, 0.05])
    # Create DataFrame
    df = pd.DataFrame({
        'CustomerId': customer_ids,
        'TransactionId': [f'TransactionId_{i}' for i in range(n_samples)],
        'TransactionStartTime': timestamps,
        'Amount': amounts,
        'ProductCategory': product_categories,
        'ChannelId': channel_ids,
        'CurrencyCode': currency_codes,
        'CountryCode': np.random.choice([256, 254, 250], size=n_samples, p=[0.8, 0.1, 0.1]),
        'Value': np.abs(amounts) * np.random.uniform(0.8, 1.2, n_samples),
        'PricingStrategy': np.random.randint(1, 4, size=n_samples)
    })
    
    # Add some missing values to demonstrate imputation
    missing_indices = np.random.choice(df.index, size=int(n_samples * 0.05), replace=False)
    df.loc[missing_indices, 'Amount'] = np.nan
    
    # Create a proxy target variable (e.g., based on negative transactions or high amounts)
    # This is a simple example - in practice, you'd use actual default indicators
    df['target'] = ((df['Amount'] < 0) | (df['Amount'] > df['Amount'].quantile(0.95))).astype(int)
    
    return df


def example_1_step_by_step():
    """Example 1: Step-by-step feature engineering"""
    print("=" * 80)
    print("EXAMPLE 1: Step-by-Step Feature Engineering")
    print("=" * 80)
    
    # Create sample data
    df = create_sample_data(n_samples=500, n_customers=50)
    print(f"\nOriginal data shape: {df.shape}")
    print(f"\nFirst few rows:")
    print(df.head())
    
    # Step 1: Customer-level aggregations
    print("\n" + "-" * 80)
    print("Step 1: Customer-Level Aggregations")
    print("-" * 80)
    agg_transformer = CustomerAggregationTransformer(
        customer_id_col='CustomerId',
        amount_col='Amount'
    )
    df_agg = agg_transformer.fit_transform(df)
    print(f"\nAfter aggregation, shape: {df_agg.shape}")
    print(f"\nNew aggregation columns:")
    print(df_agg[['CustomerId', 'total_amount', 'avg_amount', 'count', 'std_amount']].head())
    
    # Step 2: Temporal feature extraction
    print("\n" + "-" * 80)
    print("Step 2: Temporal Feature Extraction")
    print("-" * 80)
    temporal_transformer = TemporalFeatureExtractor(timestamp_col='TransactionStartTime')
    df_temporal = temporal_transformer.fit_transform(df_agg)
    print(f"\nAfter temporal extraction, shape: {df_temporal.shape}")
    print(f"\nNew temporal columns:")
    temporal_cols = ['transaction_hour', 'transaction_day', 'transaction_month', 'transaction_year']
    print(df_temporal[['TransactionStartTime'] + temporal_cols].head())
    
    # Step 3: Define feature types
    numerical_features = ['Amount', 'Value', 'PricingStrategy', 
                         'total_amount', 'avg_amount', 'count', 'std_amount'] + temporal_cols
    categorical_nominal = ['ProductCategory', 'ChannelId', 'CurrencyCode']
    
    # Step 4: Create and apply preprocessing pipeline
    print("\n" + "-" * 80)
    print("Step 3: Preprocessing Pipeline (Imputation + Encoding + Scaling)")
    print("-" * 80)
    
    # Separate target
    y = df_temporal['target']
    X = df_temporal.drop(columns=['target', 'CustomerId', 'TransactionId', 'TransactionStartTime'])
    
    # Create pipeline
    pipeline = create_feature_engineering_pipeline(
        numerical_features=numerical_features,
        categorical_nominal=categorical_nominal,
        use_scaling=True,
        scaling_method='standard'
    )
    
    # Fit and transform
    X_transformed = pipeline.fit_transform(X, y)
    
    # Get feature names
    feature_names = pipeline.named_steps['preprocessor'].get_feature_names_out()
    df_processed = pd.DataFrame(X_transformed, columns=feature_names, index=X.index)
    
    print(f"\nAfter preprocessing, shape: {df_processed.shape}")
    print(f"\nFeature names:")
    print(list(df_processed.columns))
    print(f"\nFirst few rows of processed data:")
    print(df_processed.head())
    
    return df_processed, y


def example_2_woe_iv():
    """Example 2: WoE and IV Transformation"""
    print("\n\n" + "=" * 80)
    print("EXAMPLE 2: Weight of Evidence (WoE) and Information Value (IV)")
    print("=" * 80)
    
    print("""
    Weight of Evidence (WoE):
    -------------------------
    WoE is a transformation technique used in credit scoring to convert categorical 
    or binned numerical features into a numerical scale that reflects predictive power.
    
    Formula: WoE = ln((% of non-events in bin) / (% of events in bin))
    
    Interpretation:
    - Positive WoE: Higher proportion of non-events (good customers) in the bin
    - Negative WoE: Higher proportion of events (bad customers) in the bin
    - Higher absolute WoE indicates stronger predictive power
    
    Information Value (IV):
    -----------------------
    IV measures the predictive power of a feature.
    
    Formula: IV = Σ (pct_non_events - pct_events) × WoE
    
    Interpretation:
    - < 0.02: Not useful for prediction
    - 0.02 - 0.1: Weak predictive power
    - 0.1 - 0.3: Medium predictive power
    - 0.3 - 0.5: Strong predictive power
    - > 0.5: Suspicious (check for data leakage)
    """)
    
    # Create sample data
    df = create_sample_data(n_samples=1000, n_customers=100)
    
    # Apply aggregations and temporal extraction
    agg_transformer = CustomerAggregationTransformer()
    df_agg = agg_transformer.fit_transform(df)
    
    temporal_transformer = TemporalFeatureExtractor()
    df_temporal = temporal_transformer.fit_transform(df_agg)
    
    # Select features for WoE transformation
    features_for_woe = ['Amount', 'total_amount', 'avg_amount', 'std_amount', 
                       'transaction_hour', 'transaction_day']
    
    # Separate target
    y = df_temporal['target']
    X_woe = df_temporal[features_for_woe].copy()
    
    # Create and fit WoE transformer
    print("\n" + "-" * 80)
    print("Fitting WoE Transformer...")
    print("-" * 80)
    woe_transformer = WOETransformer(bins=10, strategy='quantile')
    woe_transformer.fit(X_woe, y)
    
    # Transform features
    X_woe_transformed = woe_transformer.transform(X_woe)
    print(f"\nOriginal features shape: {X_woe.shape}")
    print(f"WoE transformed features shape: {X_woe_transformed.shape}")
    print(f"\nWoE transformed features (first few rows):")
    print(X_woe_transformed.head())
    
    # Get IV scores
    print("\n" + "-" * 80)
    print("Information Value (IV) Scores:")
    print("-" * 80)
    iv_scores = woe_transformer.get_iv_scores()
    print(iv_scores.to_string(index=False))
    
    # Display interpretation
    print("\n" + "-" * 80)
    print("Feature Selection Recommendation:")
    print("-" * 80)
    strong_features = iv_scores[iv_scores['iv_score'] >= 0.1]['feature'].tolist()
    if strong_features:
        print(f"Features with medium to strong predictive power (IV >= 0.1):")
        for feat in strong_features:
            iv = iv_scores[iv_scores['feature'] == feat]['iv_score'].values[0]
            print(f"  - {feat}: IV = {iv:.4f}")
    else:
        print("No features with IV >= 0.1 found in this sample data.")
    
    return X_woe_transformed, woe_transformer, iv_scores


def example_3_complete_pipeline():
    """Example 3: Complete pipeline using build_complete_pipeline"""
    print("\n\n" + "=" * 80)
    print("EXAMPLE 3: Complete Pipeline (One-Step)")
    print("=" * 80)
    
    # Create sample data
    df = create_sample_data(n_samples=500, n_customers=50)
    print(f"\nOriginal data shape: {df.shape}")
    
    # Build complete pipeline
    print("\nBuilding complete pipeline...")
    df_transformed, preprocessing_pipeline, woe_transformer, iv_scores = build_complete_pipeline(
        df=df,
        customer_id_col='CustomerId',
        amount_col='Amount',
        timestamp_col='TransactionStartTime',
        target_col='target',
        apply_aggregations=True,
        apply_temporal_extraction=True,
        apply_woe=False,  # Set to True if you want WoE transformation
        use_scaling=True
    )
    
    print(f"\nTransformed data shape: {df_transformed.shape}")
    print(f"\nFeature names ({len(df_transformed.columns)} features):")
    print(list(df_transformed.columns))
    print(f"\nFirst few rows:")
    print(df_transformed.head())
    
    print("\n" + "-" * 80)
    print("Pipeline Summary:")
    print("-" * 80)
    print(f"- Preprocessing pipeline: {type(preprocessing_pipeline).__name__}")
    print(f"- WoE transformer: {type(woe_transformer).__name__ if woe_transformer else 'None'}")
    print(f"- IV scores available: {iv_scores is not None}")
    
    return df_transformed, preprocessing_pipeline


def example_4_new_data():
    """Example 4: Transforming new data (fit_transform vs transform)"""
    print("\n\n" + "=" * 80)
    print("EXAMPLE 4: Transforming New Data")
    print("=" * 80)
    
    # Create training data
    df_train = create_sample_data(n_samples=1000, n_customers=100)
    
    # Build and fit pipeline on training data
    print("\nFitting pipeline on training data...")
    df_train_processed, pipeline, _, _ = build_complete_pipeline(
        df=df_train,
        target_col='target',
        apply_aggregations=True,
        apply_temporal_extraction=True,
        use_scaling=True
    )
    
    print(f"Training data processed shape: {df_train_processed.shape}")
    
    # Create new (test) data
    df_test = create_sample_data(n_samples=200, n_customers=30)
    
    # Apply same transformations to test data
    print("\nApplying fitted pipeline to new test data...")
    
    # We need to apply aggregations and temporal extraction first
    agg_transformer = CustomerAggregationTransformer()
    agg_transformer.fit(df_train[['CustomerId', 'Amount']])  # Fit on training data
    df_test_agg = agg_transformer.transform(df_test)
    
    temporal_transformer = TemporalFeatureExtractor()
    df_test_temporal = temporal_transformer.fit_transform(df_test_agg)
    
    # Prepare test data (same columns as training)
    y_test = df_test_temporal['target'] if 'target' in df_test_temporal.columns else None
    X_test = df_test_temporal.drop(
        columns=['target', 'CustomerId', 'TransactionId', 'TransactionStartTime'],
        errors='ignore'
    )
    
    # Transform test data using fitted pipeline
    X_test_transformed = pipeline.transform(X_test)
    
    feature_names = pipeline.named_steps['preprocessor'].get_feature_names_out()
    df_test_processed = pd.DataFrame(
        X_test_transformed,
        columns=feature_names,
        index=X_test.index
    )
    
    print(f"Test data processed shape: {df_test_processed.shape}")
    print(f"\nFirst few rows of transformed test data:")
    print(df_test_processed.head())
    
    print("\n✓ Pipeline successfully applied to new data!")
    print("  (Note: The pipeline was fit on training data and can now transform any new data)")


if __name__ == '__main__':
    print("""
    ================================================================================
    Feature Engineering Pipeline - Demonstration
    ================================================================================
    
    This script demonstrates the usage of the feature engineering pipeline
    for credit risk modeling.
    
    Features covered:
    - Customer-level aggregations
    - Temporal feature extraction
    - Missing value imputation
    - Categorical encoding (One-Hot and Ordinal)
    - Feature scaling
    - Weight of Evidence (WoE) and Information Value (IV)
    - Transforming new data
    """)
    
    try:
        # Run examples
        example_1_step_by_step()
        example_2_woe_iv()
        example_3_complete_pipeline()
        example_4_new_data()
        
        print("\n\n" + "=" * 80)
        print("All examples completed successfully!")
        print("=" * 80)
        
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()

