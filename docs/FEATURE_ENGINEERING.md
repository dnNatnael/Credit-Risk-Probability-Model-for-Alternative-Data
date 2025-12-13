# Feature Engineering Pipeline Documentation

## Overview

This document describes the comprehensive feature engineering pipeline for credit risk modeling. The pipeline transforms raw transactional data into model-ready features using scikit-learn's `Pipeline` and `ColumnTransformer`.

## Architecture

The pipeline consists of several modular components:

1. **Customer Aggregation Transformer**: Creates customer-level aggregated features
2. **Temporal Feature Extractor**: Extracts temporal features from timestamps
3. **Preprocessing Pipeline**: Handles imputation, encoding, and scaling
4. **WoE Transformer**: Applies Weight of Evidence transformation
5. **IV Calculator**: Computes Information Value scores for feature selection

## Components

### 1. Customer Aggregation Transformer

**Purpose**: Aggregate transaction-level data to customer-level features.

**Features Created**:
- `total_amount`: Sum of all transaction amounts per customer
- `avg_amount`: Average transaction amount per customer
- `count`: Number of transactions per customer
- `std_amount`: Standard deviation of transaction amounts per customer

**Usage**:
```python
from src.feature_engineering import CustomerAggregationTransformer

transformer = CustomerAggregationTransformer(
    customer_id_col='CustomerId',
    amount_col='Amount'
)
df_transformed = transformer.fit_transform(df)
```

### 2. Temporal Feature Extractor

**Purpose**: Extract temporal features from timestamp columns.

**Features Created**:
- `transaction_hour`: Hour of day (0-23)
- `transaction_day`: Day of month (1-31)
- `transaction_month`: Month (1-12)
- `transaction_year`: Year

**Usage**:
```python
from src.feature_engineering import TemporalFeatureExtractor

transformer = TemporalFeatureExtractor(timestamp_col='TransactionStartTime')
df_transformed = transformer.fit_transform(df)
```

### 3. Preprocessing Pipeline

**Purpose**: Handle missing values, encode categorical variables, and scale features.

**Components**:
- **Missing Value Imputation**:
  - Numerical features: Median imputation (robust to outliers)
  - Categorical features: Mode imputation
  
- **Categorical Encoding**:
  - **One-Hot Encoding**: For nominal categorical features (no inherent order)
  - **Ordinal Encoding**: For ordinal categorical features (has inherent order)
  
- **Feature Scaling**:
  - **StandardScaler**: Standardization (mean=0, std=1) - default
  - **MinMaxScaler**: Normalization (range 0-1)

**Usage**:
```python
from src.feature_engineering import create_feature_engineering_pipeline

pipeline = create_feature_engineering_pipeline(
    numerical_features=['Amount', 'Value', 'total_amount'],
    categorical_nominal=['ProductCategory', 'ChannelId'],
    categorical_ordinal=['PricingStrategy'],  # Optional
    use_scaling=True,
    scaling_method='standard'  # or 'minmax'
)

X_transformed = pipeline.fit_transform(X, y)
```

### 4. Weight of Evidence (WoE) Transformer

**Purpose**: Transform features using Weight of Evidence to enhance predictive power.

**What is WoE?**
WoE is a transformation technique that converts categorical or binned numerical features into a numerical scale reflecting predictive power.

**Formula**: 
```
WoE = ln((% of non-events in bin) / (% of events in bin))
```

**Interpretation**:
- **Positive WoE**: Higher proportion of non-events (good customers) in the bin
- **Negative WoE**: Higher proportion of events (bad customers) in the bin
- **Higher absolute WoE**: Stronger predictive power

**Usage**:
```python
from src.feature_engineering import WOETransformer

woe_transformer = WOETransformer(bins=10, strategy='quantile')
woe_transformer.fit(X, y)
X_woe = woe_transformer.transform(X)

# Get IV scores
iv_scores = woe_transformer.get_iv_scores()
print(iv_scores)
```

### 5. Information Value (IV) Calculator

**Purpose**: Measure the predictive power of features.

**Formula**:
```
IV = Σ (pct_non_events - pct_events) × WoE
```

**Interpretation**:
- **< 0.02**: Not useful for prediction
- **0.02 - 0.1**: Weak predictive power
- **0.1 - 0.3**: Medium predictive power
- **0.3 - 0.5**: Strong predictive power
- **> 0.5**: Suspicious (check for data leakage)

**Usage**:
```python
iv_scores = woe_transformer.get_iv_scores()
print(iv_scores)
```

## Complete Pipeline Usage

### Method 1: Step-by-Step

```python
from src.feature_engineering import (
    CustomerAggregationTransformer,
    TemporalFeatureExtractor,
    create_feature_engineering_pipeline
)

# Step 1: Aggregations
agg_transformer = CustomerAggregationTransformer()
df = agg_transformer.fit_transform(df)

# Step 2: Temporal extraction
temporal_transformer = TemporalFeatureExtractor()
df = temporal_transformer.fit_transform(df)

# Step 3: Preprocessing
pipeline = create_feature_engineering_pipeline(...)
X_transformed = pipeline.fit_transform(X, y)
```

### Method 2: One-Step Function

```python
from src.feature_engineering import build_complete_pipeline

df_transformed, pipeline, woe_transformer, iv_scores = build_complete_pipeline(
    df=df,
    customer_id_col='CustomerId',
    amount_col='Amount',
    timestamp_col='TransactionStartTime',
    target_col='target',
    apply_aggregations=True,
    apply_temporal_extraction=True,
    apply_woe=False,
    use_scaling=True
)
```

## Transforming New Data

The pipeline can be fit once and then used to transform new data:

```python
# Fit on training data
pipeline = create_feature_engineering_pipeline(...)
pipeline.fit(X_train, y_train)

# Transform training data
X_train_transformed = pipeline.transform(X_train)

# Transform new/test data
X_test_transformed = pipeline.transform(X_test)
```

**Important**: When applying aggregations to new data, fit the aggregation transformer on training data first:

```python
# Fit aggregation transformer on training data
agg_transformer = CustomerAggregationTransformer()
agg_transformer.fit(df_train[['CustomerId', 'Amount']])

# Transform both training and test data
df_train_agg = agg_transformer.transform(df_train)
df_test_agg = agg_transformer.transform(df_test)
```

## Best Practices

1. **Feature Selection**: Use IV scores to identify features with strong predictive power (IV >= 0.1)

2. **Missing Values**: 
   - Use median for numerical features (robust to outliers)
   - Use mode for categorical features
   - Document missing value patterns

3. **Categorical Encoding**:
   - Use One-Hot Encoding for nominal features
   - Use Ordinal Encoding only when there's a clear ordering

4. **Scaling**:
   - StandardScaler is preferred for most cases
   - MinMaxScaler if you need features in [0,1] range

5. **WoE Transformation**:
   - Typically applied to numerical features
   - Helps with monotonicity in logistic regression
   - Improves model interpretability

6. **Production Considerations**:
   - Save fitted transformers/pipelines for reuse
   - Ensure consistent feature order in production
   - Monitor feature distributions for drift

## Example Workflow

See `examples/feature_engineering_example.py` for a complete working example.

## Dependencies

All required dependencies are in `requirements.txt`:
- scikit-learn
- pandas
- numpy

No additional libraries required - WoE/IV implementation is custom-built using only scikit-learn and pandas.

