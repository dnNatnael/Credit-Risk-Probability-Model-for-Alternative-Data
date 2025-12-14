"""
Feature Engineering Pipeline for Credit Risk Modeling

This module provides a comprehensive, production-ready feature engineering pipeline
using scikit-learn Pipeline and ColumnTransformer. It includes:

1. Customer-level aggregation features
2. Temporal feature extraction
3. Categorical encoding (One-Hot and Label)
4. Missing value imputation
5. Feature scaling (StandardScaler and MinMaxScaler)
6. Weight of Evidence (WoE) and Information Value (IV) transformation
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OrdinalEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from typing import List, Optional
import warnings
warnings.filterwarnings('ignore')


# =============================================================================
# CUSTOM TRANSFORMERS
# =============================================================================

class CustomerAggregationTransformer(BaseEstimator, TransformerMixin):
    """
    Custom transformer to create customer-level aggregation features.
    
    Creates:
    - Total transaction amount per customer
    - Average transaction amount per customer
    - Transaction count per customer
    - Standard deviation of transaction amounts per customer
    """
    def __init__(self, customer_id_col: str = 'CustomerId', amount_col: str = 'Amount'):
        """
        Parameters:
        -----------
        customer_id_col : str
            Column name containing customer identifiers
        amount_col : str
            Column name containing transaction amounts
        """
        self.customer_id_col = customer_id_col
        self.amount_col = amount_col
        self.agg_stats_ = None
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        """
        Compute aggregation statistics per customer.
        
        Parameters:
        -----------
        X : pd.DataFrame
            Input DataFrame with customer_id_col and amount_col
        y : pd.Series, optional
            Target variable (not used, for sklearn compatibility)
        """
        # Compute aggregation statistics per customer
        self.agg_stats_ = X.groupby(self.customer_id_col)[self.amount_col].agg([
            ('total_amount', 'sum'),
            ('avg_amount', 'mean'),
            ('count', 'count'),
            ('std_amount', 'std')
        ]).fillna(0)  # Fill NaN std with 0 for customers with single transaction
        return self
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform data by merging aggregation features.
        
        Parameters:
        -----------
        X : pd.DataFrame
            Input DataFrame
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with aggregation features merged
        """
        X = X.copy()
        X = X.merge(
            self.agg_stats_,
            left_on=self.customer_id_col,
            right_index=True,
            how='left'
        )
        # Fill any missing values (for new customers not seen in fit)
        agg_cols = ['total_amount', 'avg_amount', 'count', 'std_amount']
        X[agg_cols] = X[agg_cols].fillna(0)
        return X


class TemporalFeatureExtractor(BaseEstimator, TransformerMixin):
    """
    Extract temporal features from a timestamp column.
    
    Extracts:
    - Transaction hour (0-23)
    - Transaction day (1-31)
    - Transaction month (1-12)
    - Transaction year
    """
    def __init__(self, timestamp_col: str = 'TransactionStartTime'):
        """
        Parameters:
        -----------
        timestamp_col : str
            Column name containing datetime/timestamp
        """
        self.timestamp_col = timestamp_col
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):  # noqa: ARG002
        """No fitting required for this transformer."""
        return self
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Extract temporal features from timestamp column.
        
        Parameters:
        -----------
        X : pd.DataFrame
            Input DataFrame with timestamp column
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with temporal features added
        """
        X = X.copy()
        # Ensure timestamp column is datetime
        if not pd.api.types.is_datetime64_any_dtype(X[self.timestamp_col]):
            X[self.timestamp_col] = pd.to_datetime(X[self.timestamp_col], errors='coerce')
        # Extract temporal features
        X['transaction_hour'] = X[self.timestamp_col].dt.hour
        X['transaction_day'] = X[self.timestamp_col].dt.day
        X['transaction_month'] = X[self.timestamp_col].dt.month
        X['transaction_year'] = X[self.timestamp_col].dt.year
        return X


class WOETransformer(BaseEstimator, TransformerMixin):
    """
    Weight of Evidence (WoE) Transformer.
    
    WoE = ln((% of non-events in bin) / (% of events in bin))
    
    WoE is a way to transform categorical or binned numerical features into
    a numerical scale that reflects the predictive power with respect to the target.
    Higher absolute WoE values indicate stronger predictive power.
    """
    def __init__(self, bins: int = 10, strategy: str = 'quantile'):
        """
        Parameters:
        -----------
        bins : int
            Number of bins for continuous features (default: 10)
        strategy : str
            Binning strategy: 'quantile' or 'uniform' (default: 'quantile')
        """
        self.bins = bins
        self.strategy = strategy
        self.woe_dict_ = {}
        self.iv_dict_ = {}
        self.feature_names_ = []
        self.bin_edges_ = {}  # Store bin edges for continuous features
        self.is_numeric_dict_ = {}  # Store whether each feature is numeric 
    def fit(self, X: pd.DataFrame, y: pd.Series):
        """
        Compute WoE and IV for each feature.
        
        Parameters:
        -----------
        X : pd.DataFrame
            Input features
        y : pd.Series
            Binary target variable (0/1)
        """
        self.feature_names_ = list(X.columns)
        # Calculate overall event rate
        total_events = y.sum()
        total_non_events = len(y) - total_events
        if total_events == 0 or total_non_events == 0:
            raise ValueError("Target variable must contain both events and non-events")
        for col in X.columns:
            # Determine if feature is continuous or categorical
            is_numeric = pd.api.types.is_numeric_dtype(X[col])
            self.is_numeric_dict_[col] = is_numeric
            if is_numeric and X[col].nunique() > self.bins:
                # Bin continuous features and store bin edges
                if self.strategy == 'quantile':
                    X_binned, bin_edges = pd.qcut(
                        X[col],
                        q=self.bins,
                        duplicates='drop',
                        retbins=True
                    )
                else:
                    X_binned, bin_edges = pd.cut(
                        X[col],
                        bins=self.bins,
                        duplicates='drop',
                        retbins=True
                    )
                self.bin_edges_[col] = bin_edges
            else:
                # Use categories directly for categorical or low-cardinality numeric
                X_binned = X[col].astype(str)
                self.bin_edges_[col] = None
            # Calculate WoE for each bin/category
            woe_map = {}
            iv = 0
            for category in X_binned.unique():
                if pd.isna(category):
                    continue
                mask = X_binned == category
                events_in_bin = y[mask].sum()
                non_events_in_bin = len(y[mask]) - events_in_bin
                # Calculate percentages
                pct_events = events_in_bin / total_events if total_events > 0 else 0
                pct_non_events = non_events_in_bin / total_non_events if total_non_events > 0 else 0
                # Calculate WoE (avoid division by zero and log of zero)
                if pct_non_events > 0 and pct_events > 0:
                    woe = np.log(pct_non_events / pct_events)
                elif pct_non_events > 0:
                    woe = 3  # Cap at 3 for very rare events
                elif pct_events > 0:
                    woe = -3  # Cap at -3 for very common events
                else:
                    woe = 0
                woe_map[category] = woe
                # Calculate contribution to IV
                if pct_non_events > 0 and pct_events > 0:
                    iv += (pct_non_events - pct_events) * woe
            self.woe_dict_[col] = woe_map
            self.iv_dict_[col] = iv
        return self
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform features using WoE mapping.
        
        Parameters:
        -----------
        X : pd.DataFrame
            Input features
            
        Returns:
        --------
        pd.DataFrame
            Features transformed to WoE values
        """
        X = X.copy()
        X_transformed = pd.DataFrame(index=X.index)
        
        for col in self.feature_names_:
            if col not in X.columns:
                continue
            is_numeric = self.is_numeric_dict_.get(col, pd.api.types.is_numeric_dtype(X[col]))
            woe_map = self.woe_dict_[col]
            bin_edges = self.bin_edges_.get(col, None)
            # Apply same binning as in fit
            if is_numeric and bin_edges is not None:
                # Use stored bin edges for consistent binning
                X_binned = pd.cut(X[col], bins=bin_edges, include_lowest=True, duplicates='drop')
            elif is_numeric:
                # Low-cardinality numeric - treat as categorical
                X_binned = X[col].astype(str)
            else:
                X_binned = X[col].astype(str)
            # Map to WoE values
            X_transformed[col] = X_binned.map(woe_map)
            # Handle values outside training range (map to 0 or nearest bin)
            X_transformed[col] = X_transformed[col].fillna(0)

            # Rename column to indicate WoE transformation
            X_transformed = X_transformed.rename(columns={col: f'{col}_woe'})
        return X_transformed
    def get_iv_scores(self) -> pd.DataFrame:
        """
        Get Information Value (IV) scores for all features.
        
        IV Interpretation:
        - < 0.02: Not useful for prediction
        - 0.02 - 0.1: Weak predictive power
        - 0.1 - 0.3: Medium predictive power
        - 0.3 - 0.5: Strong predictive power
        - > 0.5: Suspicious (may be too good, check for data leakage)
        
        Returns:
        --------
        pd.DataFrame
            DataFrame with feature names and IV scores
        """
        iv_df = pd.DataFrame({
            'feature': list(self.iv_dict_.keys()),
            'iv_score': list(self.iv_dict_.values())
        }).sort_values('iv_score', ascending=False)
        # Add interpretation
        iv_df['interpretation'] = pd.cut(
            iv_df['iv_score'],
            bins=[-np.inf, 0.02, 0.1, 0.3, 0.5, np.inf],
            labels=['Not useful', 'Weak', 'Medium', 'Strong', 'Suspicious']
        )
        return iv_df


# =============================================================================
# MAIN FEATURE ENGINEERING PIPELINE
# =============================================================================

def create_feature_engineering_pipeline(
    numerical_features: List[str],
    categorical_nominal: List[str],
    categorical_ordinal: Optional[List[str]] = None,
    temporal_features: Optional[List[str]] = None,
    use_woe: bool = True,  # noqa: ARG001
    use_scaling: bool = True,
    scaling_method: str = 'standard'
) -> Pipeline:
    """
    Create a comprehensive feature engineering pipeline.
    
    Parameters:
    -----------
    numerical_features : List[str]
        List of numerical feature column names
    categorical_nominal : List[str]
        List of nominal categorical feature column names (for One-Hot Encoding)
    categorical_ordinal : List[str], optional
        List of ordinal categorical feature column names (for Label Encoding)
    temporal_features : List[str], optional
        List of temporal feature column names (already extracted, e.g., transaction_hour)
    use_woe : bool
        Whether to apply WoE transformation (default: True)
    use_scaling : bool
        Whether to apply feature scaling (default: True)
    scaling_method : str
        Scaling method: 'standard' (StandardScaler) or 'minmax' (MinMaxScaler)
        (default: 'standard')
        
    Returns:
    --------
    Pipeline
        Complete feature engineering pipeline
    """
    if categorical_ordinal is None:
        categorical_ordinal = []
    if temporal_features is None:
        temporal_features = []
    # Define preprocessing transformers for each feature type
    transformers = []
    # 1. Numerical features: impute missing values and optionally scale
    if numerical_features:
        num_pipeline_steps = [
            ('imputer', SimpleImputer(strategy='median'))  # Use median for robustness to outliers
        ]
        if use_scaling:
            if scaling_method == 'standard':
                num_pipeline_steps.append(('scaler', StandardScaler()))
            elif scaling_method == 'minmax':
                num_pipeline_steps.append(('scaler', MinMaxScaler()))
            else:
                raise ValueError(f"Unknown scaling method: {scaling_method}")
        transformers.append(
            ('numerical', Pipeline(num_pipeline_steps), numerical_features)
        )
    # 2. Nominal categorical features: impute missing values and One-Hot Encode
    if categorical_nominal:
        cat_nominal_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),  # Mode imputation
            ('onehot', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'))
        ])
        transformers.append(
            ('categorical_nominal', cat_nominal_pipeline, categorical_nominal)
        )
    # 3. Ordinal categorical features: impute missing values and Ordinal Encode
    if categorical_ordinal:
        cat_ordinal_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('ordinal_encode', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
        ])
        transformers.append(
            ('categorical_ordinal', cat_ordinal_pipeline, categorical_ordinal)
        )
    # 4. Temporal features: no transformation needed (already extracted)
    # They are typically treated as numerical or categorical
    # We'll include them in numerical pipeline if they exist
    # Create ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder='passthrough',  # Keep other columns as-is
        verbose_feature_names_out=False
    )
    # Create final pipeline
    pipeline_steps = [('preprocessor', preprocessor)]
    # Optionally add WoE transformation after preprocessing
    # Note: WoE requires target variable, so it should be applied separately
    # in a pipeline that has access to y during fit
    return Pipeline(pipeline_steps)


def create_woe_pipeline(
    features_for_woe: List[str],  # noqa: ARG001
    bins: int = 10,
    strategy: str = 'quantile'
) -> WOETransformer:
    """
    Create a WoE transformation pipeline for selected features.
    
    Parameters:
    -----------
    features_for_woe : List[str]
        List of feature names to apply WoE transformation
    bins : int
        Number of bins for continuous features
    strategy : str
        Binning strategy: 'quantile' or 'uniform'
        
    Returns:
    --------
    WOETransformer
        WoE transformer for the specified features
    """
    return WOETransformer(bins=bins, strategy=strategy)


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def build_complete_pipeline(
    df: pd.DataFrame,
    customer_id_col: str = 'CustomerId',
    amount_col: str = 'Amount',
    timestamp_col: str = 'TransactionStartTime',
    target_col: Optional[str] = None,
    numerical_features: Optional[List[str]] = None,
    categorical_nominal: Optional[List[str]] = None,
    categorical_ordinal: Optional[List[str]] = None,
    apply_aggregations: bool = True,
    apply_temporal_extraction: bool = True,
    apply_woe: bool = False,
    use_scaling: bool = True
) -> tuple:
    """
    Build a complete feature engineering pipeline from raw transactional data.
    
    This function orchestrates the entire feature engineering process:
    1. Customer-level aggregations
    2. Temporal feature extraction
    3. Missing value imputation
    4. Categorical encoding
    5. Feature scaling
    6. Optional WoE transformation
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input DataFrame with raw transactional data
    customer_id_col : str
        Column name for customer ID
    amount_col : str
        Column name for transaction amounts
    timestamp_col : str
        Column name for transaction timestamps
    target_col : str, optional
        Column name for target variable (required if apply_woe=True)
    numerical_features : List[str], optional
        List of numerical features (auto-detected if None)
    categorical_nominal : List[str], optional
        List of nominal categorical features (auto-detected if None)
    categorical_ordinal : List[str], optional
        List of ordinal categorical features
    apply_aggregations : bool
        Whether to apply customer-level aggregations
    apply_temporal_extraction : bool
        Whether to extract temporal features
    apply_woe : bool
        Whether to apply WoE transformation (requires target_col)
    use_scaling : bool
        Whether to apply feature scaling
        
    Returns:
    --------
    tuple
        (transformed_df, preprocessing_pipeline, woe_transformer, iv_scores)
    """
    df_processed = df.copy()
    # Step 1: Customer-level aggregations
    if apply_aggregations:
        agg_transformer = CustomerAggregationTransformer(
            customer_id_col=customer_id_col,
            amount_col=amount_col
        )
        df_processed = agg_transformer.fit_transform(df_processed)
    # Step 2: Temporal feature extraction
    if apply_temporal_extraction:
        temporal_transformer = TemporalFeatureExtractor(timestamp_col=timestamp_col)
        df_processed = temporal_transformer.fit_transform(df_processed)
    # Auto-detect feature types if not provided
    if numerical_features is None:
        numerical_features = df_processed.select_dtypes(
            include=[np.number]
        ).columns.tolist()
        # Exclude target and ID columns
        exclude_cols = [customer_id_col, target_col] if target_col else [customer_id_col]
        numerical_features = [col for col in numerical_features if col not in exclude_cols]
    if categorical_nominal is None:
        categorical_nominal = df_processed.select_dtypes(
            include=['object', 'category']
        ).columns.tolist()
        # Exclude ID columns
        exclude_cols = [customer_id_col, timestamp_col, target_col]
        categorical_nominal = [
            col for col in categorical_nominal
            if col not in exclude_cols and col not in numerical_features
        ]
    # Add temporal features to numerical if they exist
    temporal_cols = ['transaction_hour', 'transaction_day', 'transaction_month', 'transaction_year']
    for col in temporal_cols:
        if col in df_processed.columns and col not in numerical_features:
            numerical_features.append(col)
    # Separate target if provided
    y = None
    if target_col and target_col in df_processed.columns:
        y = df_processed[target_col]
        df_processed = df_processed.drop(columns=[target_col])
    # Step 3: Create and fit preprocessing pipeline
    preprocessing_pipeline = create_feature_engineering_pipeline(
        numerical_features=numerical_features,
        categorical_nominal=categorical_nominal,
        categorical_ordinal=categorical_ordinal or [],
        use_woe=False,  # WoE applied separately
        use_scaling=use_scaling
    )
    # Fit and transform
    if y is not None:
        X_transformed = preprocessing_pipeline.fit_transform(df_processed, y)
    else:
        X_transformed = preprocessing_pipeline.fit_transform(df_processed)
    # Get feature names
    feature_names = preprocessing_pipeline.named_steps['preprocessor'].get_feature_names_out()
    # Convert to DataFrame
    df_transformed = pd.DataFrame(
        X_transformed,
        columns=feature_names,
        index=df_processed.index
    )
    # Step 4: Optional WoE transformation
    woe_transformer = None
    iv_scores = None
    if apply_woe and target_col and y is not None:
        # Select features for WoE (typically numerical and some categorical)
        features_for_woe = numerical_features.copy()
        # Create and fit WoE transformer
        woe_transformer = WOETransformer(bins=10, strategy='quantile')
        woe_transformer.fit(df_processed[features_for_woe], y)
        # Transform features (WoE transformation available if needed)
        # df_woe = woe_transformer.transform(df_processed[features_for_woe])
        # Get IV scores
        iv_scores = woe_transformer.get_iv_scores()
        # Optionally replace original features with WoE features
        # For now, we'll return both
    return df_transformed, preprocessing_pipeline, woe_transformer, iv_scores