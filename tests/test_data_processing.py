"""
Unit tests for data processing and ML training functions.
"""

import pytest
import pandas as pd
import numpy as np
from src.data_processing import calculate_rfm_metrics, create_proxy_target_variable
from src.ml_training import MLTrainingPipeline, create_sample_dataset


class TestDataProcessing:
    """Test cases for data processing functions."""
    
    def test_calculate_rfm_metrics_basic(self):
        """Test RFM metrics calculation with sample data."""
        # Create sample transaction data
        data = {
            'CustomerId': [1, 1, 2, 2, 3],
            'TransactionDate': ['2023-01-01', '2023-01-15', '2023-01-10', '2023-01-20', '2023-01-05'],
            'TransactionAmount': [100, 200, 150, 300, 50]
        }
        df = pd.DataFrame(data)
        
        # Calculate RFM metrics
        rfm_df = calculate_rfm_metrics(df)
        
        # Assertions
        assert len(rfm_df) == 3  # Three unique customers
        assert 'Recency' in rfm_df.columns
        assert 'Frequency' in rfm_df.columns
        assert 'Monetary' in rfm_df.columns
        assert rfm_df['Frequency'].sum() == 5  # Total transactions
        assert all(rfm_df['Recency'] >= 0)  # Recency should be non-negative
        assert all(rfm_df['Frequency'] > 0)  # Frequency should be positive
        assert all(rfm_df['Monetary'] >= 0)  # Monetary should be non-negative
    
    def test_calculate_rfm_metrics_edge_cases(self):
        """Test RFM metrics calculation with edge cases."""
        # Test with single customer
        data = {
            'CustomerId': [1, 1, 1],
            'TransactionDate': ['2023-01-01', '2023-01-02', '2023-01-03'],
            'TransactionAmount': [100, 0, -50]  # Include zero and negative amounts
        }
        df = pd.DataFrame(data)
        
        rfm_df = calculate_rfm_metrics(df)
        
        assert len(rfm_df) == 1
        assert rfm_df.iloc[0]['Frequency'] == 3
        assert rfm_df.iloc[0]['Monetary'] >= 0  # Should handle negative amounts
    
    def test_create_proxy_target_variable(self):
        """Test proxy target variable creation."""
        # Create sample transaction data
        np.random.seed(42)
        data = {
            'CustomerId': np.repeat(range(1, 11), 5),  # 10 customers, 5 transactions each
            'TransactionDate': pd.date_range('2023-01-01', periods=50, freq='D'),
            'TransactionAmount': np.random.uniform(10, 500, 50)
        }
        df = pd.DataFrame(data)
        
        # Create proxy target variable
        df_with_target, rfm_summary, metadata = create_proxy_target_variable(df)
        
        # Assertions
        assert 'is_high_risk' in df_with_target.columns
        assert df_with_target['is_high_risk'].dtype == int
        assert df_with_target['is_high_risk'].isin([0, 1]).all()
        assert len(rfm_summary) == 10  # 10 unique customers
        assert 'high_risk_cluster_id' in metadata
        assert 'target_distribution' in metadata
        assert metadata['target_distribution']['high_risk_count'] + metadata['target_distribution']['low_risk_count'] == 10
    
    def test_create_proxy_target_variable_reproducibility(self):
        """Test that proxy target creation is reproducible."""
        # Create sample data
        np.random.seed(42)
        data = {
            'CustomerId': np.repeat(range(1, 6), 3),
            'TransactionDate': pd.date_range('2023-01-01', periods=15, freq='D'),
            'TransactionAmount': np.random.uniform(10, 100, 15)
        }
        df = pd.DataFrame(data)
        
        # Create proxy target twice with same random state
        _, rfm1, metadata1 = create_proxy_target_variable(df, random_state=42)
        _, rfm2, metadata2 = create_proxy_target_variable(df, random_state=42)
        
        # Should be identical
        pd.testing.assert_frame_equal(rfm1, rfm2)
        assert metadata1['high_risk_cluster_id'] == metadata2['high_risk_cluster_id']


class TestMLTraining:
    """Test cases for ML training pipeline."""
    
    def test_create_sample_dataset(self):
        """Test sample dataset creation."""
        df = create_sample_dataset(n_samples=100, random_state=42)
        
        # Check structure
        assert len(df) == 100
        assert 'CustomerId' in df.columns
        assert 'Recency' in df.columns
        assert 'Frequency' in df.columns
        assert 'Monetary' in df.columns
        assert 'is_high_risk' in df.columns
        
        # Check data types and ranges
        assert df['is_high_risk'].dtype == int
        assert df['is_high_risk'].isin([0, 1]).all()
        assert all(df['Recency'] >= 0)
        assert all(df['Frequency'] >= 1)
        assert all(df['Monetary'] > 0)
    
    def test_ml_pipeline_initialization(self):
        """Test ML pipeline initialization."""
        pipeline = MLTrainingPipeline(experiment_name="test_experiment", random_state=42)
        
        assert pipeline.experiment_name == "test_experiment"
        assert pipeline.random_state == 42
        assert hasattr(pipeline, 'scaler')
        assert hasattr(pipeline, 'models')
        assert hasattr(pipeline, 'best_models')
        assert hasattr(pipeline, 'results')
    
    def test_data_preparation(self):
        """Test data preparation functionality."""
        # Create sample data
        df = create_sample_dataset(n_samples=100, random_state=42)
        pipeline = MLTrainingPipeline(random_state=42)
        
        # Prepare data
        X_train, X_test, y_train, y_test, feature_names = pipeline.prepare_data(
            df, target_col='is_high_risk', test_size=0.2
        )
        
        # Check shapes
        assert len(X_train) == 80  # 80% for training
        assert len(X_test) == 20   # 20% for testing
        assert len(y_train) == 80
        assert len(y_test) == 20
        
        # Check feature names
        assert feature_names == ['Recency', 'Frequency', 'Monetary']
        
        # Check that features are scaled (mean should be close to 0, std close to 1)
        assert abs(X_train.mean().mean()) < 0.1  # Mean close to 0
        assert abs(X_train.std().mean() - 1.0) < 0.1  # Std close to 1
    
    def test_model_configs(self):
        """Test model configuration retrieval."""
        pipeline = MLTrainingPipeline(random_state=42)
        configs = pipeline.get_model_configs()
        
        # Check that all expected models are present
        expected_models = ['logistic_regression', 'decision_tree', 'random_forest', 'gradient_boosting']
        for model_name in expected_models:
            assert model_name in configs
            assert 'model' in configs[model_name]
            assert 'param_grid' in configs[model_name]
    
    def test_model_evaluation(self):
        """Test model evaluation functionality."""
        from sklearn.linear_model import LogisticRegression
        
        # Create sample data and train a simple model
        df = create_sample_dataset(n_samples=100, random_state=42)
        pipeline = MLTrainingPipeline(random_state=42)
        X_train, X_test, y_train, y_test, _ = pipeline.prepare_data(df, test_size=0.2)
        
        # Train a simple model
        model = LogisticRegression(random_state=42)
        model.fit(X_train, y_train)
        
        # Evaluate model
        metrics = pipeline.evaluate_model(model, X_test, y_test)
        
        # Check that all expected metrics are present
        expected_metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']
        for metric in expected_metrics:
            assert metric in metrics
            assert 0 <= metrics[metric] <= 1  # All metrics should be between 0 and 1
    
    def test_sample_dataset_reproducibility(self):
        """Test that sample dataset creation is reproducible."""
        df1 = create_sample_dataset(n_samples=50, random_state=42)
        df2 = create_sample_dataset(n_samples=50, random_state=42)
        
        # Should be identical
        pd.testing.assert_frame_equal(df1, df2)
    
    def test_feature_scaling_consistency(self):
        """Test that feature scaling is applied consistently."""
        df = create_sample_dataset(n_samples=100, random_state=42)
        pipeline = MLTrainingPipeline(random_state=42)
        
        X_train, X_test, _, _, _ = pipeline.prepare_data(df, test_size=0.2)
        
        # Test set should be scaled using training set parameters
        # This means test set might not have exactly mean=0, std=1
        # but should be in reasonable range
        assert all(abs(X_test.mean()) < 2)  # Reasonable range for scaled features
        assert all(abs(X_test.std()) < 3)   # Reasonable range for scaled features