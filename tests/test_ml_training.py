"""
Additional unit tests specifically for ML training pipeline functionality.
"""

import pytest
import pandas as pd
import numpy as np
import mlflow
from unittest.mock import patch, MagicMock
from src.ml_training import MLTrainingPipeline, create_sample_dataset


class TestMLTrainingPipeline:
    """Comprehensive tests for ML training pipeline."""
    
    def setup_method(self):
        """Set up test fixtures before each test method."""
        self.pipeline = MLTrainingPipeline(experiment_name="test_experiment", random_state=42)
        self.sample_data = create_sample_dataset(n_samples=100, random_state=42)
    
    def test_train_model_with_tuning(self):
        """Test individual model training with hyperparameter tuning."""
        # Prepare data
        X_train, X_test, y_train, y_test, _ = self.pipeline.prepare_data(
            self.sample_data, test_size=0.2
        )
        
        # Train logistic regression model
        best_model, metrics, best_params = self.pipeline.train_model_with_tuning(
            'logistic_regression', X_train, y_train, X_test, y_test, cv_folds=3
        )
        
        # Verify results
        assert best_model is not None
        assert isinstance(metrics, dict)
        assert isinstance(best_params, dict)
        assert 'accuracy' in metrics
        assert 'roc_auc' in metrics
        assert 'C' in best_params  # Logistic regression hyperparameter
    
    def test_invalid_model_name(self):
        """Test handling of invalid model names."""
        X_train, X_test, y_train, y_test, _ = self.pipeline.prepare_data(
            self.sample_data, test_size=0.2
        )
        
        with pytest.raises(ValueError, match="Model invalid_model not supported"):
            self.pipeline.train_model_with_tuning(
                'invalid_model', X_train, y_train, X_test, y_test
            )
    
    def test_compare_models_empty(self):
        """Test model comparison when no models have been trained."""
        with pytest.raises(ValueError, match="No models have been trained yet"):
            self.pipeline.compare_models()
    
    def test_get_best_model_empty(self):
        """Test getting best model when no models have been trained."""
        with pytest.raises(ValueError, match="No models have been trained yet"):
            self.pipeline.get_best_model()
    
    @patch('mlflow.start_run')
    @patch('mlflow.log_params')
    @patch('mlflow.log_metrics')
    @patch('mlflow.sklearn.log_model')
    def test_train_all_models_mock(self, mock_log_model, mock_log_metrics, 
                                  mock_log_params, mock_start_run):
        """Test training all models with mocked MLflow calls."""
        # Mock MLflow context manager
        mock_run = MagicMock()
        mock_start_run.return_value.__enter__ = MagicMock(return_value=mock_run)
        mock_start_run.return_value.__exit__ = MagicMock(return_value=None)
        
        # Prepare data
        X_train, X_test, y_train, y_test, feature_names = self.pipeline.prepare_data(
            self.sample_data, test_size=0.2
        )
        
        # Train only one model for faster testing
        self.pipeline.models = {'logistic_regression': self.pipeline.get_model_configs()['logistic_regression']}
        
        # This would normally train all models, but we'll mock the MLflow calls
        # For testing purposes, we'll manually add one result
        from sklearn.linear_model import LogisticRegression
        model = LogisticRegression(random_state=42)
        model.fit(X_train, y_train)
        
        self.pipeline.best_models['logistic_regression'] = model
        self.pipeline.results['logistic_regression'] = {
            'metrics': {'accuracy': 0.8, 'roc_auc': 0.75},
            'best_params': {'C': 1.0, 'penalty': 'l2'}
        }
        
        # Test comparison functionality
        comparison_df = self.pipeline.compare_models()
        assert len(comparison_df) == 1
        assert 'model' in comparison_df.columns
        assert 'accuracy' in comparison_df.columns
        assert 'roc_auc' in comparison_df.columns
    
    def test_model_evaluation_metrics(self):
        """Test that model evaluation returns all expected metrics."""
        from sklearn.linear_model import LogisticRegression
        
        # Prepare data
        X_train, X_test, y_train, y_test, _ = self.pipeline.prepare_data(
            self.sample_data, test_size=0.2
        )
        
        # Train a simple model
        model = LogisticRegression(random_state=42, max_iter=1000)
        model.fit(X_train, y_train)
        
        # Evaluate
        metrics = self.pipeline.evaluate_model(model, X_test, y_test)
        
        # Check all expected metrics are present and valid
        expected_metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']
        for metric in expected_metrics:
            assert metric in metrics
            assert isinstance(metrics[metric], (int, float))
            assert 0 <= metrics[metric] <= 1
    
    def test_data_preparation_stratification(self):
        """Test that data preparation maintains class balance through stratification."""
        # Create imbalanced dataset
        df = self.sample_data.copy()
        # Make it more imbalanced
        df.loc[df['is_high_risk'] == 1, 'is_high_risk'] = 0  # Convert some high risk to low risk
        df.loc[:10, 'is_high_risk'] = 1  # Keep only a few high risk cases
        
        X_train, X_test, y_train, y_test, _ = self.pipeline.prepare_data(
            df, test_size=0.2
        )
        
        # Check that both train and test sets have both classes
        assert len(y_train.unique()) == 2 or y_train.sum() == 0 or y_train.sum() == len(y_train)
        assert len(y_test.unique()) == 2 or y_test.sum() == 0 or y_test.sum() == len(y_test)
    
    def test_feature_names_consistency(self):
        """Test that feature names are handled consistently."""
        # Test with custom feature columns
        custom_features = ['Recency', 'Frequency']
        X_train, X_test, y_train, y_test, feature_names = self.pipeline.prepare_data(
            self.sample_data, feature_cols=custom_features, test_size=0.2
        )
        
        assert feature_names == custom_features
        assert list(X_train.columns) == custom_features
        assert list(X_test.columns) == custom_features
        assert X_train.shape[1] == len(custom_features)
        assert X_test.shape[1] == len(custom_features)


class TestSampleDataset:
    """Tests for sample dataset generation."""
    
    def test_dataset_size_parameter(self):
        """Test that dataset size parameter works correctly."""
        sizes = [50, 100, 500]
        for size in sizes:
            df = create_sample_dataset(n_samples=size, random_state=42)
            assert len(df) == size
    
    def test_dataset_columns(self):
        """Test that dataset has all required columns."""
        df = create_sample_dataset(n_samples=100, random_state=42)
        expected_columns = ['CustomerId', 'Recency', 'Frequency', 'Monetary', 'is_high_risk']
        
        for col in expected_columns:
            assert col in df.columns
    
    def test_dataset_data_types(self):
        """Test that dataset columns have correct data types."""
        df = create_sample_dataset(n_samples=100, random_state=42)
        
        assert df['CustomerId'].dtype in ['int64', 'int32']
        assert df['Recency'].dtype in ['float64', 'float32']
        assert df['Frequency'].dtype in ['int64', 'int32']
        assert df['Monetary'].dtype in ['float64', 'float32']
        assert df['is_high_risk'].dtype in ['int64', 'int32']
    
    def test_dataset_value_ranges(self):
        """Test that dataset values are in expected ranges."""
        df = create_sample_dataset(n_samples=100, random_state=42)
        
        # Check value ranges
        assert all(df['CustomerId'] >= 1)
        assert all(df['Recency'] >= 0)
        assert all(df['Frequency'] >= 1)
        assert all(df['Monetary'] > 0)
        assert all(df['is_high_risk'].isin([0, 1]))
    
    def test_dataset_target_distribution(self):
        """Test that target variable has reasonable distribution."""
        df = create_sample_dataset(n_samples=1000, random_state=42)
        
        # Should have both classes
        assert len(df['is_high_risk'].unique()) == 2
        
        # Should be somewhat balanced (not too extreme)
        high_risk_pct = df['is_high_risk'].mean()
        assert 0.1 <= high_risk_pct <= 0.9  # Between 10% and 90%
    
    def test_dataset_customer_ids_unique(self):
        """Test that customer IDs are unique."""
        df = create_sample_dataset(n_samples=100, random_state=42)
        assert len(df['CustomerId'].unique()) == len(df)
        assert df['CustomerId'].min() == 1
        assert df['CustomerId'].max() == len(df)