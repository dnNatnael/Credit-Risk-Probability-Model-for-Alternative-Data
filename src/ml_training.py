"""
Machine Learning Training Pipeline with MLflow Tracking

This module provides a complete ML training pipeline with:
- Model training for multiple algorithms
- Hyperparameter tuning with Grid Search
- MLflow experiment tracking
- Model evaluation and comparison
- Model registration in MLflow Model Registry
"""

import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, roc_auc_score, classification_report
)
from sklearn.preprocessing import StandardScaler
from typing import Dict, Tuple, Any, List
import warnings
warnings.filterwarnings('ignore')


class MLTrainingPipeline:
    """
    Complete ML training pipeline with MLflow tracking and model comparison.
    """
    
    def __init__(self, experiment_name: str = "credit_risk_modeling", random_state: int = 42):
        """
        Initialize the ML training pipeline.
        
        Parameters:
        -----------
        experiment_name : str
            Name of the MLflow experiment
        random_state : int
            Random state for reproducibility
        """
        self.experiment_name = experiment_name
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.models = {}
        self.best_models = {}
        self.results = {}
        
        # Set up MLflow experiment
        mlflow.set_experiment(experiment_name)
    
    def prepare_data(self, df: pd.DataFrame, target_col: str = 'is_high_risk', 
                    feature_cols: List[str] = None, test_size: float = 0.2) -> Tuple:
        """
        Prepare data for training by splitting and scaling features.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input DataFrame with features and target
        target_col : str
            Name of the target column
        feature_cols : List[str], optional
            List of feature column names. If None, uses RFM features.
        test_size : float
            Proportion of data for testing
            
        Returns:
        --------
        tuple
            (X_train, X_test, y_train, y_test, feature_names)
        """
        if feature_cols is None:
            feature_cols = ['Recency', 'Frequency', 'Monetary']
        
        # Extract features and target
        X = df[feature_cols].copy()
        y = df[target_col].copy()
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state, stratify=y
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Convert back to DataFrames for easier handling
        X_train_scaled = pd.DataFrame(X_train_scaled, columns=feature_cols, index=X_train.index)
        X_test_scaled = pd.DataFrame(X_test_scaled, columns=feature_cols, index=X_test.index)
        
        return X_train_scaled, X_test_scaled, y_train, y_test, feature_cols
    
    def get_model_configs(self) -> Dict[str, Dict]:
        """
        Get model configurations with hyperparameter grids for tuning.
        
        Returns:
        --------
        dict
            Dictionary with model configurations
        """
        return {
            'logistic_regression': {
                'model': LogisticRegression(random_state=self.random_state, max_iter=1000),
                'param_grid': {
                    'C': [0.1, 1.0, 10.0],
                    'penalty': ['l1', 'l2'],
                    'solver': ['liblinear']
                }
            },
            'decision_tree': {
                'model': DecisionTreeClassifier(random_state=self.random_state),
                'param_grid': {
                    'max_depth': [3, 5, 10, None],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4]
                }
            },
            'random_forest': {
                'model': RandomForestClassifier(random_state=self.random_state),
                'param_grid': {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [3, 5, 10, None],
                    'min_samples_split': [2, 5, 10]
                }
            },
            'gradient_boosting': {
                'model': GradientBoostingClassifier(random_state=self.random_state),
                'param_grid': {
                    'n_estimators': [50, 100, 200],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'max_depth': [3, 5, 7]
                }
            }
        }
    
    def evaluate_model(self, model, X_test, y_test) -> Dict[str, float]:
        """
        Evaluate model performance using multiple metrics.
        
        Parameters:
        -----------
        model : sklearn model
            Trained model to evaluate
        X_test : pd.DataFrame
            Test features
        y_test : pd.Series
            Test target values
            
        Returns:
        --------
        dict
            Dictionary with evaluation metrics
        """
        # Make predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1_score': f1_score(y_test, y_pred, zero_division=0),
            'roc_auc': roc_auc_score(y_test, y_pred_proba)
        }
        
        return metrics
    
    def train_model_with_tuning(self, model_name: str, X_train, y_train, X_test, y_test, 
                               cv_folds: int = 5) -> Tuple[Any, Dict[str, float], Dict]:
        """
        Train a model with hyperparameter tuning using GridSearchCV.
        
        Parameters:
        -----------
        model_name : str
            Name of the model to train
        X_train, y_train : pd.DataFrame, pd.Series
            Training data
        X_test, y_test : pd.DataFrame, pd.Series
            Test data
        cv_folds : int
            Number of cross-validation folds
            
        Returns:
        --------
        tuple
            (best_model, metrics, best_params)
        """
        model_configs = self.get_model_configs()
        
        if model_name not in model_configs:
            raise ValueError(f"Model {model_name} not supported")
        
        config = model_configs[model_name]
        
        # Perform grid search
        grid_search = GridSearchCV(
            estimator=config['model'],
            param_grid=config['param_grid'],
            cv=cv_folds,
            scoring='roc_auc',
            n_jobs=-1
        )
        
        # Fit the grid search
        grid_search.fit(X_train, y_train)
        
        # Get best model
        best_model = grid_search.best_estimator_
        
        # Evaluate on test set
        metrics = self.evaluate_model(best_model, X_test, y_test)
        
        return best_model, metrics, grid_search.best_params_
    
    def train_all_models(self, X_train, X_test, y_train, y_test, feature_names: List[str]):
        """
        Train all models with hyperparameter tuning and MLflow tracking.
        
        Parameters:
        -----------
        X_train, X_test : pd.DataFrame
            Training and test features
        y_train, y_test : pd.Series
            Training and test targets
        feature_names : List[str]
            Names of feature columns
        """
        model_configs = self.get_model_configs()
        
        for model_name in model_configs.keys():
            with mlflow.start_run(run_name=f"{model_name}_training"):
                print(f"\nTraining {model_name}...")
                
                # Train model with hyperparameter tuning
                best_model, metrics, best_params = self.train_model_with_tuning(
                    model_name, X_train, y_train, X_test, y_test
                )
                
                # Store results
                self.best_models[model_name] = best_model
                self.results[model_name] = {
                    'metrics': metrics,
                    'best_params': best_params
                }
                
                # Log to MLflow
                mlflow.log_params(best_params)
                mlflow.log_metrics(metrics)
                
                # Log model
                mlflow.sklearn.log_model(
                    best_model, 
                    f"{model_name}_model",
                    input_example=X_train.head(1)
                )
                
                # Log feature importance if available
                if hasattr(best_model, 'feature_importances_'):
                    feature_importance = dict(zip(feature_names, best_model.feature_importances_))
                    mlflow.log_params({f"feature_importance_{k}": v for k, v in feature_importance.items()})
                
                # Print results
                print(f"Best parameters: {best_params}")
                print(f"Test metrics: {metrics}")
    
    def compare_models(self) -> pd.DataFrame:
        """
        Compare all trained models and return results DataFrame.
        
        Returns:
        --------
        pd.DataFrame
            Comparison of model performance
        """
        if not self.results:
            raise ValueError("No models have been trained yet")
        
        comparison_data = []
        for model_name, result in self.results.items():
            row = {'model': model_name}
            row.update(result['metrics'])
            comparison_data.append(row)
        
        comparison_df = pd.DataFrame(comparison_data)
        comparison_df = comparison_df.sort_values('roc_auc', ascending=False)
        
        return comparison_df
    
    def get_best_model(self, metric: str = 'roc_auc') -> Tuple[str, Any, Dict]:
        """
        Get the best performing model based on specified metric.
        
        Parameters:
        -----------
        metric : str
            Metric to use for comparison (default: 'roc_auc')
            
        Returns:
        --------
        tuple
            (model_name, model_object, metrics)
        """
        if not self.results:
            raise ValueError("No models have been trained yet")
        
        best_score = -1
        best_model_name = None
        
        for model_name, result in self.results.items():
            if result['metrics'][metric] > best_score:
                best_score = result['metrics'][metric]
                best_model_name = model_name
        
        return (
            best_model_name, 
            self.best_models[best_model_name], 
            self.results[best_model_name]['metrics']
        )
    
    def register_best_model(self, model_name: str = None, metric: str = 'roc_auc'):
        """
        Register the best model in MLflow Model Registry.
        
        Parameters:
        -----------
        model_name : str, optional
            Specific model to register. If None, uses best performing model.
        metric : str
            Metric to use for determining best model
        """
        if model_name is None:
            model_name, _, _ = self.get_best_model(metric)
        
        # Start a new run for model registration
        with mlflow.start_run(run_name=f"{model_name}_registration"):
            model = self.best_models[model_name]
            metrics = self.results[model_name]['metrics']
            
            # Log the model with registration
            model_uri = mlflow.sklearn.log_model(
                model, 
                "model",
                registered_model_name=f"credit_risk_{model_name}"
            ).model_uri
            
            # Log metrics
            mlflow.log_metrics(metrics)
            
            print(f"Registered model: credit_risk_{model_name}")
            print(f"Model URI: {model_uri}")
            print(f"Performance: {metrics}")
            
            return model_uri


def create_sample_dataset(n_samples: int = 1000, random_state: int = 42) -> pd.DataFrame:
    """
    Create a sample dataset for demonstration purposes.
    
    Parameters:
    -----------
    n_samples : int
        Number of samples to generate
    random_state : int
        Random state for reproducibility
        
    Returns:
    --------
    pd.DataFrame
        Sample dataset with RFM features and target
    """
    np.random.seed(random_state)
    
    # Generate RFM features with some correlation to create realistic patterns
    recency = np.random.exponential(scale=30, size=n_samples)
    frequency = np.random.poisson(lam=5, size=n_samples) + 1
    monetary = np.random.lognormal(mean=4, sigma=1, size=n_samples)
    
    # Create target variable with some logic
    # High risk: high recency, low frequency, low monetary
    risk_score = (
        (recency / recency.max()) * 0.4 +
        (1 - frequency / frequency.max()) * 0.3 +
        (1 - monetary / monetary.max()) * 0.3
    )
    
    # Create binary target (top 30% risk scores are high risk)
    threshold = np.percentile(risk_score, 70)
    is_high_risk = (risk_score >= threshold).astype(int)
    
    # Create DataFrame
    df = pd.DataFrame({
        'CustomerId': range(1, n_samples + 1),
        'Recency': recency,
        'Frequency': frequency,
        'Monetary': monetary,
        'is_high_risk': is_high_risk
    })
    
    return df


def main():
    """
    Main function to demonstrate the complete ML training pipeline.
    """
    print("Starting ML Training Pipeline with MLflow Tracking")
    print("=" * 60)
    
    # Create sample dataset
    print("\n1. Creating sample dataset...")
    df = create_sample_dataset(n_samples=1000, random_state=42)
    print(f"Dataset shape: {df.shape}")
    print(f"Target distribution: {df['is_high_risk'].value_counts().to_dict()}")
    
    # Initialize training pipeline
    print("\n2. Initializing ML training pipeline...")
    pipeline = MLTrainingPipeline(experiment_name="credit_risk_modeling", random_state=42)
    
    # Prepare data
    print("\n3. Preparing data...")
    X_train, X_test, y_train, y_test, feature_names = pipeline.prepare_data(
        df, target_col='is_high_risk', test_size=0.2
    )
    print(f"Training set shape: {X_train.shape}")
    print(f"Test set shape: {X_test.shape}")
    
    # Train all models
    print("\n4. Training models with hyperparameter tuning...")
    pipeline.train_all_models(X_train, X_test, y_train, y_test, feature_names)
    
    # Compare models
    print("\n5. Comparing model performance...")
    comparison_df = pipeline.compare_models()
    print("\nModel Comparison (sorted by ROC-AUC):")
    print(comparison_df.round(4))
    
    # Get best model
    print("\n6. Identifying best model...")
    best_name, best_model, best_metrics = pipeline.get_best_model()
    print(f"Best model: {best_name}")
    print(f"Best metrics: {best_metrics}")
    
    # Register best model
    print("\n7. Registering best model in MLflow...")
    model_uri = pipeline.register_best_model()
    
    print("\n" + "=" * 60)
    print("Training pipeline completed successfully!")
    print(f"Check MLflow UI at: http://localhost:5000")
    print("Run 'mlflow ui' to start the MLflow server")


if __name__ == "__main__":
    main()