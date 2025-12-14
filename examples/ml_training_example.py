#!/usr/bin/env python3
"""
Example demonstrating the ML training pipeline with MLflow tracking.

This example shows how to:
1. Create a sample dataset
2. Initialize the ML training pipeline
3. Train multiple models with hyperparameter tuning
4. Compare model performance
5. Register the best model in MLflow

Run this example:
    python examples/ml_training_example.py
"""

import sys
from pathlib import Path

# Add src directory to Python path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from ml_training import MLTrainingPipeline, create_sample_dataset
import pandas as pd
import numpy as np


def basic_training_example():
    """Basic example of the ML training pipeline."""
    print("🚀 Basic ML Training Pipeline Example")
    print("=" * 50)
    
    # Step 1: Create sample dataset
    print("\n1. Creating sample dataset...")
    df = create_sample_dataset(n_samples=500, random_state=42)
    print(f"   Dataset shape: {df.shape}")
    print(f"   Features: {list(df.columns)}")
    print(f"   Target distribution:")
    print(f"   {df['is_high_risk'].value_counts().to_dict()}")
    
    # Step 2: Initialize pipeline
    print("\n2. Initializing ML pipeline...")
    pipeline = MLTrainingPipeline(
        experiment_name="basic_example",
        random_state=42
    )
    
    # Step 3: Prepare data
    print("\n3. Preparing data...")
    X_train, X_test, y_train, y_test, feature_names = pipeline.prepare_data(
        df, target_col='is_high_risk', test_size=0.2
    )
    print(f"   Training set: {X_train.shape}")
    print(f"   Test set: {X_test.shape}")
    print(f"   Features: {feature_names}")
    
    # Step 4: Train models (subset for faster example)
    print("\n4. Training models...")
    
    # Train just two models for this example
    models_to_train = ['logistic_regression', 'random_forest']
    
    for model_name in models_to_train:
        print(f"   Training {model_name}...")
        best_model, metrics, best_params = pipeline.train_model_with_tuning(
            model_name, X_train, y_train, X_test, y_test, cv_folds=3
        )
        
        # Store results manually for this example
        pipeline.best_models[model_name] = best_model
        pipeline.results[model_name] = {
            'metrics': metrics,
            'best_params': best_params
        }
        
        print(f"   ✓ {model_name} completed")
        print(f"     Best params: {best_params}")
        print(f"     ROC-AUC: {metrics['roc_auc']:.4f}")
    
    # Step 5: Compare models
    print("\n5. Comparing models...")
    comparison_df = pipeline.compare_models()
    print(comparison_df.round(4))
    
    # Step 6: Get best model
    print("\n6. Best model selection...")
    best_name, best_model, best_metrics = pipeline.get_best_model()
    print(f"   Best model: {best_name}")
    print(f"   ROC-AUC: {best_metrics['roc_auc']:.4f}")
    print(f"   Accuracy: {best_metrics['accuracy']:.4f}")
    
    return pipeline, best_model, best_metrics


def custom_features_example():
    """Example using custom features."""
    print("\n\n🔧 Custom Features Example")
    print("=" * 50)
    
    # Create dataset with additional features
    df = create_sample_dataset(n_samples=300, random_state=42)
    
    # Add some engineered features
    df['Recency_log'] = np.log1p(df['Recency'])
    df['Frequency_squared'] = df['Frequency'] ** 2
    df['Monetary_per_transaction'] = df['Monetary'] / df['Frequency']
    
    print(f"Dataset with engineered features: {df.shape}")
    
    # Use custom feature set
    custom_features = ['Recency', 'Frequency', 'Monetary', 'Recency_log', 'Monetary_per_transaction']
    
    pipeline = MLTrainingPipeline(experiment_name="custom_features_example")
    
    X_train, X_test, y_train, y_test, feature_names = pipeline.prepare_data(
        df, 
        target_col='is_high_risk',
        feature_cols=custom_features,
        test_size=0.25
    )
    
    print(f"Using features: {feature_names}")
    
    # Train a single model for demonstration
    best_model, metrics, best_params = pipeline.train_model_with_tuning(
        'gradient_boosting', X_train, y_train, X_test, y_test, cv_folds=3
    )
    
    print(f"Gradient Boosting Results:")
    print(f"  Best params: {best_params}")
    print(f"  Metrics: {metrics}")
    
    # Show feature importance
    if hasattr(best_model, 'feature_importances_'):
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': best_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print(f"\nFeature Importance:")
        print(importance_df.round(4))
    
    return pipeline, best_model


def model_evaluation_example():
    """Example focusing on detailed model evaluation."""
    print("\n\n📊 Model Evaluation Example")
    print("=" * 50)
    
    # Create a larger dataset for more stable metrics
    df = create_sample_dataset(n_samples=800, random_state=42)
    
    pipeline = MLTrainingPipeline(experiment_name="evaluation_example")
    X_train, X_test, y_train, y_test, feature_names = pipeline.prepare_data(df)
    
    # Train decision tree for interpretability
    print("Training Decision Tree for detailed evaluation...")
    
    best_model, metrics, best_params = pipeline.train_model_with_tuning(
        'decision_tree', X_train, y_train, X_test, y_test, cv_folds=5
    )
    
    print(f"\nDecision Tree Results:")
    print(f"Best parameters: {best_params}")
    print(f"\nDetailed Metrics:")
    for metric, value in metrics.items():
        print(f"  {metric:12}: {value:.4f}")
    
    # Additional evaluation
    from sklearn.metrics import confusion_matrix, classification_report
    
    y_pred = best_model.predict(X_test)
    y_pred_proba = best_model.predict_proba(X_test)[:, 1]
    
    print(f"\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(f"  [[{cm[0,0]:3d}, {cm[0,1]:3d}]")
    print(f"   [{cm[1,0]:3d}, {cm[1,1]:3d}]]")
    
    print(f"\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Low Risk', 'High Risk']))
    
    # Feature importance for decision tree
    if hasattr(best_model, 'feature_importances_'):
        print(f"\nFeature Importance:")
        for feature, importance in zip(feature_names, best_model.feature_importances_):
            print(f"  {feature:12}: {importance:.4f}")
    
    return best_model, metrics


def main():
    """Run all examples."""
    print("🎯 ML Training Pipeline Examples")
    print("=" * 60)
    
    try:
        # Run basic example
        pipeline1, model1, metrics1 = basic_training_example()
        
        # Run custom features example
        pipeline2, model2 = custom_features_example()
        
        # Run evaluation example
        model3, metrics3 = model_evaluation_example()
        
        print("\n\n✅ All examples completed successfully!")
        print("=" * 60)
        print("🚀 Next Steps:")
        print("1. Start MLflow UI: mlflow ui")
        print("2. Open http://localhost:5000")
        print("3. Explore the experiments: basic_example, custom_features_example, evaluation_example")
        print("4. Run the full pipeline: python scripts/run_training.py")
        print("5. Run tests: pytest tests/ -v")
        
    except Exception as e:
        print(f"\n❌ Error in examples: {str(e)}")
        print("Make sure all dependencies are installed: pip install -r requirements.txt")


if __name__ == "__main__":
    main()