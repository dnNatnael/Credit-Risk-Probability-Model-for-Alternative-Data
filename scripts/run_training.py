#!/usr/bin/env python3
"""
Script to run the complete ML training pipeline with MLflow tracking.

This script demonstrates:
1. Data preparation and feature engineering
2. Model training with hyperparameter tuning
3. MLflow experiment tracking
4. Model comparison and selection
5. Model registration in MLflow Model Registry

Usage:
    python scripts/run_training.py [--experiment-name EXPERIMENT_NAME] [--n-samples N_SAMPLES]
"""

import argparse
import sys
import os
from pathlib import Path

# Add src directory to Python path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from ml_training import MLTrainingPipeline, create_sample_dataset
import mlflow
import pandas as pd


def main():
    """Main function to run the training pipeline."""
    parser = argparse.ArgumentParser(description='Run ML training pipeline with MLflow tracking')
    parser.add_argument('--experiment-name', type=str, default='credit_risk_modeling',
                       help='Name of the MLflow experiment')
    parser.add_argument('--n-samples', type=int, default=1000,
                       help='Number of samples in the dataset')
    parser.add_argument('--test-size', type=float, default=0.2,
                       help='Proportion of data for testing')
    parser.add_argument('--random-state', type=int, default=42,
                       help='Random state for reproducibility')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("ML TRAINING PIPELINE WITH MLFLOW TRACKING")
    print("=" * 80)
    print(f"Experiment Name: {args.experiment_name}")
    print(f"Dataset Size: {args.n_samples} samples")
    print(f"Test Size: {args.test_size}")
    print(f"Random State: {args.random_state}")
    print()
    
    try:
        # Step 1: Create sample dataset
        print("📊 Step 1: Creating sample dataset...")
        df = create_sample_dataset(n_samples=args.n_samples, random_state=args.random_state)
        print(f"   ✓ Dataset created with shape: {df.shape}")
        print(f"   ✓ Target distribution: {df['is_high_risk'].value_counts().to_dict()}")
        print(f"   ✓ High-risk percentage: {df['is_high_risk'].mean():.2%}")
        print()
        
        # Step 2: Initialize training pipeline
        print("🔧 Step 2: Initializing ML training pipeline...")
        pipeline = MLTrainingPipeline(
            experiment_name=args.experiment_name, 
            random_state=args.random_state
        )
        print(f"   ✓ Pipeline initialized with experiment: {args.experiment_name}")
        print()
        
        # Step 3: Prepare data
        print("🔄 Step 3: Preparing data...")
        X_train, X_test, y_train, y_test, feature_names = pipeline.prepare_data(
            df, target_col='is_high_risk', test_size=args.test_size
        )
        print(f"   ✓ Training set shape: {X_train.shape}")
        print(f"   ✓ Test set shape: {X_test.shape}")
        print(f"   ✓ Features: {feature_names}")
        print(f"   ✓ Training target distribution: {y_train.value_counts().to_dict()}")
        print(f"   ✓ Test target distribution: {y_test.value_counts().to_dict()}")
        print()
        
        # Step 4: Train all models
        print("🤖 Step 4: Training models with hyperparameter tuning...")
        print("   This may take a few minutes...")
        pipeline.train_all_models(X_train, X_test, y_train, y_test, feature_names)
        print("   ✓ All models trained successfully!")
        print()
        
        # Step 5: Compare models
        print("📈 Step 5: Comparing model performance...")
        comparison_df = pipeline.compare_models()
        print("   Model Performance Comparison (sorted by ROC-AUC):")
        print("   " + "=" * 70)
        
        # Format the comparison table nicely
        pd.set_option('display.float_format', '{:.4f}'.format)
        print(comparison_df.to_string(index=False).replace('\n', '\n   '))
        print()
        
        # Step 6: Identify best model
        print("🏆 Step 6: Identifying best model...")
        best_name, best_model, best_metrics = pipeline.get_best_model()
        print(f"   ✓ Best model: {best_name}")
        print("   ✓ Best model metrics:")
        for metric, value in best_metrics.items():
            print(f"      {metric}: {value:.4f}")
        print()
        
        # Step 7: Register best model
        print("📝 Step 7: Registering best model in MLflow...")
        model_uri = pipeline.register_best_model()
        print(f"   ✓ Model registered successfully!")
        print(f"   ✓ Model URI: {model_uri}")
        print()
        
        # Step 8: Summary and next steps
        print("✅ TRAINING PIPELINE COMPLETED SUCCESSFULLY!")
        print("=" * 80)
        print("📋 SUMMARY:")
        print(f"   • Dataset: {args.n_samples} samples with {len(feature_names)} features")
        print(f"   • Models trained: {len(pipeline.results)}")
        print(f"   • Best model: {best_name} (ROC-AUC: {best_metrics['roc_auc']:.4f})")
        print(f"   • MLflow experiment: {args.experiment_name}")
        print()
        print("🚀 NEXT STEPS:")
        print("   1. Start MLflow UI to explore results:")
        print("      mlflow ui")
        print("   2. Open http://localhost:5000 in your browser")
        print("   3. Navigate to the experiment to compare runs")
        print("   4. Check the Model Registry for the registered model")
        print()
        print("📊 RUN TESTS:")
        print("   pytest tests/ -v")
        print()
        
    except Exception as e:
        print(f"❌ Error during training pipeline: {str(e)}")
        print("Please check the error message and try again.")
        sys.exit(1)


if __name__ == "__main__":
    main()