#!/usr/bin/env python3
"""
Train a Gradient Boosted Trees model for bot detection.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import os


def load_and_prepare_data(csv_path):
    """Load features CSV and prepare for training."""
    print(f"Loading data from {csv_path}...")
    df = pd.read_csv(csv_path)
    
    print(f"Loaded {len(df)} sessions")
    print(f"Columns: {len(df.columns)}")
    
    # Check if label column exists
    if 'label' not in df.columns:
        raise ValueError("No 'label' column found in CSV. Please run extract_features.py first.")
    
    # Separate features and labels
    # Exclude metadata columns
    exclude_cols = ['user_id', 'session_id', 'source_file', 'label']
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    X = df[feature_cols].fillna(0).replace([np.inf, -np.inf], 0)
    y = df['label']
    
    # Check label distribution
    print(f"\nLabel distribution:")
    print(y.value_counts().sort_index())
    print(f"  Human (1): {(y == 1).sum()} sessions")
    print(f"  Bot (0): {(y == 0).sum()} sessions")
    
    # Check if we have both classes
    if len(y.unique()) < 2:
        print("\nWARNING: Only one class found in data! Cannot train a binary classifier.")
        print("Please ensure you have both human and bot data.")
        return None, None, None, None
    
    return X, y, feature_cols, df


def train_gradient_boosting(X, y, test_size=0.2, random_state=42):
    """Train a Gradient Boosting Classifier."""
    print(f"\nSplitting data: {test_size*100:.0f}% test, {(1-test_size)*100:.0f}% train")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    print(f"Training set: {len(X_train)} sessions")
    print(f"Test set: {len(X_test)} sessions")
    
    # Initialize Gradient Boosting Classifier
    print("\nTraining Gradient Boosting Classifier...")
    gb_model = GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        random_state=random_state,
        verbose=1
    )
    
    # Train the model
    gb_model.fit(X_train, y_train)
    
    # Make predictions
    y_train_pred = gb_model.predict(X_train)
    y_test_pred = gb_model.predict(X_test)
    
    # Calculate accuracies
    train_accuracy = accuracy_score(y_train, y_train_pred)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    
    print(f"\n{'='*60}")
    print(f"Training Accuracy: {train_accuracy*100:.2f}%")
    print(f"Test Accuracy: {test_accuracy*100:.2f}%")
    print(f"{'='*60}")
    
    # Detailed classification report
    print("\nClassification Report (Test Set):")
    print(classification_report(y_test, y_test_pred, target_names=['Bot', 'Human']))
    
    # Confusion matrix
    print("\nConfusion Matrix (Test Set):")
    cm = confusion_matrix(y_test, y_test_pred)
    print(f"                Predicted")
    print(f"              Bot  Human")
    print(f"Actual Bot    {cm[0,0]:4d}  {cm[0,1]:4d}")
    print(f"       Human   {cm[1,0]:4d}  {cm[1,1]:4d}")
    
    # Feature importance
    print("\nTop 10 Most Important Features:")
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': gb_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    for idx, row in feature_importance.head(10).iterrows():
        print(f"  {row['feature']:40s} {row['importance']:.4f}")
    
    return gb_model, X_test, y_test, y_test_pred


def main():
    """Main training function."""
    csv_path = '/home/benutzer1/Thesis/CLASSIFIER/features.csv'
    model_path = '/home/benutzer1/Thesis/CLASSIFIER/bot_detection_model.pkl'
    
    # Check if features CSV exists
    if not os.path.exists(csv_path):
        print(f"Features CSV not found at {csv_path}")
        print("Please run extract_features.py first to generate features.")
        return
    
    # Load and prepare data
    result = load_and_prepare_data(csv_path)
    if result is None:
        return
    
    X, y, feature_cols, df = result
    
    # Train model
    model, X_test, y_test, y_pred = train_gradient_boosting(X, y)
    
    if model is None:
        return
    
    # Save the model
    print(f"\nSaving model to {model_path}...")
    joblib.dump(model, model_path)
    print("Model saved successfully!")
    
    # Print final summary
    test_accuracy = accuracy_score(y_test, y_pred)
    print(f"\n{'='*60}")
    print(f"FINAL TEST ACCURACY: {test_accuracy*100:.2f}%")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()


