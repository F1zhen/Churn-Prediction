"""
Customer Churn Prediction Script
This script loads the trained model and makes predictions on new data.
"""

import pandas as pd
import numpy as np
import joblib
import json
import os
import sys

def load_model_and_scaler():
    """Load the saved model, scaler, and model info"""
    try:
        model = joblib.load('../models/best_churn_model.pkl')
        scaler = joblib.load('../models/scaler.pkl')
        
        with open('../models/model_info.json', 'r') as f:
            model_info = json.load(f)
        
        print("✓ Model and scaler loaded successfully!")
        print(f"  Model: {model_info['model_name']}")
        print(f"  Training date: {model_info['training_date']}")
        print(f"  ROC-AUC Score: {model_info['performance']['ROC-AUC']:.4f}\n")
        
        return model, scaler, model_info
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)

def predict_churn(data, model, scaler, model_info, use_scaler=False):
    """
    Make churn predictions on new data
    
    Parameters:
    -----------
    data : pandas.DataFrame
        Input data with the same features as training data
    model : sklearn model
        Trained model
    scaler : sklearn.preprocessing.StandardScaler
        Fitted scaler
    model_info : dict
        Model metadata
    use_scaler : bool
        Whether to use the scaler (for models like Logistic Regression)
    
    Returns:
    --------
    predictions : numpy.array
        Binary predictions (0 = no churn, 1 = churn)
    probabilities : numpy.array
        Churn probabilities
    """
    # Validate features
    required_features = model_info['features']
    
    if not all(feature in data.columns for feature in required_features):
        missing = [f for f in required_features if f not in data.columns]
        raise ValueError(f"Missing required features: {missing}")
    
    # Select and order features correctly
    X = data[required_features]
    
    # Apply scaling if needed
    if use_scaler:
        X = scaler.transform(X)
    
    # Make predictions
    predictions = model.predict(X)
    probabilities = model.predict_proba(X)[:, 1]
    
    return predictions, probabilities

def main():
    """Main function for command-line usage"""
    # Load model
    model, scaler, model_info = load_model_and_scaler()
    
    # Example: Load processed data for demonstration
    print("Loading sample data...")
    X_sample = pd.read_csv('../data/processed/training.csv', index_col=0).head(10)
    
    # Determine if we need to use scaler
    use_scaler = 'Logistic' in model_info['model_name']
    
    # Make predictions
    predictions, probabilities = predict_churn(X_sample, model, scaler, model_info, use_scaler)
    
    # Display results
    print("\nSample Predictions:")
    print("="*70)
    results_df = pd.DataFrame({
        'Customer_Index': X_sample.index,
        'Churn_Prediction': predictions,
        'Churn_Probability': probabilities,
        'Risk_Level': ['High' if p > 0.7 else 'Medium' if p > 0.4 else 'Low' for p in probabilities]
    })
    
    print(results_df.to_string(index=False))
    print("="*70)
    print(f"\nTotal customers analyzed: {len(predictions)}")
    print(f"Predicted to churn: {predictions.sum()} ({predictions.mean()*100:.1f}%)")
    print(f"High risk customers (prob > 0.7): {sum(probabilities > 0.7)}")
    print(f"Medium risk customers (0.4 < prob <= 0.7): {sum((probabilities > 0.4) & (probabilities <= 0.7))}")
    print(f"Low risk customers (prob <= 0.4): {sum(probabilities <= 0.4)}")

if __name__ == "__main__":
    main()
