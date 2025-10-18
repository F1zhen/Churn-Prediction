"""
Customer Churn Prediction - Model Training Script
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Machine Learning Libraries
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve
)

# Models
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier

import joblib
import os
import json

def main():
    # Set style
    sns.set_style('whitegrid')
    plt.rcParams['figure.figsize'] = (12, 6)
    
    print('Libraries imported successfully!')
    print('=' * 70)
    
    # Load data
    print('\n1. Loading preprocessed data...')
    X = pd.read_csv('../data/processed/training.csv', index_col=0)
    y = pd.read_csv('../data/processed/test.csv', index_col=0)['Exited']
    
    print(f'Features shape: {X.shape}')
    print(f'Target shape: {y.shape}')
    print(f'Churn rate: {y.mean()*100:.2f}%')
    
    # Split data
    print('\n2. Splitting data into train and validation sets...')
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f'Training set size: {X_train.shape[0]}')
    print(f'Validation set size: {X_val.shape[0]}')
    
    # Scale features
    print('\n3. Scaling features...')
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    print('Features scaled successfully!')
    
    # Train models
    print('\n4. Training multiple models...')
    print('=' * 70)
    models = {
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'Decision Tree': DecisionTreeClassifier(random_state=42, max_depth=10),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
        'XGBoost': XGBClassifier(n_estimators=100, random_state=42, eval_metric='logloss', verbosity=0)
    }
    
    results = {}
    trained_models = {}
    
    for name, model in models.items():
        print(f'Training {name}...')
        
        if 'Logistic' in name:
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_val_scaled)
            y_pred_proba = model.predict_proba(X_val_scaled)[:, 1]
        else:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_val)
            y_pred_proba = model.predict_proba(X_val)[:, 1]
        
        accuracy = accuracy_score(y_val, y_pred)
        precision = precision_score(y_val, y_pred)
        recall = recall_score(y_val, y_pred)
        f1 = f1_score(y_val, y_pred)
        roc_auc = roc_auc_score(y_val, y_pred_proba)
        
        results[name] = {
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1-Score': f1,
            'ROC-AUC': roc_auc
        }
        
        trained_models[name] = {
            'model': model,
            'predictions': y_pred,
            'probabilities': y_pred_proba
        }
        
        print(f'  Accuracy: {accuracy:.4f}, ROC-AUC: {roc_auc:.4f}')
    
    print('\n5. Model comparison...')
    print('=' * 70)
    results_df = pd.DataFrame(results).T.round(4).sort_values('ROC-AUC', ascending=False)
    print(results_df)
    print(f'\nBest model: {results_df.index[0]}')
    
    # Save best model
    print('\n6. Saving best model...')
    best_model_name = results_df.index[0]
    best_model = trained_models[best_model_name]['model']
    
    os.makedirs('../models', exist_ok=True)
    os.makedirs('../results', exist_ok=True)
    
    joblib.dump(best_model, '../models/best_churn_model.pkl')
    joblib.dump(scaler, '../models/scaler.pkl')
    
    model_info = {
        'model_name': best_model_name,
        'features': X.columns.tolist(),
        'performance': results[best_model_name],
        'training_date': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    with open('../models/model_info.json', 'w') as f:
        json.dump(model_info, f, indent=4)
    
    print('✓ Model saved successfully!')
    
    # Create visualizations
    print('\n7. Creating visualizations...')
    
    # Model comparison
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    results_df.plot(kind='bar', ax=axes[0])
    axes[0].set_title('Model Performance Comparison - All Metrics', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Model', fontsize=12)
    axes[0].set_ylabel('Score', fontsize=12)
    axes[0].legend(loc='lower right')
    axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=45, ha='right')
    axes[0].set_ylim([0, 1])
    axes[0].grid(True, alpha=0.3)
    
    results_df['ROC-AUC'].plot(kind='barh', ax=axes[1], color='steelblue')
    axes[1].set_title('ROC-AUC Score Comparison', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('ROC-AUC Score', fontsize=12)
    axes[1].set_ylabel('Model', fontsize=12)
    axes[1].set_xlim([0.7, 0.9])
    axes[1].grid(True, alpha=0.3)
    
    for i, v in enumerate(results_df['ROC-AUC']):
        axes[1].text(v, i, f' {v:.4f}', va='center')
    
    plt.tight_layout()
    plt.savefig('../results/model_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Confusion matrix
    best_predictions = trained_models[best_model_name]['predictions']
    cm = confusion_matrix(y_val, best_predictions)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True,
                xticklabels=['Not Churned', 'Churned'],
                yticklabels=['Not Churned', 'Churned'])
    plt.title(f'Confusion Matrix - {best_model_name}', fontsize=14, fontweight='bold')
    plt.ylabel('Actual', fontsize=12)
    plt.xlabel('Predicted', fontsize=12)
    plt.savefig('../results/confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # ROC curves
    plt.figure(figsize=(10, 8))
    for name in models.keys():
        y_proba = trained_models[name]['probabilities']
        fpr, tpr, _ = roc_curve(y_val, y_proba)
        auc_score = roc_auc_score(y_val, y_proba)
        plt.plot(fpr, tpr, label=f'{name} (AUC = {auc_score:.4f})', linewidth=2)
    
    plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier', linewidth=2)
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curves - All Models', fontsize=14, fontweight='bold')
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.savefig('../results/roc_curves.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Feature importance
    if hasattr(best_model, 'feature_importances_'):
        feature_importance = pd.DataFrame({
            'Feature': X.columns,
            'Importance': best_model.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        print('\nTop 10 Most Important Features:')
        print('=' * 50)
        print(feature_importance.head(10).to_string(index=False))
        
        plt.figure(figsize=(10, 8))
        plt.barh(range(len(feature_importance)), feature_importance['Importance'])
        plt.yticks(range(len(feature_importance)), feature_importance['Feature'])
        plt.xlabel('Importance', fontsize=12)
        plt.ylabel('Feature', fontsize=12)
        plt.title(f'Feature Importance - {best_model_name}', fontsize=14, fontweight='bold')
        plt.gca().invert_yaxis()
        plt.grid(True, alpha=0.3, axis='x')
        plt.tight_layout()
        plt.savefig('../results/feature_importance.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    print('✓ All visualizations saved!')
    
    # Final report
    print('\n' + '=' * 70)
    print('CUSTOMER CHURN PREDICTION - FINAL REPORT')
    print('=' * 70)
    print(f'\nDataset Information:')
    print(f'  - Total samples: {len(X)}')
    print(f'  - Number of features: {X.shape[1]}')
    print(f'  - Churn rate: {y.mean()*100:.2f}%')
    print(f'\nData Split:')
    print(f'  - Training samples: {len(X_train)} ({len(X_train)/len(X)*100:.1f}%)')
    print(f'  - Validation samples: {len(X_val)} ({len(X_val)/len(X)*100:.1f}%)')
    print(f'\nModels Evaluated: {len(models)}')
    print(f'\nBest Model: {best_model_name}')
    print(f'\nPerformance Metrics (Validation Set):')
    for metric, value in results[best_model_name].items():
        print(f'  - {metric}: {value:.4f}')
    print(f'\nFiles Generated:')
    print(f'  - Model: models/best_churn_model.pkl')
    print(f'  - Scaler: models/scaler.pkl')
    print(f'  - Model info: models/model_info.json')
    print(f'  - Visualizations: results/')
    print('=' * 70)
    print('\n✓ Customer Churn Prediction Project Completed Successfully!')

if __name__ == '__main__':
    main()
