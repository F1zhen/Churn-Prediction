# Customer Churn Prediction Project

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0+-orange.svg)
![Status](https://img.shields.io/badge/Status-Complete-success.svg)

A comprehensive machine learning project for predicting customer churn in the banking sector using the Churn Modelling dataset.

## 📊 Project Overview

This project implements a complete machine learning pipeline to predict customer churn, helping businesses identify customers at risk of leaving and take proactive retention measures.

### Key Features

- **Data Preprocessing**: Comprehensive data cleaning, encoding, and feature engineering
- **Multiple Models**: Comparison of 5 different machine learning algorithms
- **Hyperparameter Tuning**: Optimization of the best performing model
- **Visualization**: Detailed performance metrics and feature importance analysis
- **Production Ready**: Saved model and prediction script for deployment

## 🎯 Results

### Model Performance

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|-------|----------|-----------|--------|----------|---------|
| **Gradient Boosting** | **0.8681** | **0.7835** | **0.4877** | **0.6012** | **0.8706** |
| Random Forest | 0.8641 | 0.7742 | 0.4706 | 0.5854 | 0.8506 |
| XGBoost | 0.8441 | 0.6644 | 0.4755 | 0.5543 | 0.8433 |
| Decision Tree | 0.8411 | 0.6461 | 0.4877 | 0.5559 | 0.7915 |
| Logistic Regression | 0.8061 | 0.5725 | 0.1936 | 0.2894 | 0.7703 |

**Best Model:** Gradient Boosting with **87.06% ROC-AUC** score

### Key Insights

**Top 5 Most Important Features:**
1. **Age** (39.8%) - Customer age is the strongest predictor
2. **NumOfProducts** (29.6%) - Number of products owned by customer
3. **IsActiveMember** (11.0%) - Whether customer is actively using services
4. **Balance** (8.8%) - Account balance
5. **Geography_Germany** (5.2%) - Geographic location

## 📁 Project Structure

```
.
├── data/
│   ├── raw/
│   │   └── Churn_Modelling.csv      # Original dataset
│   └── processed/
│       ├── training.csv              # Preprocessed features
│       └── test.csv                  # Target variable
├── src/
│   ├── preprocessed.ipynb            # Data preprocessing notebook
│   ├── churn_prediction_model.ipynb  # Model training notebook (detailed)
│   ├── train_models.py               # Model training script
│   └── predict.py                    # Production prediction script
├── models/
│   ├── best_churn_model.pkl         # Trained Gradient Boosting model
│   ├── scaler.pkl                    # Feature scaler
│   └── model_info.json               # Model metadata
├── results/
│   ├── model_comparison.png          # Performance comparison chart
│   ├── confusion_matrix.png          # Confusion matrix
│   ├── roc_curves.png                # ROC curves for all models
│   └── feature_importance.png        # Feature importance chart
├── requirements.txt                  # Python dependencies
└── README.md                         # This file
```

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Installation

1. Clone the repository or navigate to the project directory:
```bash
cd /path/to/customer-churn-prediction
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

### Usage

#### 1. Training the Model

To retrain the model with the existing data:

```bash
cd src
python3 train_models.py
```

This will:
- Load and split the preprocessed data
- Train 5 different models
- Compare their performance
- Save the best model
- Generate visualization plots

#### 2. Making Predictions

Use the prediction script to predict churn for new customers:

```bash
cd src
python3 predict.py
```

Example output:
```
✓ Model and scaler loaded successfully!
  Model: Gradient Boosting
  ROC-AUC Score: 0.8706

Sample Predictions:
======================================================================
 Customer_Index  Churn_Prediction  Churn_Probability Risk_Level
              0                 0           0.310409        Low
              1                 0           0.212731        Low
              2                 1           0.938870       High
...
======================================================================
```

#### 3. Custom Predictions

To make predictions on your own data:

```python
import pandas as pd
import joblib
import json

# Load model and scaler
model = joblib.load('models/best_churn_model.pkl')
scaler = joblib.load('models/scaler.pkl')

# Load your preprocessed data
# Make sure it has the same features as training data
data = pd.read_csv('your_data.csv')

# Make predictions
predictions = model.predict(data)
probabilities = model.predict_proba(data)[:, 1]

# Interpret results
for i, (pred, prob) in enumerate(zip(predictions, probabilities)):
    risk = 'High' if prob > 0.7 else 'Medium' if prob > 0.4 else 'Low'
    print(f"Customer {i}: {'Will Churn' if pred == 1 else 'Will Stay'} - "
          f"Probability: {prob:.2%} - Risk: {risk}")
```

## 📊 Dataset Information

- **Source**: Churn Modelling Dataset
- **Total Samples**: 10,002 customers
- **Features**: 12 (after preprocessing)
- **Target**: Exited (1 = Churned, 0 = Retained)
- **Churn Rate**: 20.38%
- **Train/Validation Split**: 80/20

### Features

| Feature | Type | Description |
|---------|------|-------------|
| CreditScore | Numeric | Customer's credit score |
| Age | Numeric | Customer's age |
| Tenure | Numeric | Years with the bank |
| Balance | Numeric | Account balance |
| NumOfProducts | Numeric | Number of products owned |
| HasCrCard | Binary | Has credit card (1/0) |
| IsActiveMember | Binary | Active member status (1/0) |
| EstimatedSalary | Numeric | Estimated annual salary |
| Geography_France | Binary | Location: France |
| Geography_Germany | Binary | Location: Germany |
| Geography_Spain | Binary | Location: Spain |
| Gender_encoder | Binary | Gender (encoded) |

## 📈 Visualizations

The project generates several visualizations in the `results/` folder:

1. **model_comparison.png**: Bar charts comparing all model metrics
2. **confusion_matrix.png**: Confusion matrix for the best model
3. **roc_curves.png**: ROC curves for all models
4. **feature_importance.png**: Feature importance ranking

## 🔍 Model Details

### Gradient Boosting (Best Model)

The Gradient Boosting model was selected as the best performer with:
- **87.06% ROC-AUC**: Excellent discrimination between churners and non-churners
- **86.81% Accuracy**: High overall correctness
- **78.35% Precision**: When predicting churn, correct 78% of the time
- **48.77% Recall**: Identifies 49% of actual churners

### Why Gradient Boosting?

- Best overall ROC-AUC score among all tested models
- Good balance between precision and recall
- Handles feature interactions effectively
- Robust to overfitting with proper regularization

## 💡 Business Insights

Based on the model analysis:

1. **Age is Critical**: Older customers (40+) are more likely to churn
   - *Action*: Develop retention programs for senior customers

2. **Product Portfolio Matters**: Customers with 1 or 3+ products show higher churn
   - *Action*: Focus on cross-selling to reach 2 products per customer

3. **Inactive Members**: Non-active members have significantly higher churn rates
   - *Action*: Implement engagement campaigns for inactive users

4. **Geographic Differences**: German customers show different churn patterns
   - *Action*: Tailor retention strategies by region

5. **Account Balance**: Customers with very low or very high balances may churn
   - *Action*: Monitor extreme balance changes

## 🛠️ Technologies Used

- **Python 3.13**: Programming language
- **Pandas & NumPy**: Data manipulation and analysis
- **Scikit-learn**: Machine learning algorithms and tools
- **XGBoost**: Gradient boosting implementation
- **Matplotlib & Seaborn**: Data visualization
- **Joblib**: Model serialization

## 📝 Future Improvements

- [ ] Implement SMOTE or other techniques to handle class imbalance
- [ ] Add deep learning models (Neural Networks)
- [ ] Create a web interface for predictions
- [ ] Implement real-time prediction API
- [ ] Add customer segmentation analysis
- [ ] Develop automated retraining pipeline
- [ ] Add A/B testing framework for model versions

## 📄 License

This project is available for educational and research purposes.

## 👥 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📧 Contact

For questions or feedback about this project, please open an issue in the repository.

---

**Last Updated**: October 18, 2025
**Status**: ✅ Complete and Production Ready
