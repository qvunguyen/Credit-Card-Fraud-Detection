# Credit Card Fraud Detection

This repository contains a Python script that implements various machine learning models to detect credit card fraud based on a dataset of anonymized credit card transactions.

The dataset used in this project can be found on Kaggle: [Credit Card Fraud Detection Dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

## Overview

The script follows these steps:

1. Import libraries
2. Load the dataset
3. Preprocess the data (handle missing values, duplicates, and outliers)
4. Standardize numerical features
5. Split the data into training and testing sets
6. Handle imbalanced data using SMOTE
7. Train multiple models (Logistic Regression, Random Forest, KNN, XGBoost, and LightGBM)
8. Evaluate model performance using various metrics
9. Perform hyperparameter tuning (optional)

Based on the evaluation metrics, the Random Forest model performs the best among the selected models.

## Model Performance

The following table shows the performance of the models used in this project:

| Model              | AUPRC    | Precision (class 1) | Recall (class 1) | F1-score (class 1) | False Positives | False Negatives |
|--------------------|----------|---------------------|------------------|--------------------|-----------------|-----------------|
| Logistic Regression| 0.7322   | 0.80                | 0.64             | 0.71               | 16              | 28              |
| Random Forest      | 0.8348   | 0.91                | 0.76             | 0.83               | 7               | 23              |
| KNN                | 0.5461   | 0.24                | 0.83             | 0.37               | 209             | 13              |
| XGBoost            | 0.7943   | 0.89                | 0.68             | 0.77               | 9               | 25              |
| LightGBM           | 0.8159   | 0.88                | 0.73             | 0.80               | 10              | 21              |

