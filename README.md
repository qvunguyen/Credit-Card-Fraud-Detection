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

## Usage

To run the script, you will need JupyterNotebook and the following libraries installed:

1. numpy
2. pandas
3. scikit-learn
4. imbalanced-learn
5. scipy
6. xgboost
7. lightgbm

Download the dataset from the source and place it in the same directory as the script. Then, simply run the script using Jupyternotebook.


## Conclusion

Based on the evaluation metrics, the Random Forest model performs the best among the selected models for this imbalanced dataset. The reasoning behind this conclusion is as follows:

1. AUPRC (Area Under the Precision-Recall Curve): Random Forest has the highest AUPRC (0.8348) among all models, indicating better overall performance in distinguishing between the classes when dealing with imbalanced data.
2. Precision, Recall, and F1-score: Random Forest has the highest precision (0.91) for the positive class (fraud), which means it has the lowest false positive rate among the models. It also has a good recall (0.76), which means it can detect a considerable proportion of the actual fraud cases. The F1-score (0.83) for the positive class in Random Forest is also the highest, indicating a good balance between precision and recall.
3. Confusion Matrix: The confusion matrix of the Random Forest model shows the smallest number of false positives (7) and a relatively low number of false negatives (23) compared to other models.

Although the accuracy is high for all the models, it is not a reliable metric in this case due to the highly imbalanced nature of the dataset. The other metrics mentioned above provide a better perspective on the model's performance.

Considering all these factors, the Random Forest model seems to be the best performer among the selected models for this imbalanced dataset.

