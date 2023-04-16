#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 15 00:48:19 2023

@author: soka_1215
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

# Read the data
data = pd.read_csv("housing.csv")

# Drop rows with missing values
data = data.dropna()

# Separate features and target
X = data.drop("ocean_proximity", axis=1)
y = data["ocean_proximity"]

# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Set the number of folds
num_folds = 10

# Set the regularization parameter (lambda)
lambda_value = 0.5

# Create the Logistic Regression model with the specified regularization parameter
logistic_regression = LogisticRegression(C=1/lambda_value, multi_class="multinomial", solver="lbfgs", max_iter=1000, random_state=42)

# Initialize KFold cross-validation
kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)

# Initialize variables to store results
results = []

# Perform cross-validation
for i, (train_index, test_index) in enumerate(kf.split(X_scaled, y)):
    X_train, X_test = X_scaled[train_index], X_scaled[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    # Train the model
    logistic_regression.fit(X_train, y_train)

    # Make predictions
    y_pred = logistic_regression.predict(X_test)

    # Calculate error rate
    error_rate = 1 - accuracy_score(y_test, y_pred)

    # Store the results
    results.append((i + 1, lambda_value, error_rate))

# Create a DataFrame with the results
results_df = pd.DataFrame(results, columns=["Fold", "λ", "Error Rate"])

# Print the results
print(results_df)
