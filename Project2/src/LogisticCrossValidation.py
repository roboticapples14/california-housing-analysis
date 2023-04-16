#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 15 01:17:47 2023

@author: soka_1215
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV, train_test_split, StratifiedKFold
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

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Create a Logistic Regression model
logistic_regression = LogisticRegression(multi_class="multinomial", solver="saga", max_iter=500, random_state=42)

# Specify the range of lambda values to search
lambdas = np.logspace(-5, 5, 11)  # Example: 11 values from 10^-5 to 10^5

# Create the parameter grid for GridSearchCV
param_grid = {'C': 1 / lambdas}  # Note: C is the inverse of lambda

# Initialize StratifiedKFold cross-validation
stratified_kfold = StratifiedKFold(n_splits=2, shuffle=True, random_state=42)

# Initialize GridSearchCV with the model, parameter grid, and cross-validation settings
grid_search = GridSearchCV(logistic_regression, param_grid, cv=stratified_kfold, scoring='accuracy', n_jobs=-1)

# Fit the grid search to the training data
grid_search.fit(X_train, y_train)

# Get the best lambda value
best_lambda = 1 / grid_search.best_params_['C']
print(f"Best lambda value: {best_lambda}")

# Get the best model
best_model = grid_search.best_estimator_

# Make predictions using the best model
y_pred = best_model.predict(X_test)

# Calculate the error rate for the best model
error_rate = 1 - accuracy_score(y_test, y_pred)
print(f"Error rate: {error_rate}")


