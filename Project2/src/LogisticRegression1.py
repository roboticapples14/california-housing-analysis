#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 15 00:37:29 2023

@author: soka_1215
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
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

# Set the regularization parameter (lambda)
lambda_value = 0.5

# Create the Logistic Regression model with the specified regularization parameter
logistic_regression = LogisticRegression(C=1/lambda_value, multi_class="multinomial", solver="lbfgs", max_iter=1000, random_state=42)

# Train the model
logistic_regression.fit(X_train, y_train)

# Make predictions
y_pred = logistic_regression.predict(X_test)

# Evaluate the model
print(classification_report(y_test, y_pred))

