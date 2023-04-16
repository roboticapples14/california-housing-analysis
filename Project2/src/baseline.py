#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 15 22:57:55 2023

@author: soka_1215
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, train_test_split, StratifiedKFold
from sklearn.dummy import DummyClassifier
from sklearn.preprocessing import StandardScaler

# Read the data
data = pd.read_csv("housing.csv")

# Drop rows with missing values
data = data.dropna()

# Remove rows with "ISLAND" in the "ocean_proximity" column
data = data[data['ocean_proximity'] != 'ISLAND']

# Separate features and target
X = data.drop("ocean_proximity", axis=1)
y = data["ocean_proximity"]

# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Create a baseline model using the "most frequent" strategy
baseline_model = DummyClassifier(strategy="most_frequent")

# Initialize StratifiedKFold cross-validation
stratified_kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

# Calculate the cross-validated error rate for the baseline model
error_rates = 1 - cross_val_score(baseline_model, X_scaled, y, cv=stratified_kfold, scoring='accuracy')

# Print the error rates for each fold
for i, error_rate in enumerate(error_rates, start=1):
    print(f"Fold {i}: Error rate = {error_rate:.2f}")
