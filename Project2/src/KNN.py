#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 15 23:23:05 2023

@author: soka_1215
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

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

# Set the number of folds for cross-validation
n_folds = 10

# Create a KFold object for cross-validation
kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)

# Define a range of possible k values for KNN
k_values = range(1, 31)

# Initialize variables to store optimal k and error rates for each fold
optimal_k_values = []
error_rates = []

# Perform cross-validation for each fold
for train_index, test_index in kf.split(X_scaled, y):
    X_train, X_test = X_scaled[train_index], X_scaled[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    
    # Initialize variables to store best k and its corresponding error rate
    best_k = None
    best_error_rate = float('inf')
    
    # Find the optimal k value for this fold
    for k in k_values:
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train, y_train)
        y_pred = knn.predict(X_test)
        
        error_rate = 1 - accuracy_score(y_test, y_pred)
        
        if error_rate < best_error_rate:
            best_k = k
            best_error_rate = error_rate
            
    # Store the optimal k and error rate for this fold
    optimal_k_values.append(best_k)
    error_rates.append(best_error_rate)

# Display the results
for i, (optimal_k, error_rate) in enumerate(zip(optimal_k_values, error_rates), start=1):
    print(f"Fold {i}: Optimal k = {optimal_k}, Error rate = {error_rate:.2f}")
