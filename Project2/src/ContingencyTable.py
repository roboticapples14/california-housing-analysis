#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 17 10:20:29 2023

@author: soka_1215
"""
# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
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

# Split the data into training and testing sets (keeping the unscaled version of the test set)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
X_test_unscaled = X.loc[y_test.index]

# Train the baseline model
baseline_model = DummyClassifier(strategy="most_frequent")
baseline_model.fit(X_train, y_train)

# Make predictions with the baseline model
y_pred_baseline = baseline_model.predict(X_test)

# Train the KNN model
knn_model = KNeighborsClassifier(n_neighbors=11)  # Replace '5' with the optimal number of neighbors
knn_model.fit(X_train, y_train)

# Make predictions with the KNN model
y_pred_knn = knn_model.predict(X_test)

# Train the logistic regression model
logistic_regression_model = LogisticRegression(multi_class="multinomial", solver="saga", max_iter=5000, random_state=42)
logistic_regression_model.fit(X_train, y_train)

# Make predictions with the logistic regression model
y_pred_logistic_regression = logistic_regression_model.predict(X_test)

# Initialize the contingency table
contingency_table = {
    ('baseline', 'knn'): {'a': 0, 'b': 0, 'c': 0, 'd': 0},
    ('baseline', 'logistic_regression'): {'a': 0, 'b': 0, 'c': 0, 'd': 0},
    ('knn', 'logistic_regression'): {'a': 0, 'b': 0, 'c': 0, 'd': 0}
}

# Calculate the values for the contingency table
for baseline_pred, knn_pred, logistic_pred, true_label in zip(y_pred_baseline, y_pred_knn, y_pred_logistic_regression, y_test):
    baseline_correct = baseline_pred == true_label
    knn_correct = knn_pred == true_label
    logistic_correct = logistic_pred == true_label

    if baseline_correct and knn_correct:
        contingency_table[('baseline', 'knn')]['a'] += 1
    elif baseline_correct and not knn_correct:
        contingency_table[('baseline', 'knn')]['b'] += 1
    elif not baseline_correct and knn_correct:
        contingency_table[('baseline', 'knn')]['c'] += 1
    else:
        contingency_table[('baseline', 'knn')]['d'] += 1

    if baseline_correct and logistic_correct:
        contingency_table[('baseline', 'logistic_regression')]['a'] += 1
    elif baseline_correct and not logistic_correct:
        contingency_table[('baseline', 'logistic_regression')]['b'] += 1
    elif not baseline_correct and logistic_correct:
        contingency_table[('baseline', 'logistic_regression')]['c'] += 1
    else:
        contingency_table[('baseline', 'logistic_regression')]['d'] += 1

    if knn_correct and logistic_correct:
        contingency_table[('knn', 'logistic_regression')]['a'] += 1
    elif knn_correct and not logistic_correct:
        contingency_table[('knn', 'logistic_regression')]['b'] += 1
    elif not knn_correct and logistic_correct:
        contingency_table[('knn', 'logistic_regression')]['c'] += 1
    else:
        contingency_table[('knn', 'logistic_regression')]['d'] += 1

print("2x2 contingency tables for each pair of models:")
print("Baseline vs. KNN:", contingency_table[('baseline', 'knn')])
print("Baseline vs. Logistic Regression:", contingency_table[('baseline', 'logistic_regression')])
print("KNN vs. Logistic Regression:", contingency_table[('knn', 'logistic_regression')])



