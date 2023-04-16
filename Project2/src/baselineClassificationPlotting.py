#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 15 23:15:19 2023

@author: soka_1215
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier

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

# Train the baseline classifier
baseline_classifier = DummyClassifier(strategy="stratified", random_state=42)
baseline_classifier.fit(X_train, y_train)

# Make predictions
y_pred_baseline = baseline_classifier.predict(X_test)

label_to_color = {
    '<1H OCEAN': 0,
    'INLAND': 1,
    'NEAR OCEAN': 2,
    'NEAR BAY': 3,
}
y_pred_numeric = np.array([label_to_color[label] for label in y_pred_baseline])

# Create a scatter plot using "longitude" and "latitude" as the x and y axes, colored by predicted labels from the baseline classifier
plt.figure(figsize=(10, 8), dpi=150)
for category in y.unique():
    plt.scatter(X_test_unscaled[y_pred_baseline == category]['longitude'], X_test_unscaled[y_pred_baseline == category]['latitude'], label=category, alpha=0.5)
plt.title("Baseline")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.legend()
plt.show()
