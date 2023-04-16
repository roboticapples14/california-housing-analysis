#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 15 23:30:04 2023

@author: soka_1215
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Read the data
data = pd.read_csv("housing.csv")

# Drop rows with missing values and "ISLAND" rows
data = data.dropna()
data = data[data['ocean_proximity'] != 'ISLAND']

# Separate features and target
X = data.drop("ocean_proximity", axis=1)
y = data["ocean_proximity"]

# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
X_test_unscaled = X.loc[y_test.index]


# Create the KNN model with the optimal k found in fold 4
knn = KNeighborsClassifier(n_neighbors=7)

# Train the model
knn.fit(X_train, y_train)

# Make predictions
y_pred = knn.predict(X_test)

label_to_color = {
    '<1H OCEAN': 0,
    'INLAND': 1,
    'NEAR OCEAN': 2,
    'NEAR BAY': 3,
}
y_pred_numeric = np.array([label_to_color[label] for label in y_pred])

plt.figure(figsize=(12, 8))

# Create a new figure with a custom size (10x8 inches) and resolution (150 dpi)
plt.figure(figsize=(10, 8), dpi=150)

# Loop through each unique category in the target variable 'y'
for category in y.unique():
    # For each category, plot the points in X_test_unscaled where the predicted category is equal to the current category
    plt.scatter(X_test_unscaled[y_pred == category]['longitude'],
                X_test_unscaled[y_pred == category]['latitude'],
                label=category,  # Set the label for the current category
                alpha=0.5)  # Set the transparency of the plotted points to 0.5
plt.title("KNN")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.legend()
plt.show()




