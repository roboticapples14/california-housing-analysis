#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 16 09:38:31 2023

@author: soka_1215
"""

import pandas as pd
import matplotlib.pyplot as plt

# Read the data
data = pd.read_csv("housing.csv")

# Drop rows with missing values
data = data.dropna()

# Remove rows with "ISLAND" in the "ocean_proximity" column
data = data[data['ocean_proximity'] != 'ISLAND']

# Separate features and target
X = data.drop("ocean_proximity", axis=1)
y = data["ocean_proximity"]

# Create a scatter plot using "longitude" and "latitude" as the x and y axes
plt.figure(figsize=(10, 8), dpi=150)
for category in y.unique():
    plt.scatter(X[y == category]['longitude'], X[y == category]['latitude'], label=category, alpha=0.5)
plt.title("Original Data")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.legend()
plt.show()
