import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load the California housing dataset from CSV file
data = pd.read_csv('../Data/housing.csv')

# Get the numeric columns of the dataset
numeric_cols = data.select_dtypes(include=np.number).columns.tolist()

# Create histograms for each numeric column
for col in numeric_cols:
    plt.hist(data[col], bins=30)
    plt.xlabel(col)
    plt.ylabel('Frequency')
    plt.show()
