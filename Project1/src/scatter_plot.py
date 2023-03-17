import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the California housing dataset from CSV file
data = pd.read_csv('../Data/housing.csv')

# Get the numeric columns of the dataset
numeric_cols = data.select_dtypes(include=np.number).columns.tolist()

sns.pairplot(data[numeric_cols], height=1.5, aspect=1, plot_kws=dict(s=5))

# Calculate statistics for each column
for i, col1 in enumerate(numeric_cols):
    for j, col2 in enumerate(numeric_cols):
        if i < j:
            plt.scatter(data[col1], data[col2])
            plt.xlabel(col1)
            plt.ylabel(col2)
            plt.title(f"{col1} vs {col2}")
            plt.show()