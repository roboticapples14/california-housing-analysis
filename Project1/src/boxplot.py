import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the California housing dataset from CSV file
data = pd.read_csv('../Data/housing.csv')

# Get the numeric columns of the dataset
numeric_cols = data.select_dtypes(include=np.number).columns.tolist()

# Create boxplots for each numeric attribute
for col in numeric_cols:
    sns.boxplot(x=data[col])
    plt.show()
