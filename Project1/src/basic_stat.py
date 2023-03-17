import numpy as np
import pandas as pd
import seaborn as sns

# Load the California housing dataset from CSV file
data = pd.read_csv('../Data/housing.csv')

# Get the numeric columns of the dataset
numeric_cols = data.select_dtypes(include=np.number).columns.tolist()

# Calculate statistics for each column
for col in numeric_cols:
    values = data[col].values
    mean_x = values.mean()
    std_x = values.std(ddof=1)
    median_x = np.median(values)
    range_x = values.max() - values.min()

    # Display results
    print('Column:', col)
    print('Mean:', mean_x)
    print('Standard Deviation:', std_x)
    print('Median:', median_x)
    print('Range:', range_x)
    print('\n')
    
# Calculate correlation matrix
corr_matrix = data[numeric_cols].corr()

# Plot heatmap of correlation matrix
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')

# Checking for missing data
print(data.isnull().sum())