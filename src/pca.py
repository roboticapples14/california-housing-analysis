import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
import os
DATA_PATH = '/Users/nvalett/Documents/Natalie/DTU/ML/California_Housing_DB/Data'
housing_df_raw = pd.read_csv(os.path.join(DATA_PATH, 'housing.csv'))

print(len(housing_df_raw))
housing_df = housing_df_raw.dropna()
print(len(housing_df))



housing_df_numeric = housing_df.iloc[:, 2:housing_df.shape[1] - 1]
# print(housing_df_numeric.shape)
# print(housing_df_numeric.head())

# m = entries, n = attributes
m, n = housing_df_numeric.shape

# normalize data
df_normalized=(housing_df_numeric - housing_df_numeric.mean()) / housing_df_numeric.std()

# PCA

# PCA via SVD
U,S,V = np.linalg.svd(df_normalized, full_matrices=True)

print(U)
print(S)
print(V)

rho = (S*S) / (S*S).sum() 

# rho = variablility explained
variance = {attr: var for attr, var in zip(df_normalized.columns, rho)}
print(variance)
 
sns.barplot(x=list(range(1,len(rho)+1)),
            y=rho, color="limegreen")
plt.xlabel('SVs', fontsize=16)
plt.ylabel('Percent Variance Explained', fontsize=16)
plt.savefig(os.path.join(DATA_PATH, 'svd_scree_plot.png'),dpi=100)
plt.show()

# # Compute variance explained by principal components

threshold = 0.9

# Plot variance explained
plt.figure()
plt.plot(range(1,len(rho)+1),rho,'x-')
plt.plot(range(1,len(rho)+1),np.cumsum(rho),'o-')
plt.plot([1,len(rho)],[threshold, threshold],'k--')
plt.title('Variance explained by principal components');
plt.xlabel('Principal component');
plt.ylabel('Variance explained');
plt.legend(['Individual','Cumulative','Threshold'])
plt.grid()
plt.savefig(os.path.join(DATA_PATH, 'PCA_variance_explained.png'),dpi=100)
plt.show()
