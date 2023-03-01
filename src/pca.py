# exercise 2.1.3
# (requires data structures from ex. 2.2.1)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA

housing_df_raw = pd.read_csv('../Data/housing.csv')

housing_df = housing_df_raw.dropna()



housing_df_numeric = housing_df.iloc[:, 0:housing_df.shape[1] - 1]
# print(housing_df_numeric.shape)
# print(housing_df_numeric.head())

# m = entries, n = attributes
N, M = housing_df_numeric.shape

# normalize data
df_normalized=(housing_df_numeric - housing_df_numeric.mean()) / housing_df_numeric.std()


# PCA


# PCA via SVD
U,S,V = np.linalg.svd(df_normalized, full_matrices=True)

#print(U)
#print(S)
print(V)

rho = (S*S) / (S*S).sum() 

 
sns.barplot(x=list(range(1,len(rho)+1)),
            y=rho, color="limegreen")
plt.xlabel('SVs', fontsize=16)
plt.ylabel('Percent Variance Explained', fontsize=16)
plt.savefig('../Data/svd_scree_plot.png',dpi=100)
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
plt.savefig('../Data/PCA_variance_explained.png',dpi=100)
plt.show()

X_orig = housing_df_numeric.values
X = df_normalized.values
attributeNames=["Longitude", "Latitude", "Med Age", "Total Rooms", "Total Bedrooms", "Population", "Households", "Median Income", "Medican House Value"]



#############
# Simpler PCA
#############

# pca_df = pd.DataFrame(data=df_normalized)
# pca = PCA(n_components=n)
# pca.fit(pca_df)

# # print(pca.components_)
# # Reformat and view results
# loadings = pd.DataFrame(pca.components_.T,
# columns=['PC%s' % _ for _ in range(len(df_normalized.columns))],
# index=housing_df_numeric.columns)
# print(loadings)

# plt.plot(pca.explained_variance_ratio_)
# plt.ylabel('Explained Variance')
# plt.xlabel('Components')
# plt.show()
