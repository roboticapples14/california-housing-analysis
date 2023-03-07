
from pca import *

import matplotlib.pyplot as plt
from scipy.linalg import svd
import seaborn as sn

X_mean= X.mean(0)
Y = df_normalized.values
N,M = X.shape

# We saw in 2.1.3 that the first 3 components explaiend more than 90
# percent of the variance. Let's look at their coefficients:
pcs = [0,1,2]

legendStrs = ['PC'+str(e+1) for e in pcs]
c = ['r','g','b']
bw = .2
r = np.arange(1,M+1)
Vh = V.T
for i in pcs:    
    plt.bar(r+i*bw, Vh[:,i], width=bw)

print("X: ", X)
print(r)
print(bw)
plt.xticks(r+bw, attributeNames)
plt.xlabel('Attributes')
plt.ylabel('Component coefficients')
plt.legend(legendStrs)
plt.grid()
plt.title('Housing: PCA Component Coefficients')
plt.xticks(fontsize =7)
plt.xticks(rotation = 45)

plt.show()


Z = Y @ Vh
i = 0
j = 1
plt.scatter(Z[:,i], Z[:,j])
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('Normalized PCA Plot')
plt.show()


i = 2
j = 3
plt.scatter(Z[:,i], Z[:,j])
plt.xlabel('PC3')
plt.ylabel('PC4')
plt.title('Normalized PCA Plot')
plt.show()



