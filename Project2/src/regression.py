import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
import os
import sklearn.linear_model as lm
from matplotlib.pylab import figure, subplot, plot, xlabel, ylabel, hist, show

DATA_PATH = '../../data'
housing_df_raw = pd.read_csv(os.path.join(DATA_PATH, 'housing.csv'))
housing_df = housing_df_raw.dropna()

housing_df_numeric = housing_df.iloc[:, :housing_df.shape[1] - 1]

# n = attributes, m = entries
N, M = housing_df_numeric.shape

# normalize data
df_normalized=(housing_df_numeric - housing_df_numeric.mean()) / housing_df_numeric.std()
X = housing_df_numeric.values

# Split dataset into features and target vector
med_inc_index = 7
y = X[:,med_inc_index]

X_cols = list(range(0,med_inc_index)) + list(range(med_inc_index+1,M))
X = X[:,X_cols]

# Fit ordinary least squares regression model
model = lm.LinearRegression()
model.fit(X,y)

# Predict alcohol content
y_est = model.predict(X)
residual = y_est-y

# Display scatter plot
figure()
subplot(2,1,1)
plot(y, y_est, '.')
xlabel('Median Income (true)'); ylabel('Median Income (estimated)');
subplot(2,1,2)
hist(residual,40)

show()

