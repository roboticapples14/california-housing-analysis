from cmath import nan
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import model_selection
from matplotlib.pylab import (figure, semilogx, loglog, xlabel, ylabel, legend, 
                           title, subplot, show, grid)
import torch
import torch.nn as nn
import scipy.stats as stats
from scipy.stats import t
from sklearn.decomposition import PCA
from prepData import Data
from Models import linearRegression, baseline, rlr_validate, setup_mpl
from scipy.io import loadmat
from Models import ANN

DATA_PATH = 'California_Housing_DB/Data/housing.csv'
data=Data(DATA_PATH)
#One out of k transformation
data.one_out_of_k('ocean_proximity')
#Add rooms per house and bedrooms per house  and population per household attributes
data.add_divide_col('total_rooms','households')
data.add_divide_col('total_bedrooms','households')
data.add_divide_col('population','households')

#Add households per person, total rooms per person, and total bedrooms per person
data.add_divide_col('total_rooms','population')
data.add_divide_col('total_bedrooms','population')
data.add_divide_col('households','population')


lambdas = np.power(10.,range(-1,3))
hs = np.asarray([3, 4, 5, 6, 7])
# hs = np.asarray([5])
# Create crossvalidation partition for evaluation
K = 10
lambda_K = len(lambdas)
h_K = len(hs)

CV = model_selection.KFold(K, shuffle=True)
X,y = data.get_X_y('median_income')
N, M = np.shape(X)
attributeNames=data.getAttributeNames() 
X = np.concatenate((np.ones((X.shape[0],1)),X),1)
attributeNames = [u'Offset']+attributeNames
M = M+1

# Initialize variables
annLoss = nn.MSELoss()
Error_train = np.empty((K,4))
Error_test = np.empty((K,4))
Error_train_rlr = np.empty((K*lambda_K,4))
Error_test_rlr = np.empty((K*lambda_K,4))
# Error_train_ann = np.empty((K,h_K))
# Error_test_ann = np.empty((K,h_K))
Error_train_ann = np.empty((K*h_K, 4))
Error_test_ann = np.empty((K*h_K, 4))
Error_train_nofeatures = np.empty((K,4))
Error_test_nofeatures = np.empty((K,4))
w_rlr = np.empty((M,K))
mu = np.empty((K, M-1))
sigma = np.empty((K, M-1))
w_noreg = np.empty((M,K))
k=0
for train_index, test_index in CV.split(X,y):
    print(f"CV fold {k}")
    # extract training and test set for current CV fold
    X_train = X[train_index]
    y_train = y[train_index]
    X_test = X[test_index]
    y_test = y[test_index]
    internal_cross_validation = 10    

    # opt_val_err, opt_lambda, mean_w_vs_lambda, train_err_vs_lambda, test_err_vs_lambda = rlr_validate(X_train, y_train, lambdas)

    # Standardize outer fold based on training set, and save the mean and standard
    # deviations since they're part of the model (they would be needed for
    # making new predictions) - for brevity we won't always store these in the scripts
    mu[k, :] = np.mean(X_train[:, 1:], 0)
    sigma[k, :] = np.std(X_train[:, 1:], 0)
    
    X_train[:, 1:] = (X_train[:, 1:] - mu[k, :] ) / sigma[k, :] 
    X_test[:, 1:] = (X_test[:, 1:] - mu[k, :] ) / sigma[k, :] 
    
    Xty = X_train.T @ y_train
    XtX = X_train.T @ X_train
    
    for h in range(0,len(hs)):
        hidden_units = hs[h]
        ann = ANN(M=M, n_hidden_units=hidden_units)
        # train the ANN
        best_final_loss, learning_curve = ann.fit(X_train, y_train)

        y_est = ann.predict(X_test) # forward pass, predict labels on training set
        if (y_est.shape != y.shape):
            y_est = y_est.squeeze(1)
        loss = annLoss(y_est, torch.Tensor(y_test)) # determine loss
        loss_value = loss.data.numpy() 
        print(f"Loss: {loss_value}")
        Error_test_ann[k * h_K + h] = [k+1, hidden_units, None, loss_value]


    # Compute mean squared error without using the input data at all
    # Error_train_nofeatures[k] = np.square(y_train-y_train.mean()).sum(axis=0)/y_train.shape[0]
    error = np.square(y_test-y_test.mean()).sum(axis=0)/y_test.shape[0]
    Error_test_nofeatures[k] = [k+1, None, None, error]

    # inner fold for lambda
    for l in range(0,len(lambdas)):
        lambdaI = lambdas[l] * np.eye(M)
        lambdaI[0,0] = 0 # Do no regularize the bias term
        w_rlr[:,k] = np.linalg.solve(XtX+lambdaI,Xty).squeeze()
        # Compute mean squared error with regularization with lambda
        Error_train_rlr[k] = np.square(y_train-X_train @ w_rlr[:,k]).sum(axis=0)/y_train.shape[0]
        error = np.square(y_test-X_test @ w_rlr[:,k]).sum(axis=0)/y_test.shape[0]
        Error_test_rlr[k * lambda_K + l] = [k+1, None, lambdas[l], error]
        
        print(f"loss with lambda {lambdas[l]}: {error}")

    # Estimate weights for unregularized linear regression, on entire training set
    w_noreg[:,k] = np.linalg.solve(XtX,Xty).squeeze()
    # Compute mean squared error without regularization
    Error_train[k] = np.square(y_train-X_train @ w_noreg[:,k]).sum(axis=0)/y_train.shape[0]
    error = np.square(y_test-X_test @ w_noreg[:,k]).sum(axis=0)/y_test.shape[0]
    Error_test[k] = [k+1, None, None, error]

    # Display the results for the last cross-validation fold
        
    # To inspect the used indices, use these print statements
    #print('Cross validation fold {0}/{1}:'.format(k+1,K))
    #print('Train indices: {0}'.format(train_index))
    #print('Test indices: {0}\n'.format(test_index))

    k+=1


# for i, (e1, e2) in enumerate(zip(Error_test_rlr, Error_test_nofeatures)):
#     print(f"Error for fold {i}:\nrlr: {e1[0]}, baseline: {e2[0]}")

ann_df = pd.DataFrame(Error_test_ann, 
             columns=['folds', 
                      'hidden_units',
                      'lambda',
                      'loss'])
rlr_df = pd.DataFrame(Error_test_rlr, 
             columns=['folds', 
                      'hidden_units',
                      'lambda',
                      'loss'])
base_df = pd.DataFrame(Error_test_nofeatures, 
             columns=['folds', 
                      'hidden_units',
                      'lambda',
                      'loss'])


df = pd.concat([ann_df, rlr_df, base_df], axis=0)
df.drop(0, 1)
df.sort_values('folds')
df.to_csv('cv_results.csv')
print(df)


'''
Analysis of results below
    Average errors, p-values, confidence intervals...
'''

# df = pd.read_csv('cv_results_new.csv')
# ann_e = df["ann_E"].tolist()
# rlr_e = df["rlr_E"].tolist()
# b_e = df["b_E"].tolist()
# b_e = [e for e in b_e if not math.isnan(e)]


# ann_e_ref = np.asarray(ann_e).reshape((10,5))
# rlr_e_ref = np.asarray(rlr_e).reshape((10,5))

# # average error of each parameter
# ann_total1 = 0
# ann_total2 = 0
# ann_total3 = 0
# ann_total4 = 0
# ann_total5 = 0
# rlr_total1 = 0
# rlr_total2 = 0
# rlr_total3 = 0
# rlr_total4 = 0
# rlr_total5 = 0
# for i in range(len(ann_e_ref)):
#     ann_total1 += ann_e_ref[i][0]
#     ann_total2 += ann_e_ref[i][1]
#     ann_total3 += ann_e_ref[i][2]
#     ann_total4 += ann_e_ref[i][3]
#     ann_total5 += ann_e_ref[i][4]
#     rlr_total1 += rlr_e_ref[i][0]
#     rlr_total2 += rlr_e_ref[i][1]
#     rlr_total3 += rlr_e_ref[i][2]
#     rlr_total4 += rlr_e_ref[i][3]
#     rlr_total5 += rlr_e_ref[i][4]

# ann_e_1 = ann_total1/10
# ann_e_2 = ann_total2/10
# ann_e_3 = ann_total3/10
# ann_e_4 = ann_total4/10
# ann_e_5 = ann_total5/10
# rlr_e_1 = rlr_total1/10
# rlr_e_2 = rlr_total2/10
# rlr_e_3 = rlr_total3/10
# rlr_e_4 = rlr_total4/10
# rlr_e_5 = rlr_total5/10

# # average error in every fold
# ann_e_avg = []
# rlr_e_avg = []

# ann_total = 0
# rlr_total = 0
# for i in range(len(ann_e)):
#     ann_total += ann_e[i]
#     rlr_total += rlr_e[i]
#     if ((i + 1) % 5 == 0):
#         ann_e_avg.append(ann_total/5)
#         rlr_e_avg.append(rlr_total/5)
#         ann_total = 0
#         rlr_total = 0

# print(ann_e_avg)
# print(rlr_e_avg)


# setup_mpl()

# line1, = plt.plot(ann_e_avg, 'o', label='ANN')
# line2, = plt.plot(rlr_e_avg, 'o', label='RLR')
# line3, = plt.plot(b_e, 'o', label='Baseline')
# plt.legend(handles=[line1, line2, line3])
# plt.xlabel('Fold')
# plt.ylabel('Error')
# plt.show()

# # Performing the paired sample t-test
# p_val_a_r = stats.ttest_rel(ann_e_avg, rlr_e_avg)
# p_val_a_b = stats.ttest_rel(ann_e_avg, b_e)
# p_val_r_b = stats.ttest_rel(rlr_e_avg, b_e)
# print(p_val_a_r)
# print(p_val_a_b)
# print(p_val_r_b)


# # confidence intervals
# for x in [b_e, ann_e, rlr_e]:
#     x = np.asarray(x)
#     m = x.mean() 
#     s = x.std() 
#     dof = len(x)-1 
#     confidence = 0.95
#     t_crit = np.abs(t.ppf((1-confidence)/2,dof))
#     (m-s*t_crit/np.sqrt(len(x)), m+s*t_crit/np.sqrt(len(x))) 

#     values = [np.random.choice(x,size=len(x),replace=True).mean() for i in range(1000)] 
#     print(np.percentile(values,[100*(1-confidence)/2,100*(1-(1-confidence)/2)]))
