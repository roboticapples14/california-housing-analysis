import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
from scipy import stats
from sklearn import model_selection
from prepData import Data
from Models import ANN
from visualize import visualize_decision_boundary, draw_neural_net

DATA_PATH = 'California_Housing_DB/Data/housing.csv'
data = Data(DATA_PATH)
data.remove_col(9)

y_colname = 'median_income'
classNames = [y_colname]
X, y = data.get_X_y(y_colname)

# Normalize data
X = stats.zscore(X)

ann = ANN(M=data.size[1]-1, n_hidden_units=4)
max_iter = 10000
attributeNames = data.getAttributeNames()

# K-fold CrossValidation (4 folds here to speed up this example)
K = 4
hs = np.asarray([3, 4, 5, 6])
h_K = len(hs)
CV = model_selection.KFold(K,shuffle=True)
N, M = np.shape(X)
annLoss = nn.MSELoss()

# Setup figure for display of the decision boundary for the several crossvalidation folds.
decision_boundaries = plt.figure(1, figsize=(10,10))
# Determine a size of a plot grid that fits visualizations for the chosen number
# of cross-validation splits, if K=4, this is simply a 2-by-2 grid.
subplot_size_1 = int(np.floor(np.sqrt(K))) 
subplot_size_2 = int(np.ceil(K/subplot_size_1))
# Set overall title for all of the subplots
plt.suptitle('Data and model decision boundaries', fontsize=20)
# Change spacing of subplots
plt.subplots_adjust(left=0, bottom=0, right=1, top=.9, wspace=.5, hspace=0.25)

# Setup figure for display of learning curves and error rates in fold
summaries, summaries_axes = plt.subplots(1, 2, figsize=(10,5))
# Make a list for storing assigned color of learning curve for up to K=10
color_list = ['tab:orange', 'tab:green', 'tab:purple', 'tab:brown', 'tab:pink',
              'tab:gray', 'tab:olive', 'tab:cyan', 'tab:red', 'tab:blue']


# Do cross-validation:
# make a list for storing generalizaition error in each loop
Error_train_ann = np.empty((K*h_K, 3))
Error_test_ann = np.empty((K*h_K, 3))
# Loop over each cross-validation split. The CV.split-method returns the 
# indices to be used for training and testing in each split, and calling 
# the enumerate-method with this simply returns this indices along with 
# a counter k:
for k, (train_index, test_index) in enumerate(CV.split(X,y)): 
    print('\nCrossvalidation fold: {0}/{1}'.format(k+1,K))    
    # Extract training and test set for current CV fold, 
    # and convert them to PyTorch tensors
    X_train = torch.Tensor(X[train_index,:] )
    y_train = torch.Tensor(y[train_index] )
    X_test = torch.Tensor(X[test_index,:] )
    y_test = torch.Tensor(y[test_index] )

    for h in range(0,len(hs)):
        hidden_units = hs[h]
        ann = ANN(M=M, n_hidden_units=hidden_units)
        # train the ANN
        best_final_loss, learning_curve = ann.fit(X_train, y_train)
        
        # test and get error
        # y_est = ann.predict(X_test)
        # mse = annLoss(y_est, torch.Tensor(y_test)).item()
        # Error_test_ann[k * h_K + h] = [k, h, mse]

        y_est = ann.predict(X_test) # forward pass, predict labels on training set
        if (y_est.shape != y.shape):
            y_est = y_est.squeeze(1)
        loss = annLoss(y_est, torch.Tensor(y_test)) # determine loss
        loss_value = loss.data.numpy() 
        print(f"Loss: {loss_value}")
        Error_test_ann[k * h_K + h] = [k+1, hidden_units, loss_value]
    

df = pd.DataFrame(Error_test_ann, 
             columns=['folds', 
                      'hidden_units',
                      'loss'])
df.to_csv('ann_loss.csv')
print(df)



#     # Display the learning curve for the best net in the current fold
#     h, = summaries_axes[0].plot(learning_curve, color=color_list[k])
#     h.set_label('CV fold {0}'.format(k+1))
#     summaries_axes[0].set_xlabel('Iterations')
#     summaries_axes[0].set_xlim((0, max_iter))
#     summaries_axes[0].set_ylabel('Loss')
#     summaries_axes[0].set_title('Learning curves')
    
# # Display the MSE across folds
# summaries_axes[1].bar(np.arange(1, K+1), np.squeeze(np.asarray(errors)), color=color_list)
# summaries_axes[1].set_xlabel('Fold')
# summaries_axes[1].set_xticks(np.arange(1, K+1))
# summaries_axes[1].set_ylabel('MSE')
# summaries_axes[1].set_title('Test mean-squared-error')

# # Display a diagram of the best network in last fold
# print('Diagram of best neural net in last fold:')
# weights = [ann.net[i].weight.data.numpy().T for i in [0,2]]
# biases = [ann.net[i].bias.data.numpy() for i in [0,2]]
# tf =  [str(ann.net[i]) for i in [1,2]]
# draw_neural_net(weights, biases, tf, attribute_names=attributeNames)


# # When dealing with regression outputs, a simple way of looking at the quality
# # of predictions visually is by plotting the estimated value as a function of 
# # the true/known value - these values should all be along a straight line "y=x", 
# # and if the points are above the line, the model overestimates, whereas if the
# # points are below the y=x line, then the model underestimates the value
# plt.figure(figsize=(10,10))
# y_est = y_est.data.numpy().squeeze(1)
# y_true = y_test.data.numpy()
# axis_range = [np.min([y_est, y_true])-1,np.max([y_est, y_true])+1]
# plt.plot(axis_range,axis_range,'k--')
# plt.plot(y_true, y_est,'ob',alpha=.25)
# plt.legend(['Perfect estimation','Model estimations'])
# plt.title('Median income: estimated versus true value (for last CV-fold)')
# plt.ylim(axis_range); plt.xlim(axis_range)
# plt.xlabel('True value')
# plt.ylabel('Estimated value')
# plt.grid()

# plt.show()

# Print the average classification error rate
# print('\nEstimated generalization error, RMSE: {0}'.format(round(np.sqrt(np.mean(errors)), 4)))
