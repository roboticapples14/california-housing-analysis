import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from sklearn import model_selection
from prepData import Data
from Models import ANN
from visualize import visualize_decision_boundary, draw_neural_net

DATA_PATH = 'California_Housing_DB/Data/housing.csv'
data = Data(DATA_PATH)
data.remove_col(9)
attributeNames = data.getAttributeNames()

y_colname = 'median_income'
classNames = [y_colname]
X_train,X_test,y_train,y_test = data.get_train_test(y_colname)


ann = ANN(M=data.size[1]-1, n_hidden_units=5)
max_iter = 10000


best_final_loss, learning_curve = ann.fit(X_train, y_train)

# test and get error
y_est = ann.predict(X_test)

for est, true in zip(y_est, y_test):
    print("estimated y: {}, true y: {}".format(est[0], true))

error_rate = ann.get_residual(y_est, y_test)


# Display a diagram of the best network in last fold
print('Diagram of best neural net in last fold:')
weights = [ann.net[i].weight.data.numpy().T for i in [0,2]]
biases = [ann.net[i].bias.data.numpy() for i in [0,2]]
tf =  [str(ann.net[i]) for i in [1,2]]
