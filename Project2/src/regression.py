import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from prepData import Data
from Models import linearRegression, baseline

DATA_PATH = 'California_Housing_DB/Data/housing.csv'
data = Data(DATA_PATH)
data.remove_col(9)

X_train,X_test,y_train,y_test = data.get_train_test('median_income')

lr = linearRegression()
baseline = baseline()

models = [lr, baseline]

X_train_ones = lr.add_ones_col(X_train)
theta = lr.trainLinearReg(X_train_ones, y_train, 0)
print(theta)

# # train
# lr.fit(X_train,y_train)
# # test
# y_est = lr.predict(X_test)

# # results
# # coefs= model.get_coefs()
# print("y_est: ", y_est)
# residual = lr.get_residual(y_est, y_test)
# print('\nThe error in the training data is : {}'.format(lr.mse(y_test,y_est)))





# # Display scatter plot
# figure()
# subplot(2,1,1)
# plot(y_test, y_est, '.')
# xlabel('Median Income (true)'); ylabel('Median Income (estimated)');
# subplot(2,1,2)
# hist(residual,40)
# show()


# print('\nThe error in the training data is : {}'.format(lr.mse(y_test,y_est)))
#The error w/o any manipulation is 1.226, witbh j rooms, bedrooms, 
#and ppl per house it is .9276, and if we add the same measures per person, 
#it goes down to .8987
#Oow wow added ones line and it dropped again


# #adds rooms per bedrooms 
# room_idx =3
# bedroom_idx = 4

# X_room_per_bedroom = (X[:,room_idx]/X[:,bedroom_idx]).reshape(-1,1)

# X = np.asarray(np.bmat('X, X_room_per_bedroom'))