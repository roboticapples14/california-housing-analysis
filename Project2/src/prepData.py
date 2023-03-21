import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

class Data:
    def __init__(self, data):
        '''
        :param data: filepath to csv with our data
        '''
        self.df = pd.read_csv(data).dropna()
        # m = entries, n = attributes
        N, M = self.df.shape
        self.size = (N, M)

    def getAttributeNames(self):
        return list(self.df.columns)        

    def remove_col(self, i):
        '''
        remove ith column of the data
        '''
        self.df.drop(self.df.columns[[i]], axis=1, inplace=True)  # df.columns is zero-based pd.Index
        self.size = self.df.shape

    def add_ones_col(self, X):
        '''
        adds a ones matrix to X
        Not needed if LinearRegression model's fit_intercept is True (default), so should not be needed
        '''
        ones_matrix = np.ones((X.shape[0],1))
        X = np.asarray(np.bmat('ones_matrix, X'))
        return X

    def get_train_test(self, col_name, test_size=0.2, add_ones=False):
        '''
        splits the data into features X and target vector y, where y is the column col_name,
        divided into training and test divisions
        :param col_name: name of y column
        :param test_size: percentage of full data for test set
        :returns: tuple (X train (dataframe), X test (dataframe), y train (np array), y test (np array))
        '''
        y = self.df.pop(col_name)
        X = self.df
        X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)
        if add_ones:
            X_train = self.add_ones_col(X_train)
            X_test = self.add_ones_col(X_test)
        return (X_train,X_test,np.asarray(y_train),np.asarray(y_test))

    def get_X_y(self, col_name):
        y = np.asarray(self.df.pop(col_name))
        X = np.asarray(self.df)
        return (X, y)