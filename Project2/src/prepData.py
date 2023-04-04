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
        
    def one_out_of_k(self, col_name):
        '''
        Transforms the catagorical data in col_nane to a one-out-of-k 
        feature transformation. The matrix created is added to the end(right)
        of the current array, and the column supplied is deleted
        :param col_name: column with catagorical data
        '''
        new = self.df[[col_name]].copy()
        encoded = pd.get_dummies(new, prefix=col_name)
        self.df = pd.concat([self.df, encoded], axis=1)
        colNames = self.df.columns.values.tolist()
        index=colNames.index(col_name)
        self.remove_col(index)
        self.size = self.df.shape
        
    def standardize(self):
        '''
        Standardizes entire dataset with mean 0 and std 1. This seems wrong 
        because in theory, you should only do this to the test data set,
        but then again, this is what the assignment says to do.    
        
        Returns: tuple(sigma (the array of the different attributes' std) and 
                              mu (the array of the different attributes' 
                              averages))
        '''
        columns = list(self.df)
        sigma= [0]*len(columns)
        mu = [0]*len(columns)
        k=0
        for i in columns:
            mu[k]=  self.df[i].mean()
            sigma[k] = self.df[i].std()
            self.df[i] =( self.df[i] - mu[k] ) / sigma[k]
            k+=1
        return(sigma,mu)

    def add_divide_col(self, numCol, denCol):
        '''
        Adds another column with values numCol/denCol. It is the method users'
        responsibility to ensure that the denCol does not contain 0's
        :param numCol: name of numerator column
        :param denCol: name of denominator column
        :return: none
        '''
        self.df[''+numCol+ ' / '+ denCol] = self.df[numCol] / self.df[denCol]
        self.size = self.df.shape
        
    def add_multiply_col(self, col1, col2):
        '''
        Adds another column with values col1*col2. I
        :param numCol: name of first column
        :param denCol: name of second column
        :return: none
        '''
        self.df[''+col1+ ' * '+ col2] = self.df[col1] / self.df[col2]
        self.size = self.df.shape
        
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
        '''
        param col_name: the column being assigned as y
        returns: tuple (X(2d array),y(1d array))
        '''
        y = np.asarray(self.df.pop(col_name))
        X = np.asarray(self.df)
        return (X, y)