import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import torch
import sklearn.linear_model as lm
from sklearn.dummy import DummyRegressor

class Model(object):
    def __init__(self, model):
        self.model = model

    def fit(self, X_train, y_train):
        ''' 
        Takes training data X and true values of y and trains the model to predict y based on X values
        :param X: Training data (database attributes - ommitting y, the column to be predicted - to consider in model)
        :param y: "True" target values (what our model is learning to predict)
        '''
        self.model.fit(X_train,y_train)

    def predict(self, X_test):
        '''
        Takes X and outputs predictions of y based on a trained model
        :param X: Testing data (database attributes - ommitting y, the column to be predicted - to consider in model)
        '''
        # Predicts values of column X from the trained model
        y_est = self.model.predict(X_test)
        return y_est

    def get_residual(self, y_est, y_test):
        '''
        calculates our model's error
        :param y_est: our model's predictions y
        :param y: "True" target values (what our model was learning to predict)
        '''
        residual = y_est-y_test
        return residual

    def get_coefs(self):
        '''
        Coefficient of the features in the decision function.
        '''
        return self.model.coef_

    def mse(self, y_true, y_est):
        summation = 0
        n = len(y_true)
        for i in range (0,n):  #looping through each element of the list
            difference = y_true[i] - y_est[i]  #finding the difference between observed and predicted value
            squared_difference = difference**2  #taking square of the differene 
            summation = summation + squared_difference  #taking a sum of all the differences
        MSE = summation/n
        return(MSE)

class linearRegression(Model):
    '''
    Linear Regression model
    '''
    def __init__(self, fit_intercept=True):
        # if fit_intercept is false, we need to append one's column to our X feature matrix
        # if it's True, we don't need this
        model = lm.LinearRegression(fit_intercept=fit_intercept)
        super().__init__(model)

class baseline(Model):
    '''
    baseline model will simply predict the mean of the y data it was trained on
    '''
    def __init__(self):
        model = DummyRegressor(strategy='mean')
        super().__init__(model)

class ANN(Model):
    '''
    Artificial Neural Network Model
    '''
    def __init__(self, M=8, n_hidden_units=1):
        '''
        :param M: number of inputs (features in X)
        :param n_hidden_units: number of hidden nodes
        '''
        # model = lambda: torch.nn.Sequential(
        #                     torch.nn.Linear(M, n_hidden_units), # M input features to H hiden units (nodes)
        #                     # 1st transfer function, either Tanh or ReLU:
        #                     torch.nn.ReLU(), #torch.nn.Tanh(),
        #                     torch.nn.Linear(n_hidden_units, 1), # H hidden units to 1 output neuron
        #                     torch.nn.Sigmoid() # final tranfer function
        #                     )

        # Define the model
        model = lambda: torch.nn.Sequential(
                    torch.nn.Linear(M, n_hidden_units), #M features to n_hidden_units
                    torch.nn.ReLU(),   #Tanh(),   # 1st transfer function,
                    torch.nn.Linear(n_hidden_units, 1), # n_hidden_units to 1 output neuron
                    # no final tranfer function, i.e. "linear output"
                    )
        super().__init__(model)
        self.net = None
        
    def fit(self, X_train, y_train, loss_fn=torch.nn.MSELoss(), n_replicates=3, max_iter=10000):
        '''
        train neural net
        :param X: X train
        :param y: y train
        :param loss_fn: how error is calculated
        :param n_replicates: number of models to train, the neural network with the lowest loss is returned.
        :param max_iter: the maximum number of iterations to do
        :returns: (final_loss, learning_curve)
        '''
        if not isinstance(X_train, torch.Tensor):
            X_train = torch.Tensor(np.asarray(X_train))
        if not isinstance(y_train, torch.Tensor):
            y_train = torch.Tensor(np.asarray(y_train))
        
        y_train = y_train.unsqueeze(1)
        net, final_loss, learning_curve = train_neural_net(self.model,
                                                    loss_fn,
                                                    X=X_train,
                                                    y=y_train,
                                                    n_replicates=3,
                                                    max_iter=max_iter)
        self.net = net
        return (final_loss, learning_curve)

    def predict(self, X_test):
        if not isinstance(X_test, torch.Tensor):
            X_test = torch.Tensor(np.asarray(X_test))
        y_est = self.net(X_test) # activation of final note, i.e. prediction of network
        # y_est = (y_sigmoid > .5).type(dtype=torch.uint8) # threshold output of sigmoidal function
        return y_est

    def get_residual(self, y_est, y_test):
        '''
        calculates our neural net's error
        :param y_est: our model's predictions y
        :param y: "True" target values (what our model was learning to predict)
        '''
        if not isinstance(y_est, torch.Tensor):
            y_est = torch.Tensor(np.asarray(y_est))
        if not isinstance(y_test, torch.Tensor):
            y_test = torch.Tensor(np.asarray(y_test))
        y_est = y_est.squeeze(1)
        criterion = torch.nn.MSELoss()
        mse = criterion(y_est, y_test)
        return mse.detach().numpy()


def train_neural_net(model, loss_fn, X, y,
                     n_replicates=3, max_iter = 10000, tolerance=1e-6):
    """
    Train a neural network with PyTorch based on a training set consisting of
    observations X and class y. The model and loss_fn inputs define the
    architecture to train and the cost-function update the weights based on,
    respectively.
    
    Usage:
        Assuming loaded dataset (X,y) has been split into a training and 
        test set called (X_train, y_train) and (X_test, y_test), and
        that the dataset has been cast into PyTorch tensors using e.g.:
            X_train = torch.tensor(X_train, dtype=torch.float)
        Here illustrating a binary classification example based on e.g.
        M=2 features with H=2 hidden units:
    
        >>> # Define the overall architechture to use
        >>> model = lambda: torch.nn.Sequential( 
                    torch.nn.Linear(M, H),  # M features to H hiden units
                    torch.nn.Tanh(),        # 1st transfer function
                    torch.nn.Linear(H, 1),  # H hidden units to 1 output neuron
                    torch.nn.Sigmoid()      # final tranfer function
                    ) 
        >>> loss_fn = torch.nn.BCELoss() # define loss to use
        >>> net, final_loss, learning_curve = train_neural_net(model,
                                                       loss_fn,
                                                       X=X_train,
                                                       y=y_train,
                                                       n_replicates=3)
        >>> y_test_est = net(X_test) # predictions of network on test set
        >>> # To optain "hard" class predictions, threshold the y_test_est
        >>> See exercise ex8_2_2.py for indepth example.
        
        For multi-class with C classes, we need to change this model to e.g.:
        >>> model = lambda: torch.nn.Sequential(
                            torch.nn.Linear(M, H), #M features to H hiden units
                            torch.nn.ReLU(), # 1st transfer function
                            torch.nn.Linear(H, C), # H hidden units to C classes
                            torch.nn.Softmax(dim=1) # final tranfer function
                            )
        >>> loss_fn = torch.nn.CrossEntropyLoss()
        
        And the final class prediction is based on the argmax of the output
        nodes:
        >>> y_class = torch.max(y_test_est, dim=1)[1]
        
    Args:
        model:          A function handle to make a torch.nn.Sequential.
        loss_fn:        A torch.nn-loss, e.g.  torch.nn.BCELoss() for binary 
                        binary classification, torch.nn.CrossEntropyLoss() for
                        multiclass classification, or torch.nn.MSELoss() for
                        regression (see https://pytorch.org/docs/stable/nn.html#loss-functions)
        n_replicates:   An integer specifying number of replicates to train,
                        the neural network with the lowest loss is returned.
        max_iter:       An integer specifying the maximum number of iterations
                        to do (default 10000).
        tolerenace:     A float describing the tolerance/convergence criterion
                        for minimum relative change in loss (default 1e-6)
                        
        
    Returns:
        A list of three elements:
            best_net:       A trained torch.nn.Sequential that had the lowest 
                            loss of the trained replicates
            final_loss:     An float specifying the loss of best performing net
            learning_curve: A list containing the learning curve of the best net.
    """
    
    import torch
    # Specify maximum number of iterations for training
    logging_frequency = 1000 # display the loss every 1000th iteration
    best_final_loss = 1e100
    for r in range(n_replicates):
        print('\n\tReplicate: {}/{}'.format(r+1, n_replicates))
        # Make a new net (calling model() makes a new initialization of weights) 
        net = model()
        
        # initialize weights based on limits that scale with number of in- and
        # outputs to the layer, increasing the chance that we converge to 
        # a good solution
        torch.nn.init.xavier_uniform_(net[0].weight)
        torch.nn.init.xavier_uniform_(net[2].weight)
                     
        # We can optimize the weights by means of stochastic gradient descent
        # The learning rate, lr, can be adjusted if training doesn't perform as
        # intended try reducing the lr. If the learning curve hasn't converged
        # (i.e. "flattend out"), you can try try increasing the maximum number of
        # iterations, but also potentially increasing the learning rate:
        #optimizer = torch.optim.SGD(net.parameters(), lr = 5e-3)
        
        # A more complicated optimizer is the Adam-algortihm, which is an extension
        # of SGD to adaptively change the learing rate, which is widely used:
        optimizer = torch.optim.Adam(net.parameters())
        
        # Train the network while displaying and storing the loss
        print('\t\t{}\t{}\t\t\t{}'.format('Iter', 'Loss','Rel. loss'))
        learning_curve = [] # setup storage for loss at each step
        old_loss = 1e6
        for i in range(max_iter):
            y_est = net(X) # forward pass, predict labels on training set
            if (y_est.shape != y.shape):
                y_est = y_est.squeeze(1)
            loss = loss_fn(y_est, y) # determine loss
            loss_value = loss.data.numpy() #get numpy array instead of tensor
            learning_curve.append(loss_value) # record loss for later display
            
            # Convergence check, see if the percentual loss decrease is within
            # tolerance:
            p_delta_loss = np.abs(loss_value-old_loss)/old_loss
            if p_delta_loss < tolerance: break
            old_loss = loss_value
            
            # display loss with some frequency:
            if (i != 0) & ((i+1) % logging_frequency == 0):
                print_str = '\t\t' + str(i+1) + '\t' + str(loss_value) + '\t' + str(p_delta_loss)
                print(print_str)
            # do backpropagation of loss and optimize weights 
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            
            
        # display final loss
        print('\t\tFinal loss:')
        print_str = '\t\t' + str(i+1) + '\t' + str(loss_value) + '\t' + str(p_delta_loss)
        print(print_str)
        
        if loss_value < best_final_loss: 
            best_net = net
            best_final_loss = loss_value
            best_learning_curve = learning_curve
        
    # Return the best curve along with its final loss and learing curve
    return best_net, best_final_loss, best_learning_curve