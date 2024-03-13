import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype

# creating standard variables (u-x)/sigma
def normalize(X): 
    if isinstance(X, pd.DataFrame):
        for c in X.columns:
            if is_numeric_dtype(X[c]):
                u = np.mean(X[c])
                s = np.std(X[c])
                X[c] = (X[c] - u) / s
        return
    for j in range(X.shape[1]):
        u = np.mean(X[:,j])
        s = np.std(X[:,j])
        X[:,j] = (X[:,j] - u) / s

# ∇L(β) = −2XT (y − Xβ)
def loss_gradient(X, y, B): 
    return -np.dot(np.transpose(X), y - np.dot(X, B))

# L(β)=(y−Xβ)T(y−Xβ)+λβTβ
def loss_ridge(X, y, B, lmbda): 
    mid = y - np.dot(X, B)
    return np.dot(np.transpose(mid), mid)+ lmbda * np.dot(np.transpose(B), B)

 # with L2 regularization: ∇L(β) = −XT (y − Xβ) + λβ
def loss_gradient_ridge(X, y, B, lmbda):
    return -np.dot(np.transpose(X), y - np.dot(X, B))+ lmbda * B

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# logistic regression without regularization
def log_likelihood_gradient(X, y, B, lmbda): 
    z = np.dot(X, B)
    p = sigmoid(z)
    # Gradient of the log likelihood
    gradient = -np.dot(np.transpose(X), (y - p))
    return gradient

def minimize(X, y, loss_gradient,
              eta=0.00001, lmbda=0.0,
              max_iter=1000, addB0=True,
              precision=1e-9):
    # check X and y dimensions and set n and p to X dimensions
    if X.ndim != 2:
        raise ValueError("X must be n x p for p features")
    n, p = X.shape
    if y.shape != (n, 1):
        raise ValueError(f"y must be n={n} x 1 not {y.shape}")
    # for linear or logistic regression, we estimate B0 by adding a column of 1s and increase p by 1
    if addB0:
        X0 = np.ones((n,1))
        X = np.hstack((X0, X))
        p += 1

    # initiate a random vector of Bs of size p
    B = np.random.random_sample(size=(p, 1)) * 2 - 1   # make between [-1,1)

    # start the minimization procedure 
    eps = 1e-5 # prevent division by 0
    
    h = 0
    for i in range(max_iter): 
        gradient_vector = loss_gradient(X, y, B, lmbda) 

        # update h vector
        h += gradient_vector**2

        # updata B
        B =  B - eta/(np.sqrt(h)+eps) * gradient_vector

        # get the norm of gradient
        norm_of_gradient = np.linalg.norm(gradient_vector)

        if norm_of_gradient < precision:
            break
    #print("b",B,gradient_vector,"norm", norm_of_gradient,i)
    return B
    

class LinearRegression: 
    def __init__(self,
                 eta=0.00001, lmbda=0.0,
                 max_iter=1000):
        self.eta = eta
        self.lmbda = lmbda
        self.max_iter = max_iter

    def predict(self, X):
        # add 1s into X
        n = X.shape[0]
        B0 = np.ones(shape=(n, 1))
        X = np.hstack([B0, X])
        return np.dot(X, self.B)

    def fit(self, X, y):
        self.B = minimize(X, y,
                          loss_gradient,
                          self.eta,
                          self.lmbda,
                          self.max_iter)


class LogisticRegression: 
    "Use the above class as a guide."
    def __init__(self,
                 eta=0.00001, lmbda=0.0,
                 max_iter=1000):
        self.eta = eta
        self.lmbda = lmbda
        self.max_iter = max_iter

    def fit(self, X, y):
        self.B = minimize(X, y,
                    log_likelihood_gradient,
                    self.eta,
                    self.lmbda,
                    self.max_iter) 

    def predict_proba(self, X):
        """
        Compute the probability that the target is 1. Basically do
        the usual linear regression and then pass through a sigmoid.
        """
        n = X.shape[0]
        B0 = np.ones(shape=(n, 1))
        X = np.hstack([B0, X])
        z = np.dot(X, self.B)
        prob = sigmoid(z)
        return prob

    def predict(self, X):
        """
        Call self.predict_proba() to get probabilities then, for each x in X,
        return a 1 if P(y==1,x) > 0.5 else 0.
        """
        prob = self.predict_proba(X)
        # prob> 0.5
        out = []
        for i in prob:
            if i > 0.5:
                out.append(1)
            else:
                out.append(0)
        return np.array(out)


class RidgeRegression: 
    def __init__(self,
                 eta=0.00001, lmbda=0.0,
                 max_iter=1000):
        self.eta = eta
        self.lmbda = lmbda
        self.max_iter = max_iter

    def predict(self, X):
        n = X.shape[0]
        B0 = np.ones(shape=(n, 1))
        X = np.hstack([B0, X])
        return np.dot(X, self.B)
    
    def fit(self, X, y):
        # estimate B0 separately
        min = minimize(X, y,
                    loss_gradient_ridge,
                    self.eta,
                    self.lmbda,
                    self.max_iter,
                    addB0=False) 
        intercept = np.mean(y)
        self.B = np.insert(min, 0, intercept)
        
