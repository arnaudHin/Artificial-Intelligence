
## Author : Arnaud HINCELIN
## Date : Oct 21 2021
## Goal : Understand who works a unique neuron learning (=perceptron) and matrix computations
## Project : Build learning neuron functions with matrix (only vectorized computations)
## This unique neurone stands for the famous LINEAR REGRESSION, developed to resolve linear problems ! 

import numpy as np
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
from math import sqrt


"""
Perceptron : 
x_1 -> w_1 |
            | z -> a
x_2 -> w_2 |

n = 2 | m = 100 | L = 1
"""

## ------------------------------------------
## Functions
## ------------------------------------------
m = 100
n = 2

def create_dataset():
    X, y = make_blobs(n_samples= m, n_features= n, centers= n, random_state= 0)
    one = np.ones( (X.shape[0], 1) )
    y = y.reshape( (y.shape[0], 1) )

    plt.scatter(X[:,0], X[:,1], c=y, cmap='plasma')
    plt.show()

    #np.concatenate((arrays), axis=1) -> concat arrays with columns
    #np.concatenate((arrays), axis=0) -> concat arrays with rows
    #np.concatenate((arrays), axis=None) -> flatten and concat all datas
    X = np.concatenate((one, X), axis=1) #(m, n+2)
    print(X[:4,:])
    print("dim X : ", X.shape)
    print("dim y : ", y.shape)

    return (X,y)

def create_weights(X, num_neurones):
    num_features = X.shape[1]
    W = np.random.randn(num_features, num_neurones)
    return W


def create_weights_stf(X, num_neurones):
    num_features = X.shape[1]
    epsilon = sqrt()
    W = np.random.randn(num_features, num_neurones)
    return W


def forward(inputs, weigths):
    Z = np.array( inputs.dot(weigths) )
    if( (Z.shape[0] != inputs.shape[0]) or (Z.shape[1] != 1) ): #check shape
        print("Error forward")
    return Z

def ReLU(inputs_datas):
	return np.max(inputs_datas, 0)

def Sigmoid(inputs_datas):
	return 1/(1+np.exp(-inputs_datas))

def loss_function(A, y):

    if(A.shape != y.shape):
        print("Error loss function")

    #np.sum(X) -> sum of all -> it is a scalar ! shape=(1,)
    #np.sum(X, axis=0) -> sum of all colums -> it is a vector ! shape=(1, colums)
    #np.sum(X, axis=1) -> sum of all lines -> it is a vector ! shape=(1, lines)
    l = ( y*np.log(A) + (1-y) * (np.log(1-A)) ) #(m, 1)
    L = (-1/(A.shape[0]) ) * np.sum(l) #scalar (1,)

    return L
    
def batch_gradient_descent(W, dW, alpha):
    if(W.shape != dW.shape):
        print("Error batch grandient descent")
    
    W_new = W-alpha*dW
    return W_new

def delta_computation(X, A, Y):
    if(A.shape != Y.shape):
        print("Error delta 0")

    dW = np.array( (1/m) * (X.T).dot( A-Y ) )

    if(dW.shape != (n+1, 1)):
        print("error delta 2")

    return dW

def show_loss(loss):

    print("loss final : ",loss[len(loss)-1] )
    #np.arange( a ) = create int np.array from 0 to (a-1) with shift of 1 
    plt.plot( np.arange( len(loss) ) , loss   )
    plt.show()


def run_artificial_neuron(X, y, iter_n, learning_rate):
    (W) = create_weights(X, 1)
    loss = []

    for i in range(iter_n):
        Z = forward(inputs=X, weigths=W) 
        A = Sigmoid(Z)
        loss.append( loss_function(A, y) )
        dW = delta_computation(X, A, y)
        W = batch_gradient_descent(W=W, dW=dW, alpha=learning_rate)

    print(W[0])
    print(W[1])

    show_loss(loss)


## ------------------------------------------
## main program
## ------------------------------------------

(X, y) = create_dataset()

run_artificial_neuron(X, y, 100, 0.1)


















