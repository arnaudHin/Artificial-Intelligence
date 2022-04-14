
## Author : Arnaud HINCELIN
## Date : Nov 20 2021
## Goal : Understand who works a multi-layer percetron neuron learning (=perceptron) and matrix computations
## Project : Build learning neuron functions with matrix (only vectorized computations)
## This unique neurone stands for the famous LINEAR REGRESSION, developed to resolve linear problems ! 

#from cython.cimports.libc.stdlib import malloc, free

"""
Sympbols : 
- m = number of exemples
- n = number of features
- L = number of layers
- K = number of classes to predict
- S_l = number of N in a layer l
- h = hypothesis function, prediction of the NN
- z(l) = vector of outputs of N of layer l
- A(l) = vector of activations of a layer l
- theta(l) = vector of weights of each synapse of a layer
- J = général cost of the NN
- deltas(l) = vector of cost of the layer l
-- delats(l,n) = cost of a N n
- grad(l) = vector of gradient of a layer
- Dgrad(l) = vector o

Algorithm to develop and train NN : 
-> Preprocess dataset : (X_train, y_train, X_test, y_test)
-> Create NN & init theta with random (0-1)
-> while (converge){
    -> FP : compute each A(l) and h
    -> J : Compute cost of NN
    -> BP : compute each delta(l), and compute grad(l), and finish with compute of Dgrad(l)
    -> Gradient checking to check descent of gradient
    -> Compute gradient descent to min(J), with batch_gradient_descent
}





"""
import numpy

print( numpy.random.random(5).shape )





