
## Author : Arnaud HINCELIN
## Date : Oct 22 2021
## Goal : Understand who works a unique neuron learning (=perceptron) and matrix computations
## Project : Build learning neuron functions with matrix (only vectorized computations)
## This unique neurone stands for the famous LINEAR REGRESSION, developed to resolve linear problems ! 


import numpy as np


"""
How create NN from scratch : 

1. Initialize Network. 
2. Forward Propagate. 
3. Back Propagate Error. 
4. Train Network. 
5. Predict. 
"""



"""
1. NN

"""

class Neural_Network():
    
    def __init__(self, nbLayers) -> None:
        self.Nlayers = nbLayers
        self.layers = []
    
    def init_NN(self):
        pass

    
class Layer():
    
    def __init__(self, nbNeurones, nbInput, indexLayer) -> None:
        self.NNeurones = nbNeurones
        self.NInputs = nbInput
        self.Neurones = []
        self.Activations = np.zeros( (nbNeurones, 1) )
        self.WeigthsMatrix = np.zeros( (nbNeurones, self.NInputs+1 ) )
        self.numLayer = indexLayer

    def init_Layer(self):
        for i in range (self.NNeurones):
            self.Neurones.append( Neuron(nbInput=self.NInputs+1) )
            self.WeigthsMatrix[i,:] = self.Neurones[i].getWeight()
        print("W matrix of layer {} ".format(self.numLayer), self.WeigthsMatrix)

    
    def forward_layer(self, input):

        input = np.array(input)
        input = np.concatenate( ([1], input), axis=0 )
        input = np.reshape( input, newshape=(input.shape[0], 1))

        if(input.shape[0] != self.NInputs+1):
            print("Error forward_layer : input shape != NInputs shape")
        

        print("input array: ", input.shape)
        print("W array: ", self.WeigthsMatrix.shape )

        self.Activations = np.dot( a=self.WeigthsMatrix, b=input )

        return self.Activations

        


    def getNbWeigths(self):
        print("nb of params : ", self.WeigthsMatrix.size )




class Neuron():

    def __init__(self, nbInput) -> None:
        self.Ninput = nbInput
        self.Weight = np.random.randn( 1, self.Ninput )
        self.Delta = 0


        #print(self.Weight)
        
    def getWeight(self):
        return self.Weight
    

    
### 
### MAIN
###

X = np.array([3, 2, 4])
print("X : ", X)


L = Layer(nbNeurones=2, nbInput=3, indexLayer=0)
L.init_Layer()
A = L.forward_layer(X)

L2 = Layer(nbNeurones=1, nbInput=2, indexLayer=1)
L2.init_Layer()
A2 = L2.forward_layer(A)

print(A2)






