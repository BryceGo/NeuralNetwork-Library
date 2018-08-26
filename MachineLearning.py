import random
import numpy.matlib
import numpy as np
import math
#https://theclevermachine.wordpress.com/2014/09/06/derivation-error-backpropagation-gradient-descent-for-neural-networks/
class NeuralNetwork:
    dtype = np.float64
    numInput = 0
    numHidden = 0
    numOutput = 0
    def __init__(self, numberInput, numberHidden, numberOutput):
        self.numInput = numberInput
        self.numHidden = numberHidden
        self.numOutput = numberOutput
        self.weightIH = np.asarray(np.matlib.rand(self.numHidden,self.numInput),dtype=self.dtype)
        self.weightHO = np.asarray(np.matlib.rand(self.numOutput,self.numHidden), dtype=self.dtype)
        self.biasH = np.asarray(np.matlib.ones((self.numHidden,1)), dtype = self.dtype)
        self.biasO =  np.asarray(np.matlib.ones((self.numOutput,1)), dtype = self.dtype)
        return

    def feedforward(self, input):
        L1 = np.dot(self.weightIH, input)
        L1 += self.biasH
        L1 = self.sigmoid(L1)

        #Use for Backpropagation in inputs
        self.hiddenInputs = L1

        L2 = np.dot(self.weightHO, L1)
        L2 += self.biasO
        L2 = self.sigmoid(L2)
        self.outputs = L2
        return L2

	#Input should be 1xn
	#Output should be 1xn
    def train(self, input, output):    
        guess = self.feedforward(input)
        #Backpropagate from Output to Hidden Layer

        weight = output - guess
        weight = np.multiply(weight, np.multiply(self.outputs, (1 - self.outputs)))
        propagatedWeights = weight
        propagatedBias = weight
        weight = np.multiply(self.hiddenInputs, weight.T).T
        self.weightHO += weight

        propagatedWeights = np.multiply(propagatedWeights,self.weightHO)
        propagatedWeights = np.array([np.sum(propagatedWeights,axis=0)])


        self.biasO += propagatedBias

        weight = np.multiply(propagatedWeights.T,np.multiply(self.hiddenInputs, (1-self.hiddenInputs)))
        propagatedBias = weight
        weight = np.dot(input,weight.T).T
        self.weightIH += weight
        
        self.biasH += propagatedBias
        return

    def sigmoid(self, input):
        array = np.copy(input)
        for i in range(0,len(input)):
            e = math.exp(-array[i])
            array[i] = 1/(1+e)
        return array

data = np.array([[1],[0]])
answer = np.array([[0]])

x = NeuralNetwork(2,9,1)

for i in range(0,50000):
    #print(x.weightIH)
    #x.train(np.array([[1],[1]]),np.array([[0],[0]]))
    x.train(np.array([[1],[0]]),np.array([[0.89]]))
    x.train(np.array([[0],[1]]),np.array([[0.48]]))
    x.train(np.array([[1],[1]]),np.array([[0.22]]))
    x.train(np.array([[0],[0]]),np.array([[0.11]]))
    print("IH weights: " + str(x.weightIH))
    print("HO weights: " + str(x.weightHO))
    print("Bias H: " + str(x.biasH))
    print("Bias O: " + str(x.biasO))
    print("\n\n\n")
    
print(x.feedforward(np.array([[1],[0]])))
print(x.feedforward(np.array([[0],[1]])))
print(x.feedforward(np.array([[1],[1]])))
print(x.feedforward(np.array([[0],[0]])))









