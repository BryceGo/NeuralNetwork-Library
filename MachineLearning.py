import random
import numpy.matlib
import numpy as np
import math
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




