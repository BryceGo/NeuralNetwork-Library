import random
import numpy.matlib
import numpy as np
import math

class NeuralNetwork:
    dtype = np.float32
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
        self.biasO = np.asarray(np.matlib.ones((self.numOutput,1)), dtype = self.dtype)
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
        
        HO_weights = np.array([],dtype=self.dtype).reshape(0,self.numHidden)
        IH_weights = np.array([],dtype=self.dtype).reshape(0,self.numInput)

        for i in range(0,self.numOutput):
            weight = (output[i] - guess[i])
            weight *= np.multiply(self.outputs), (1 - self.outputs)
            weight *= self.hiddenInputs
            weight = weight.T
            HO_weights = np.concatenate((HO_weights,weight),axis=0)

        self.weightHO += HO_weights

        errorsHO = np.sum(HO_weight, axis=0)

        for i in range(0,len(errorsHO)):
            weight = (errorsHO[i])
            weight *= np.multiply(self.hiddenInputs, (1 - self.hiddenInputs))
            weight *= self.input
            weight = weight.T
            IH_weights = np.concatenate((IH_weights,weight),axis=0)

        self.weightIH += IH_weights
        
        return

    def sigmoid(self, input):
        array = np.copy(input)
        for i in range(0,len(input)):
            e = math.exp(array[i])
            array[i] = e/(e+1)                        
        return array


class Perceptron:
    weights = []
    length = 0
    def __init__(self, length):
        self.length = length
        for i in range(0,self.length):
            self.weights.append(random.uniform(-1,1))        
        return


    def guess(self, inputs):
        sum = 0

        print("weights are " + str(self.weights))
        print("inputs are " + str(inputs))
        for i in range(0,len(inputs)):
            sum += inputs[i]*self.weights[i]
        return self.sign(sum)

    def sign(self,input):
        print("Input is " + str(input))
        if input >= 0:
            return 1
        return -1

    def train(self, inputs, target):
        guess = self.guess(inputs)
        error = target - guess
        cost = 0
        for i in range(0,len(inputs)):
            self.weights[i] += float(0.1*error*inputs[i])
        return error

    def answer(self, inputs_list):
        targets_list = []
        for i in inputs_list:
            if i[0] - i[1] <= 1:
                targets_list.append(1)
            else:
                targets_list.append(-1)
        return targets_list


data = np.array([
x = NeuralNetwork(1,2,1)
