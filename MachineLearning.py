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
        IH_weights = np.array([],dtype=self.dtype).reshape(0, self.numInput)

        for i in range(0,self.numOutput):
            weight = (output[i] - guess[i])
            weight = np.multiply(weight, np.multiply(self.outputs[i].T, (1 - self.outputs[i].T)))
            weight = np.multiply(weight, self.hiddenInputs)
            weight = weight.T

            HO_weights = np.concatenate((HO_weights,weight),axis=0)

        self.biasO += np.multiply(output - guess, np.multiply(self.outputs, (1-self.outputs)))
        self.weightHO += HO_weights
        
        errorsHO = np.array([np.sum(HO_weights, axis=0)]).T
        for i in range(0,len(errorsHO)):
            weight = (errorsHO[i])
            weight = np.multiply(weight, np.multiply(self.hiddenInputs[i].T, (1 - self.hiddenInputs[i].T)))
            weight = np.multiply(weight, input)
            weight = weight.T
            IH_weights = np.concatenate((IH_weights,weight),axis=0)

        self.biasH += np.multiply(np.multiply((1- self.hiddenInputs), self.hiddenInputs), errorsHO)
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


data = np.array([[1],[0]])
answer = np.array([[0]])

x = NeuralNetwork(2,3,1)

for i in range(0,1000):
    #print(x.weightIH)
    x.train(np.array([[1],[0]]),np.array([[1]]))
    x.train(np.array([[0],[1]]),np.array([[1]]))
    x.train(np.array([[1],[1]]),np.array([[0]]))
    x.train(np.array([[0],[0]]),np.array([[0]]))
    print("IH weights: " + str(x.weightIH))
    print("HO weights: " + str(x.weightHO))
    print("Bias H: " + str(x.biasH))
    print("Bias O: " + str(x.biasO))
    print("\n\n\n")
    
print(x.feedforward(np.array([[0],[0]])))
print(x.feedforward(np.array([[1],[0]])))
print(x.feedforward(np.array([[0],[0]])))
print(x.feedforward(np.array([[0],[0]])))









