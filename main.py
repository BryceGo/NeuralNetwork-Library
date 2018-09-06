import random
import numpy.matlib
import numpy as np
import MachineLearning as ml

test = ml.NeuralNetwork(2,9,1)

#Training the model to learn XOR
for i in range(0,1000):
    test.train(np.array([[1],[0]]),np.array([[1]]))
    test.train(np.array([[0],[1]]),np.array([[1]]))
    test.train(np.array([[1],[1]]),np.array([[0]]))
    test.train(np.array([[0],[0]]),np.array([[0]]))
    print("IH weights: " + str(test.weightIH))
    print("HO weights: " + str(test.weightHO))
    print("Bias H: " + str(test.biasH))
    print("Bias O: " + str(test.biasO))
    print("\n\n")

#Testing the model with XOR
print(test.feedforward(np.array([[1],[0]])))
print(test.feedforward(np.array([[0],[1]])))
print(test.feedforward(np.array([[1],[1]])))
print(test.feedforward(np.array([[0],[0]])))









