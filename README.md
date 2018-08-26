# NeuralNetwork-Library
Created a Multi-layered Perceptron with one output layer, one hidden layer and one input layer.

## Usage:

### Creating the Network
To create the network, call the NeuralNetwork function.

#### NeuralNetwork(numInputs, numHidden, numOutput)
numInputs = number of Input nodes

numHidden = number of nodes in the hidden layer

numOutput = number of Output nodes.

-------

### Feeding forward

To extract a guess from the Network, create a 2d array using numpy with the dimensions as (1,numInputs).

Call the function:
#### feedforward(data)
-------
### Training the model

To train, you must have an input data using numpy's array function. Dimensions should be a (1,numInputs).

The output dimensions should also be of (1,numOutput).

Call the train function to train the model.

#### train(inputData, outputData)
------

## References

[Derivation of Error Backrpopagation](https://theclevermachine.wordpress.com/2014/09/06/derivation-error-backpropagation-gradient-descent-for-neural-networks/)

[How backpropagation works](http://neuralnetworksanddeeplearning.com/chap2.html)


