from neuralNet import NeuralNetwork
import numpy as np


def main():
    batchSize = 1
    numberOfNeurons = 3
    layers = 1
    softMaxLayer = 1

    nettwork = NeuralNetwork(layers,numberOfNeurons,softMaxLayer,batchSize)

    #make mini match for input target
    input = np.array([np.random.rand(numberOfNeurons) for i in range(batchSize)]).T
    target = np.array([np.random.rand(numberOfNeurons) for i in range(batchSize)]).T

    nettwork.forward(input, target)
    nettwork.backward(0.1, target)

   
if __name__ == "__main__":
    main()