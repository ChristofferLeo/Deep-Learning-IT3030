from neuralNet import NeuralNetwork
import numpy as np


def main():
    nettwork = NeuralNetwork(2,5,True,5)

    #make mini match for input target
    input = np.array([np.random.rand(5) for i in range(5)])
    target = np.array([np.random.rand(5) for i in range(5)])

    nettwork.forward(input, target)
    nettwork.backward(0.1, target)
if __name__ == "__main__":
    main()