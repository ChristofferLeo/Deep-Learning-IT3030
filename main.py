from neuralNet import NeuralNetwork
from ImageViewer import Visualizer
from dataGenarator import Generator
from ChristofferBrain import ChrisBrain
import doodler
import numpy as np

decoder = ['ball','ring','frame','box','flower','bar','polygon','triangle','spiral']


def main():

    #Making the Brain
    Brain = ChrisBrain("setup_1.txt")

    ## Generating Data
    numSamples = 1000 #NOTE: Input here (1000)
    disturbance = 0 #NOTE: Input here
    imageSize = int(np.sqrt(Brain.size))

    gen = Generator()
    data = gen.genarateData(numSamples, imageSize, imageSize, disturbance)



    #seperating to train, valdiate, test
    train = data[0]
    validation = data[1]
    test = data[2] 

    #Fitting
    Brain.fit(train[0], train[1], validation[0], validation[1], 0.1, 10)

    #Predicting
    Brain.predict(test[0])



if __name__ == "__main__":
    main()