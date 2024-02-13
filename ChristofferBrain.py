import neuralNet as net
import ImageViewer as v
import fileParser as fp
import numpy as np


class ChrisBrain:
    # Needs on the format (numLayers, [numNeurons], [actFunc], [weightInit],lossFunc, (regulizer, regRate), softmax, batchSize)
    def __init__(self, filePath):
        #Extracting info from file
        info = self.initFile(filePath)

        #Image size
        self.size = info[1][0]
        print(self.size)

        #Making the Network
        self.brain = net.NeuralNetwork(info)

        #Making the visualizer
        self.visulizer = v.Visualizer()

        # String date while training
        self.lossData = []
        self.validationLoss = []

        #Stroing filepath for parsing
        self.filePath = filePath

    
    # Train the network
    def fit(self, data, target, validation, valTarget, learningRate, epochs):
        #Dividing the data into batches
        dataBatch = self.brain.makeBatches(data)
        targetBatch = self.brain.makeBatches(target)

        valBatch = self.brain.makeBatches(validation)[0]
        valTargetBatch = self.brain.makeBatches(valTarget)[0]

        #Training 
        for epoch in range(epochs):

            for sampleBatch, sampleTarget in zip(dataBatch, targetBatch):
                #Setting the batch size
                self.brain.updateBatchSize(sampleBatch.shape[0])

                print("State: ", sampleBatch.shape)
                #Forward pass
                loss = self.brain.forward(sampleBatch, sampleTarget)

                #Backward pass
                self.brain.backward(learningRate, sampleTarget)

                # Storing the loss
                self.lossData.append(loss)

                self.brain.updateBatchSize(valBatch.shape[0])

                #Validation
                validationLoss = self.brain.forward(valBatch, valTargetBatch)
                self.validationLoss.append(validationLoss)
                #self.validationLoss.append(0)

                #Verbose
                #self.verbose()

        self.visulizer.drawLoss(self.lossData, self.validationLoss)

    # Predict
    def predict(self, data):
        #Adjusting batchsize
        self.brain.updateBatchSize(data.shape[0])

        #Predicting and decode
        predictions = self.brain.test(data)

        #Reformatting
        result = (data, predictions)
        
        #Draw the images TODO: Pred must be mapped 
        self.visulizer.drawImages(result, predictions)
    

    # Reading network config from file
    def initFile(self, filePath):
        #Parse the file
        data = fp.fileParser(filePath)

        globalValue = data['GLOBALS']
        layers = data['LAYERS']

        #Defining global values
        batchSize = int(globalValue['batchSize'])
        lossFunction = globalValue['loss']
        learningRate = float(globalValue['lrate'])
        regConst = globalValue['wreg']
        regFunc = globalValue['wrt']
        regulizer = (regFunc, float(regConst))
        softmax = True

        # Defining the layers
        numNeurons = []
        actFunc = []
        weightInit = []
        localRate = []
        
        inputLayer = layers.pop(0)
        numNeurons.append(int(inputLayer['input']))

        if(layers[-1]['type'] != 'softmax'):
            layers.pop(-1)
            softmax = False
            print("Softmax layer disabled")
        else:
            layers.pop(-1)

        #Itereating on each layer 
        for layer in layers:
            numNeurons.append(int(layer['size']))
            actFunc.append(layer['act'])
            weightInit.append(layer['wr'])
            if('lrate' in layer):
                localRate.append(float(layer['lrate']))
            else:
                localRate.append(learningRate)

        #Reformatinng to neuralNetwork form
        neuralInit = (len(numNeurons), numNeurons, actFunc, weightInit, localRate, lossFunction, regulizer, softmax, batchSize)
        
        return neuralInit

        