import numpy as np
from layer import Layer

class NeuralNetwork:
    def __init__(self, numerOfLayers, numberOfNeurons, softMaxLayer, batchSize):
        self.numerOfLayers = numerOfLayers
        self.numberOfNeurons = numberOfNeurons
        self.softMaxLayer = softMaxLayer
        self.batchSize = batchSize

        self.layers = [Layer(numberOfNeurons) for i in range(numerOfLayers)]

    
    #Softmax function (optinal for the last layer)
    def softMax(self, input):
        return np.exp(input) / np.sum(np.exp(input), axis=0)
    
    def derivativeSoftMax(self, input): # TODO: Check if this is correct
        return np.exp(input) / np.sum(np.exp(input), axis=0) * (1 - np.exp(input) / np.sum(np.exp(input), axis=0))
    

    #Cross entropy loss function
    def lossFunction(self, input, target):
        # TODO: Do i need to clip???
        input = self.clipNumbers(input)

        #Finding loss TODO: Save loss for each neuvron and batch?
        loss = []
        for i in range(input.shape[0]):
            loss.append(-np.sum(np.log(input[i]) * target[i] + (1 - target[i]) * np.log(1 - input[i]) / input.shape[0]))
       
        return loss
    
    
    #Clipping numbers (to avoid NaN in loss function)
    def clipNumbers(self, input):
        epsilon = 1e-15
        return np.clip(input[0], epsilon, 1 - epsilon)


    # -----  Main functions of the network -----

    #Forward pass of a batch
    def forward(self, input, target):
        #Forwarding through layer-objects
        for layer in self.layers:
            input = layer.forward(input)
        

        if self.softMaxLayer:
            input = self.softMax(input)
            self.SoftmaxCache = input
    
        #Applying lost function TODO: Check transpose
        loss = self.lossFunction(input.T, target)

        return loss

    #Backward pass of a batch
    def backward(self, learningRate, target):
        #If softmax layer
        if(self.softMaxLayer):
            #Jacobian L -> S
            J_LS = self.SoftmaxCache - target
        
            #Jacobian S -> N
            sizeMatrix = len(self.SoftmaxCache)
            J_SN = np.zeros((self.batchSize, sizeMatrix, sizeMatrix))

            for b in range(self.batchSize):
                J_SN = -np.outer(self.SoftmaxCache[b], self.SoftmaxCache[b])
                np.fill_diagonal(J_SN, self.SoftmaxCache[b] * (1 - self.SoftmaxCache[0]))


            #Jacobian L -> N
            J_LN = np.array([np.dot(J_LS, J_SN[i]) for i in range(len(J_LS))]) 

            #Backwarding through layer-objects
            for layer in reversed(self.layers):
                J_LN = layer.backWard(J_LN, self.batchSize)
        
        #If no softmax layer 
        else:
            #Jacobian L -> N
            J_LN = self.layer[0].activationFunction(self.layers[-1]) - target #TODO: Check if this is correct

            #Backwarding through layer-objects
            for layer in reversed(self.layers):
                J_LN = layer.backWard(J_LN, learningRate)
        

        # Updating weight and biases
        for layer in self.layers:
            layer.W -= learningRate * layer.J_LW
            layer.B -= learningRate * layer.J_LB

