import numpy as np
from layer import Layer

class NeuralNetwork:
    def __init__(self, numerOfLayers, numberOfNeurons, softMaxLayer, batchSize):
        self.numerOfLayers = numerOfLayers
        self.numberOfNeurons = numberOfNeurons
        self.softMaxLayer = softMaxLayer
        self.batchSize = batchSize

        self.layers = [Layer(numberOfNeurons, numberOfNeurons, batchSize) for i in range(numerOfLayers)]

    
    #Softmax function (optinal for the last layer)
    def softMax(self, input):
        return np.exp(input) / np.sum(np.exp(input), axis=0)
    
    def d_SoftMax(self, input): 
        # Making jacobian for each batch
        numClass, numBatch = input.shape

        J = np.zeros((numBatch, numClass, numClass))
        
        # Compute the jacobian for each batch
        for b in range(numBatch):
            #SoftMax out out for b -batch
            S = input[:, b]
            diagonal = np.diag(S)
            J_batch_i = diagonal - np.outer(S, S)

            #Adds to the jacobian matrix
            J[b, :, :] = J_batch_i

        return J

    #Cross entropy loss function
    def lossFunction(self, input, target):
        # TODO: Do i need to clip???
        input = self.clipNumbers(input)
        
        #Finding loss for each batch
        loss = -np.sum(np.log(input) * input + (1 - target) * np.log(1 - input), axis=1) / len(input[0])
       
        return loss
    
    
    #Clipping numbers (to avoid NaN in loss function)
    def clipNumbers(self, input):
        epsilon = 1e-15
        return np.clip(input, epsilon, 1 - epsilon)


    # -----  Main functions of the network -----

    #Forward pass of a batch
    def forward(self, input, target):
        #Forwarding through layer-objects
        for layer in self.layers:
            input = layer.forward(input)
        

        if self.softMaxLayer:
            input = self.softMax(input)
            self.SoftmaxCache = input

        #Applying lost function 
        loss = self.lossFunction(input.T, target.T)

        return loss

    #Backward pass of a batch
    def backward(self, learningRate, target):
        #If softmax layer
        if(self.softMaxLayer):
            #Jacobian L -> S
            J_LS = self.SoftmaxCache - target
            J_LS = J_LS.T
          
            #Jacobian S -> N
            J_SN = self.d_SoftMax(self.SoftmaxCache)

            #Jacobian L -> N
            J_LN = np.array([np.dot(J_LS[i].T, J_SN[i]) for i in range(len(J_SN))])

            print(J_LN)
      
            #Backwarding through layer-objects
            for layer in reversed(self.layers):
                J_LN = layer.backWard(J_LN)
        
        #If no softmax layer 
        else:
            lastLayer = self.layers[-1]

            #Jacobian L -> N
            J_LN = (lastLayer.output.T - target)
            J_LN *= lastLayer.dF_act(lastLayer.output).T
            J_LN = J_LN.T

            #Backwarding through layer-objects
            for layer in reversed(self.layers):
                J_LN = layer.backWard(J_LN)

        # Updating weight and biases
        for layer in self.layers:
            layer.W -= learningRate * layer.J_W
            layer.B -= learningRate * layer.J_B

