import numpy as np
from layer import Layer

class NeuralNetwork:
    # format: (numLayers, [numNeurons], [actFunc], [weightInit],lossFunc, (regulizer, regRate), softmax, batchSize)
    def __init__(self, info):
        
        #Extracting info
        self.numerOfLayers = info[0]
        self.numberOfNeurons = info[1]
        self.regulizer = info[6]
        self.softMaxLayer = info[7]
        self.batchSize = info[8]

        self.layers = []
        self.lossFunction = None
        self.makeNetwork(info)

    ## Make nettwork
    def makeNetwork(self, info):
         # Extracting info
        numLayers, neuronsLayer, actFuncLayer, weightInit, localRate, lossFunc, regulizer, softmax, batchSize = info

        #Setting up the network
        #self.layers.append(Layer(neuronsLayer[1], neuronsLayer[0], localRate[0], batchSize, actFuncLayer[0], weightInit[0]))
        for i in range(1, numLayers):
            self.layers.append(Layer(neuronsLayer[i], neuronsLayer[i-1], localRate[i-1], batchSize, actFuncLayer[i-1], weightInit[i-1]))

        # Setting the loss function
        self.lossFunction = lossFunc



    #Softmax function (optinal for the last layer)
    def softMax(self, input):
        #print(input)
        #print(np.sum(np.exp(input), axis=1))
        temp = np.exp(input) / np.sum(np.exp(input), axis=1).reshape((input.shape[0], 1))
        return temp
    

    def d_SoftMax(self, input): 
        # Making jacobian for each batch
        input = self.softMax(input)

        return input * (1- input)

    #Cross entropy loss function
    def Loss(self, input, target):
        #Clipping numbers (to avoid 0 in log)
        #input = np.clip(input, 1e-7, 1 - 1e-7)
        

        
        ## Cross entropy loss function
        if self.lossFunction == "CrossEntropy":
            # Cross-entropy loss calculation
            #print(input.shape[0])
            loss = -np.sum(target * np.log(input) + (1 - target) * np.log(1 - input)) / input.shape[0] #TODO:

        elif self.lossFunction == "MSE":
            # Mean squared error loss calculation
            loss = np.mean(np.square(input - target)) / input.shape[0] #TODO:

        # Apply regularization if specified
        if self.regulizer[0] == "L1":
            W = self.layers[-1].W

            L1 = self.regulizer[1] * np.sum(np.abs(W))
            loss += L1
        elif self.regulizer[0] == "L2":
            W = self.layers[-1].W

            L2 = self.regulizer[1] * np.sum(np.square(W))
            loss += L2
        #print(loss)
        return loss
    
    def updateBatchSize(self, batchSize):
        self.batchSize = batchSize
        for layer in self.layers:
            layer.B = np.tile(layer.B[0], (self.batchSize, 1))
    
    
    #Clipping numbers (to avoid NaN in loss function)
    def clipNumbers(self, input):
        epsilon = 1e-15
        return np.clip(input, epsilon, 1 - epsilon)
    
    #Update weights and biases
    def updateWeights(self, learningRate):
        if (self.regulizer[0] == "L1"):
            for layer in self.layers:
                #print("Original weight", layer.W.shape)

                layer.W -= learningRate * (layer.J_W + self.regulizer[1] * np.sign(layer.W))
                layer.B -= learningRate * layer.J_B.T
                #print("Adjusted to: ", layer.W.shape)
        
        elif (self.regulizer[0] == "L2"):
            for layer in self.layers:
                #print("Does this happen??")
                #print(layer.J_W)
                #a = 1/0
                layer.W -= learningRate * (layer.J_W + 2 * self.regulizer[1] * layer.W)
                layer.B -= learningRate * layer.J_B
        else:
            for layer in self.layers:

                layer.W -=learningRate * layer.J_W
                layer.B -= learningRate * layer.J_B
        



    # -----  Main functions of the network -----

    #Forward pass of a batch
    def forward(self, input, target):
        #Forwarding through layer-objects
        for layer in self.layers:
            input = layer.forward(input)

        if self.softMaxLayer:
            self.SoftmaxInn = input
            input = self.softMax(input)
            self.SoftmaxCache = input

        #Calculating loss       
        loss = self.Loss(input, target)
    
        return loss

    #Backward pass of a batch
    def backward(self, learningRate, target):
        #If softmax layer
        if(self.softMaxLayer):
            #Jacobian L -> S 
            J_LN = self.SoftmaxCache - target

            # Jacobian  S -> N
            dSoft = self.d_SoftMax(self.SoftmaxInn)

            J_LN = J_LN * dSoft

            
      
            #Backwarding through layer-objects
            for layer in reversed(self.layers):
                J_LN = layer.backWard(J_LN)
        
        #If no softmax layer 
        else: 
            lastLayer = self.layers[-1]

            error = lastLayer.output - target # TODO: THis depends on the loss function
            J_LN = error * lastLayer.dF_act(lastLayer.output)


            #Backwarding through layer-objects
            for layer in reversed(self.layers):
                J_LN = layer.backWard(J_LN)

        #Update weights and biases
        self.updateWeights(learningRate)


    
    def makeBatches(self, data):
        #Dividing the data into batches
        numSamples = data.shape[0]  # Number of samples is determined by the number of rows
        batches = []
        for start in range(0, numSamples, self.batchSize):
            end = min(start + self.batchSize, numSamples)  # Calculate end index for the batch
            batch = data[start:end, :]  # Slice the data to create a batch (rows for samples)
            batches.append(batch)
        return batches
    

    #Test of the network
    def test(self, data):
        #Forwarding through layer-objects
        for layer in self.layers:
            data = layer.forward(data)
        

        if self.softMaxLayer:
            data = self.softMax(data)
            #self.SoftmaxCache = input

        
        
        return np.argmax(data, axis=1)
    

