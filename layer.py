import numpy as np

class Layer:
    # Initiate the layer
    def __init__(self, numberOfNeurons, inputSize, learningRate, batchSize, actFunc = "Relu", weightInit = "Random"):
        self.numberOfNeurons = numberOfNeurons
        self.batchSize = batchSize
        self.inputSize = inputSize
        self.actFunc = actFunc
        self.learningRate = learningRate
        self.W = 0
        self.B = 0

        #Init weights
        self.initWeights(weightInit)

    #Init weights
    def initWeights(self, weightInit):
        #Init weights
        if weightInit == "Random":
            #Wieght matrix
            print("Layer using: Random" + "With nuerons: ", self.numberOfNeurons, " input: ", self.inputSize)
            self.W = np.array([np.random.rand(self.numberOfNeurons) for i in range(self.inputSize)])
            self.W = np.random.randn(self.W.shape[0], self.W.shape[1])

        
        # Glorot weight innit
        elif weightInit == "Glorot":
            print("Layer using: Random" + "With nuerons: ", self.numberOfNeurons, " input: ", self.inputSize)
            limit = np.sqrt(6 / (self.inputSize + self.numberOfNeurons))
            self.W = np.random.uniform(-limit, limit, size=(self.inputSize, self.numberOfNeurons))
            #print(self.W)
        else:
            print("Layer was not innitlized sucessfully")
            assert False
        #Bias vector 
        columnB = np.ones(self.numberOfNeurons)   
        self.B = np.tile(columnB, (self.batchSize, 1))


    # Relu activation function 
    def F_act(self, sum):
        if self.actFunc == "Relu":
            return np.maximum(0, sum)
        elif self.actFunc == "Sigmoid":
            return 1 / (1 + np.exp(-np.clip(sum, -700, 700)))
        elif self.actFunc == "Tanh":
            return np.tanh(sum)
        elif self.actFunc == "Linear":
            return sum
    
    def dF_act(self, sum):
        if self.actFunc == "Relu":
            return np.where(sum > 0, 1, 0)
        elif self.actFunc == "Sigmoid":
            temp = self.F_act(sum)
            return temp * (1 - temp)
        elif self.actFunc == "Tanh":
            return 1 - np.power(self.F_act(sum), 2)
        elif self.actFunc == "Linear":
            return np.ones(sum.shape)
        
    

    ## -----  Main functions of the layer -----

    #Forward pass of a batch
    def forward(self, outputUpstream):

        #Pasing through weights and bias for each batch
        input = np.dot(outputUpstream, self.W) + self.B 

        #Applying the activation function
        output = self.F_act(input)

        #Save input before and after weights
        self.cacheInn = input
        self.cacheOutPrevious = outputUpstream
        self.output = output
        return output
    
    #Backward pass of a batch
    def backWard(self, J_LN):

        #Find J Z -> SUM
        J_ZSUM = self.dF_act(self.cacheInn) 


        # Finding delta for layer N
        deltaN = J_LN * J_ZSUM

        # Finding weight and bias gradients
        self.J_W = np.dot(self.cacheOutPrevious.T, deltaN) 
        self.J_B = np.sum(deltaN, axis=0) 

        # Computing Jacobian upstream
        J_UP = np.dot(deltaN, self.W.T)
    
        return J_UP
    


