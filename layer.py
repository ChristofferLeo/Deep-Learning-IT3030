import numpy as np

class Layer:
    def __init__(self, numberOfNeurons):
        self.numberOfNeurons = numberOfNeurons

        #Wieght matrix
        self.W = np.array([np.random.rand(numberOfNeurons) for i in range(numberOfNeurons)])
        self.W = np.random.rand(self.W.shape[0], self.W.shape[1])

        #Bias vector
        self.B = np.array([np.random.rand(numberOfNeurons)])

        

    #Relu activation function TODO: This need to be modifiable
    def activationFunction(self, sum):
        return np.maximum(0, sum)
    
    def dActivation(self, sum):
        return np.where(sum > 0, 1, 0)
        
    
    ## -----  Main functions of the layer -----

    #Forward pass of a batch
    def forward(self, outputUpstream):
        #Storing outputUpstream for backpropagation
        self.cacheOutPrevious = outputUpstream

        #Pasing through weights and bias
        input = np.dot(self.W.T, outputUpstream) + self.B

        #Save input for backpropagation
        self.cacheInn = input
        self.cacheOutputUpstream = outputUpstream
        
        #Applying the activation function
        output = self.activationFunction(input)

        return output
    
    #Backward pass of a batch
    def backWard(self, J_LN, batchSize):
        # Find J Z -> Sum
        J_ZSum = np.zeros((batchSize, len(J_LN), len(J_LN)))
        for i in range(len(J_LN)):
            for j in range(len(J_LN)):
                if(i == j):
                    J_ZSum[i][j] = self.dActivation(self.cacheInn[i])
        
        #print(self.cacheOutputUpstream)
        for b in range(batchSize):
            J_ZW = np.outer(self.cacheOutputUpstream[b], np.diagonal(J_ZSum[0]))

        #Finding Weight Gradient
        self.J_LW = np.array([np.dot(J_ZW[i], J_LN) for i in range(len(J_LN))])
        
        #Find bias gradient
        self.J_LB = np.sum(J_LN, axis=0)
        
        #Find Jacobian upstream
        J_upStream = np.dot(J_ZW, J_LN)

        return J_upStream

