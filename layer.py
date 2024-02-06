import numpy as np

class Layer:
    # Initiate the layer
    def __init__(self, numberOfNeurons, inputSize, batchSize):
        self.numberOfNeurons = numberOfNeurons
        self.batchSize = batchSize
        self.inputSize = inputSize

        #Wieght matrix
        self.W = np.array([np.random.rand(numberOfNeurons) for i in range(inputSize)])
        self.W = np.random.rand(self.W.shape[0], self.W.shape[1])

        #Bias vector 
        columnB = np.random.rand(numberOfNeurons)   
        self.B = np.tile(columnB, (batchSize, 1)).T

    # Relu activation function 
    def F_act(self, sum):
        return np.maximum(0, sum)
    
    def dF_act(self, sum):
        return np.where(sum > 0, 1, 0)
        
    

    ## -----  Main functions of the layer -----

    #Forward pass of a batch
    def forward(self, outputUpstream):
        #Pasing through weights and bias for each batch
        input = np.dot(self.W.T, outputUpstream) + self.B #Output for each batch along Column
        
        #Applying the activation function
        output = self.F_act(input)

        #Save input before and after weights
        self.cacheInn = input
        self.cacheOutPrevious = outputUpstream.T
        self.output = output.T


        return output
    
    #Backward pass of a batch
    def backWard(self, J_LN):

        #Find J Z -> SUM
        J_ZSUM = self.dF_act(self.cacheInn) #NOTE: This is a vector

        # --- Finding weight and bias gradients ---
        #Finding delta N
        deltaN = self.dF_act(self.cacheOutPrevious) * J_LN

        # Finding weight and bias gradients
        self.J_W = np.dot(deltaN.T, self.cacheOutPrevious)
        self.J_B = np.sum(deltaN, axis=1)

        # --- Computing jacobian Upstream ---
        # Getting J_ZSUM on right matrix form
        temp = np.zeros((self.batchSize, self.numberOfNeurons, self.numberOfNeurons))
        for b in range(self.batchSize):
            np.fill_diagonal(temp[b], J_ZSUM[:,b])
        
        J_ZSUM = temp

        # Find J Z -> Y
        J_ZY = np.array([np.dot(J_ZSUM[b], self.W.T) for b in range(self.batchSize)])

        # #Find Jacobian upstream
        J_upStream = np.array([np.dot(J_LN[b], J_ZY[b]) for b in range(self.batchSize)])

        return J_upStream

