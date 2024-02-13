## TODO list for project

1. Implement forward ✅
   
2. Implemtn backward ✅

3. Implement visualizer module ✅

4. Image generator module ✅
   1. Specify types and number of them
   2. train, validate, test

5. Make Nettwork general✅
   1. choose loss✅
   2. choose act in layer✅
   3. gulrot innit weights✅

5.1 Test network🟨


X. Put it all toghter in ONE class
   X.I Train (+validate) 
   X.II predict
      X.II.I Visulize result 
         X.II.I.I Decode out to label?????
   
   X.IV. read images from file

1. Read paramters for network from file✅
   format: (numLayers, [numNeurons], [actFunc], lossFunc, (regulizer, regRate), softmax)



## Ideas for realtime plotting

import matplotlib.pyplot as plt

plt.ion()  # Turn on interactive mode
fig, ax = plt.subplots()  # Create a figure and axis object

losses = []  # A list to store loss values

for epoch in range(total_epochs):
    # Simulate epoch loss
    loss = compute_loss()  # You would replace this with your actual loss computation
    losses.append(loss)
    
    ax.clear()  # Clear the current plot
    ax.plot(losses)  # Plot the updated losses list
    ax.set_title("Real-time Loss Plot")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    plt.draw()
    plt.pause(0.1)  # Pause to allow the plot to be updated

plt.ioff()  # Turn off interactive mode
plt.show()






    ## ----- Network module -----

    #nettwork = NeuralNetwork(nettworkInfo)

    #make mini match for input target
    #input = np.array([np.random.rand(batchSize) for i in range(numNeurons)]).T
    #target = np.array([np.random.rand(batchSize) for i in range(targetSize)]).T
    #print(input)
    #print(target)
    #nettwork.forward(input, target)
    #nettwork.backward(0.01, target)
    #print(regulalizer[1])


    ## ----- Testing of visulizer module -----
    # images = doodler.gen_standard_cases(1, 5,5, decoder,[0.2,0.4], [0.2,0.4], 0.005, False, False, False)
    # print(images)

    #processor = Visualizer()
    # processor.drawImages(images,0)

    # loss_data = [
    # (np.random.rand(), np.random.rand()),  # Example tuple for batch 1
    # (np.random.rand(), np.random.rand()),  # Example tuple for batch 2
    # # Add more tuples as needed for each batch or epoch
    # ]

    # processor.drawLoss(loss_data)



    ## ----- Test of Image generator ----- 
    # gen = Generator()
    # data = gen.genarateData(10, 4, 4, 0.005)
    
    # processor.drawImages(data[0],0)
    # processor.drawImages(data[1],0)
    # processor.drawImages(data[2],0)

    ## ---- Defining Network parameters ----
    # batchSize = 4
    # imageSize = 2
    # targetSize = 9
    # layers = 3
    # #numNeurons = 3
    # numberOfNeurons = []
    # numberOfNeurons.append(imageSize**2)
    # numberOfNeurons = [imageSize**2] * (layers)
    # numberOfNeurons.append(targetSize)
    # #print(numberOfNeurons)


    # actFunction = ["Relu"] * layers
    # weightsInit = ["Glorot"] * layers
    # loss = "CrossEntropy"
    # softMaxLayer = 1
    # regulalizer = ("L2", 0.001)
    
    # nettworkInfo = (layers, numberOfNeurons, weightsInit, actFunction, loss, regulalizer, softMaxLayer, batchSize)


