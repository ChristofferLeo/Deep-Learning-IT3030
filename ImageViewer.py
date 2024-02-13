import matplotlib.pyplot as plt
import numpy as np



class Visualizer:
    def __init__(self):
        self.decoder = ['ball','ring','frame','box','flower','bar','polygon','triangle','spiral']


    def drawImages(self, data_tuple, prediction):
        # Extracting data
        images, labels = data_tuple
        num_images = len(images)
        imageSize = int(np.sqrt(images.shape[1]))

        print(labels)
        
        # Determine the grid size for subplots
        num_cols = int(np.ceil(np.sqrt(num_images)))
        num_rows = int(np.ceil(num_images / num_cols))
        
        plt.figure(figsize=(5 * num_cols, 5 * num_rows))

        # MAking subplots
        for i, image in enumerate(images):
            plt.subplot(num_rows, num_cols, i + 1)
            reshaped= image.reshape(imageSize,imageSize)
            plt.imshow(reshaped, cmap='gray', interpolation='nearest')
            plt.axis('off')
            if labels is not None and len(labels) > i:
                plt.title(" Prediction: " +  self.decoder[prediction[i]])

        #Putting it all togheter
        
        plt.tight_layout()
        plt.show()

    # Needs to be on format array of tuples (trainLoss, validationLoss)
    def drawLoss(self, lossData, validationLoss):

        #Extracting and putting into vector
        #trainLoss = [loss[0] for loss in lossData]
        #validationLoss = [loss[1] for loss in lossData]

        #Plotting
        plt.figure(figsize=(10, 6))
        plt.plot(lossData, label='Train Loss')
        plt.plot(validationLoss, label='Validation Loss')
        plt.title("Training and validation loss")
        plt.xlabel("Batch")
        plt.ylabel("Loss")
        plt.legend()
        #plt.show()
            

