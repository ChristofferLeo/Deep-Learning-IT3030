import doodler as d
import numpy as np


class Generator:
    def __init__(self):
        self.types = ['ball','ring','frame','box','flower','bar','polygon','triangle','spiral']
        #self.types = ['ball','ring','frame']
    # Return list of three sets of data
    # Each on format (images, [[targets]], [targetValue], (size,size) )
    def genarateData(self, numberOfImages, width, height, noise):
        data = d.gen_standard_cases(numberOfImages, width, height, self.types, [0.2, 0.4], [0.2, 0.4], noise, True, False, True) 
    
        #Reformatting
        images, targets, labels, dims, flat = data
        samples = images
        targets = targets
        target_labels = np.array(labels)
        
        # Preparing to split
        length = len(data[0])
        trainEnd = int(length * 0.79)
        validationEnd = int(length * 0.99)

        # Shuffle indices
        indices = np.arange(length)
        np.random.shuffle(indices)

        # Splitting indices
        trainIdx = indices[:trainEnd]
        validateIdx = indices[trainEnd:validationEnd]
        testIdx = indices[validationEnd:]

        #Splitting data
        samples_train = samples[trainIdx, :]
        targets_train = targets[trainIdx, :]
        target_labels_train = target_labels[trainIdx]

        samples_val = samples[validateIdx, :]
        targets_val = targets[validateIdx, :]
        target_labels_val = target_labels[validateIdx]

        samples_test = samples[testIdx, :]
        targets_test = targets[testIdx, :]
        target_labels_test = target_labels[testIdx]

        #Returning the three sets
        trainData = (samples_train, targets_train, target_labels_train)
        validationData = (samples_val, targets_val, target_labels_val)
        testData = (samples_test, targets_test, target_labels_test)

        result = [trainData, validationData, testData]

        return result
