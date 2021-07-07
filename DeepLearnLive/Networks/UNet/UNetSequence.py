import numpy
import math
import os
import gc
import cv2
import pandas
from tensorflow.keras.utils import Sequence
import tensorflow
import girder_client
from sklearn.utils import shuffle
import random
import numpy as np



class UNetSequence(Sequence):

    def __init__(self,datacsv,indexes,batchSize,labelName,gClient = None,tempFileDir = None,shuffle=True,augmentations = True):
        # author Rebecca Hisey
        if "GirderID" in datacsv.columns:
            self.gClient = gClient
            self.tempFileDir = tempFileDir
            self.inputs = numpy.array([self.downloadGirderData(x,datacsv) for x in indexes])
        else:
            self.inputs = numpy.array([os.path.join(datacsv["Folder"][x], datacsv['FileName'][x]) for x in indexes]) #replace with FileName
        self.batchSize = batchSize
        self.labelName = labelName
        self.targets = numpy.array([os.path.join(datacsv["Folder"][x], datacsv[labelName][x]) for x in indexes]) #Remove US (and above)
        self.shuffle = shuffle
        self.augmentations = augmentations
        self.on_epoch_end()

    def on_epoch_end(self):
        if self.shuffle:
            shuffledInputs,shuffledTargets = shuffle(self.inputs,self.targets)
            self.inputs = shuffledInputs
            self.targets = shuffledTargets

    def __len__(self):
        # author Rebecca Hisey
        length = len(self.inputs) / self.batchSize
        length = math.ceil(length)
        return length

    ##################################
    #Augmentation methods:

    def rotateImage(self,image,angle):
        center = tuple(numpy.array(image.shape[1::-1])/2)
        rot_mat = cv2.getRotationMatrix2D(center,angle,1.0)
        rotImage = cv2.warpAffine(image,rot_mat,image.shape[1::-1],flags=cv2.INTER_LINEAR)
        return rotImage

    #Performs a random flip across the specified axis
    def random_flip(self, image, axis):
        if random.choice([0,1]):
            return cv2.flip(image, axis)
        return image

    ##################################

    def readImage(self,file):
        image = cv2.imread(file)
        resized_image = cv2.resize(image, (224, 224))
        normImg = cv2.normalize(resized_image, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        preprocessingMethod = random.randint(0, 3)

    def readUSImage(self, file):
        # print("us image: " + str(file))
        image = cv2.imread(file, 0)
        resized = cv2.resize(image, (128, 128)).astype(numpy.float16)
        scaled = resized / resized.max()
        return scaled[...,numpy.newaxis]

    def readSegImage(self, file):
        # print("seg image: " + str(file))
        image = cv2.imread(file, 0)
        resized = cv2.resize(image, (128, 128))
        # print("max resized: " + str(max(resized)))
        if np.amax(resized) !=0:
            return resized[...,numpy.newaxis] / np.amax(resized)
        else:
            return resized[...,numpy.newaxis]

    def downloadGirderData(self,index,datacsv):
        # tempFileDir is a folder in which to temporarily store the files downloaded from Girder
        # by default the temporary folder is created in the current working directory, but this can
        # be modified as necessary
        if not os.path.isdir(self.tempFileDir):
            os.mkdir(self.tempFileDir)
        fileID = datacsv["GirderID"][index]
        fileName = datacsv["FileName"][index]
        numFilesWritten = 0
        if not os.path.isfile(os.path.join(self.tempFileDir, fileName)):
            self.gClient.downloadItem(fileID, self.tempFileDir)
            numFilesWritten += 1
            if numFilesWritten % 100 == 0:
                print(numFilesWritten)
        return(os.path.join(self.tempFileDir, fileName))

    def __getitem__(self,index):
        # author Rebecca Hisey
        startIndex = index*self.batchSize
        indexOfNextBatch = (index + 1)*self.batchSize
        inputBatch = numpy.array([self.readUSImage(x) for x in self.inputs[startIndex : indexOfNextBatch]])
        outputBatch = numpy.array([self.readSegImage(x) for x in self.targets[startIndex : indexOfNextBatch]])
        outputBatch = tensorflow.keras.utils.to_categorical(outputBatch, 2)
        print("input shape: " + str(inputBatch.shape))
        print("output shape: " + str(outputBatch.shape))
        return (inputBatch,outputBatch)