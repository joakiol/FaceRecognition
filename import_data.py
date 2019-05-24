import numpy as np
import matplotlib.image as mpimg

def getFaceData():

    # Image data
    pixels = 112*92
    trainImages = 200
    testImages = 200
    persons=40
    trainfaces=5
    testfaces=5


    trainX=np.ndarray((trainImages, pixels))
    trainY=np.ndarray(trainImages)

    # Logic to read images as they are saved in folder
    for i in range(persons):
        for j in range(trainfaces):
            img=mpimg.imread('Images/s'+str(i+1)+'/'+str(j+1)+'.pgm')
            trainX[i*trainfaces+j]=np.asarray(img).reshape(-1)
            trainY[i*trainfaces+j]=i+1

    testX=np.ndarray((testImages, pixels))
    testY=np.ndarray(testImages)
    for i in range(persons):
        for j in range(testfaces):
            img=mpimg.imread('Images/s'+str(i+1)+'/'+str(j+1+trainfaces)+'.pgm')
            testX[i*testfaces+j]=np.asarray(img).reshape(-1)
            testY[i*testfaces+j]=i+1

    return trainX, trainY, testX, testY

def getDarkFaceData():

    # Image data
    pixels = 192*168
    trainImages = 50
    testImages = 50
    persons=10
    trainfaces=5
    testfaces=5

    trainX=np.ndarray((trainImages, pixels))
    trainY=np.ndarray(trainImages)

    # Logic to get saved images
    for i in range(persons):
        for j in range(trainfaces):
            img=mpimg.imread('Images/d'+str(i+1)+'/'+str(j+1)+'.pgm')
            trainX[i*trainfaces+j]=np.asarray(img).reshape(-1)
            trainY[i*trainfaces+j]=i+1

    testX=np.ndarray((testImages, pixels))
    testY=np.ndarray(testImages)
    for i in range(persons):
        for j in range(testfaces):
            img=mpimg.imread('Images/d'+str(i+1)+'/'+str(j+1+trainfaces)+'.pgm')
            testX[i*testfaces+j]=np.asarray(img).reshape(-1)
            testY[i*testfaces+j]=i+1

    return trainX, trainY, testX, testY


