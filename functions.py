import numpy as np
import matplotlib.pyplot as plt
import time
from sklearn.preprocessing import normalize
from sklearn.metrics import confusion_matrix


def normalizeData(data, reshape1, reshape2, size):

    # New list to store light-normalized data
    newdata=np.zeros((len(data), len(data[0])))

    #For each image, normalize and add to new matrix
    for i in range(len(data)):
        print("Normalizing ", i+1)
        newdata[i] = lightNormalize(data[i], reshape1, reshape2, size)

    # Return normalized images
    return newdata

def lightNormalize(image, reshape1, reshape2, size):

    # Reshape image to matrix, and make matrices to store image integrals
    image = image.reshape((reshape1,reshape2))
    sumImage = np.zeros((reshape1, reshape2))
    sumImageSquared = np.zeros((reshape1, reshape2))

    # Image integral and squared image integral
    sumImage[0][0] = image[0][0]
    sumImageSquared[0][0] = image[0][0] ** 2

    for i in range(1,reshape1):
        sumImage[i][0] = sumImage[i - 1][0] + image[i][0]
        sumImageSquared[i][0] = sumImageSquared[i - 1][0] + image[i][0] ** 2

    for i in range(1,reshape2):
        sumImage[0][i] = sumImage[0][i - 1] + image[0][i]
        sumImageSquared[0][i] = sumImageSquared[0][i - 1] + image[0][i] ** 2

    for i in range(1,reshape1):
        for j in range(1,reshape2):
            sumImage[i][j] = sumImage[i-1][j]+sumImage[i][j-1]+image[i][j]-sumImage[i-1][j-1]
            sumImageSquared[i][j] = sumImageSquared[i - 1][j] + sumImageSquared[i][j - 1] + \
                                    image[i][j]**2 - sumImageSquared[i - 1][j - 1]

    # normalized stores f(w(x,y)), windowAppearences sums up appearences for each pixel, used
    # to average the pixels windows appearences in the end
    normalized = np.zeros((reshape1, reshape2))
    windowAppearences = np.zeros((reshape1, reshape2))

    #For each window position
    for i in range(reshape1-size+1):
        for j in range(reshape2-size+1):

            # Calculates the sum and squared sum of elements in window
            if j > 0 and i > 0:
                S1 = (sumImage[i+size-1][j+size-1]-sumImage[i-1][j+size-1]-sumImage[i+size-1][j-1]+sumImage[i-1][j-1])
                S2 = (sumImageSquared[i+size-1][j+size-1]-sumImageSquared[i-1][j+size-1]-sumImageSquared[i+size-1][j-1]+sumImageSquared[i-1][j-1])
            elif j > 0 and i == 0:
                S1 = (sumImage[i+size-1][j+size-1]-sumImage[i+size-1][j-1])
                S2 = (sumImageSquared[i+size-1][j+size-1]-sumImageSquared[i+size-1][j-1])
            elif j == 0 and i > 0:
                S1 = (sumImage[i+size-1][j+size-1]-sumImage[i-1][j+size-1])
                S2 = (sumImageSquared[i + size - 1][j + size - 1] - sumImageSquared[i - 1][j + size - 1])
            else:
                S1 = sumImage[i+size-1][j+size-1]
                S2 = sumImageSquared[i + size - 1][j + size - 1]

            #Calculates n, window mean and window variance
            n=size**2
            mu = S1/n
            # Taking max of var and a small number, since the window variance in some positions with small
            # size seems to give zero variance.
            var = np.maximum(1/n * (S2 - (S1**2)/n), 1e-5)

            #Adds f(w(x,y)) to normalized[x,y], and adds one to each pixel appearing in window
            normalized[i:i+size, j:j+size] += (image[i:i+size, j:j+size]-mu)/np.sqrt(var)
            windowAppearences[i:i+size, j:j+size] += 1

    # Divides normalized[x,y] by number of appearences for pixel (x,y)
    normalized /= windowAppearences

    # Return as vector
    return normalized.reshape(-1)

def getPC(trainData, smart=True, timer=False, save=False, load=False):

    # Possible to save and load the B-matrix since it takes a long time to calculate
    if load:
        eigVec = np.load("longPC.npy")
        return eigVec

    # To compare the computation time of the hard or smart way
    if timer:
        startTime = time.time()

    # Constants
    n=len(trainData)
    dim=len(trainData[0])

    # Calculates mu, A, and At. trainData seems to be transposed compared to the class slides, so
    # A/At is calculated different to obtain same size ad in class slides. A is d x n.
    trainDataMu = np.sum(trainData, axis=0) / n
    At = trainData - trainDataMu
    A = np.transpose(At)

    # Follows the steps described as the 2. alternative in report
    if smart:
        C = 1 / dim * np.matmul(At, A)
        eigVal, eigVec = np.linalg.eig(C)
        eigVec = np.matmul(A, eigVec)
        eigVec = normalize(eigVec, norm='l2', axis=0)

    # Follows the steps described as the 1. alternative in report
    else:
        C = 1/n*np.matmul(A,At)
        eigVal, eigVec = np.linalg.eig(C)

    # To compare the computation time of the hard or smart way
    if timer:
        endTime=time.time()
        print("Time of PCA: ", endTime-startTime)

    # Possible to save and load the B-matrix since it takes a long time to calculate
    if save:
        np.save("longPC.npy", eigVec)

    # eigVec is B in report.
    return eigVec

def getLDA(trainDataX, trainDataY):

    # Calculate total class mean
    totalMean = np.sum(trainDataX, axis=0)/len(trainDataX)

    #Calculate within-class means, as well as numbers of samples for each class
    classes = np.unique(trainDataY)
    classMean = np.zeros((len(classes), len(trainDataX[0])))
    numberClass = np.zeros(len(classes))
    for i in range(len(classes)):
        numberClass[i] = len(np.where(trainDataY==classes[i])[0])
        classMean[i] = np.sum(trainDataX[np.where(trainDataY==classes[i])], axis=0)/numberClass[i]

    # Calculates Sb and Sw
    Sb = np.zeros((len(trainDataX[0]), len(trainDataX[0])))
    Sw = np.zeros((len(trainDataX[0]), len(trainDataX[0])))

    for i in range(len(classes)):

        Sb += (numberClass[i]*np.outer(classMean[i]-totalMean, classMean[i]-totalMean))

        XMinusClassMean = trainDataX[np.where(trainDataY==classes[i])]-classMean[i]
        for j in XMinusClassMean:
            Sw += np.outer(j, j)

    # Calculate C
    C = np.matmul(np.linalg.inv(Sw), Sb)

    # Eigvec of C is B in fisherface
    eigVal, eigVec = np.linalg.eig(C)

    return eigVec



def transform(trainData, testData, eigenVectors, numDims):

    # Trainsforms the train and test data using the first numDims columns of eigenVectors
    newTrainData = np.matmul(trainData, eigenVectors[:,0:numDims])
    newTestData = np.matmul(testData, eigenVectors[:, 0:numDims])

    return newTrainData, newTestData


def nearestNeighbur(trainX, testX, trainY):

    predictionList = np.zeros(len(testX))

    # For each test sample, calculates the distance to all train samples, and predicts as the closest.
    for i in range(len(testX)):
        difference = np.sum((trainX - testX[i]) ** 2, axis=1)
        predictionList[i] = trainY[np.argmin(difference)]

    return predictionList

def confAndError(predicted, real):
    confMatrix = confusion_matrix(predicted, real)
    error=1-np.trace(confMatrix)/np.sum(confMatrix)
    return confMatrix, error

def errorNorm(trainX, trainY, testX, testY, type, k):

    # Plot stuff
    x = np.linspace(1, 50, 50)
    plt.figure(figsize=(8, 4))
    plt.title(str(type)+" with Yale database with light normalization")
    plt.xlabel("d'")
    plt.ylabel("error")

    # For each window size: Do light normalizetion, Calculate B/PC and plot then plot error as function of d'
    for i in k:
        normalizedTrain = normalizeData(trainX, 192, 168, i)
        normalizedTest = normalizeData(testX, 192, 168, i)
        PC = getPC(normalizedTrain)
        errorPlot(x, normalizedTrain, trainY, normalizedTest, testY, PC, type, windowSize=i)

    plt.legend()
    plt.savefig("Yale_"+str(type)+"_norm.pdf")


def errorPlot(x, trainX, trainY, testX, testY, PC, type, windowSize=0):

    # List to store errors
    errors = np.zeros(len(x))

    # Number of classes
    classes = len(np.unique(trainY))

    # For each dimension number d'.
    for i in range(len(x)):

        print("d = ", int(x[i]))

        # Transforms train and test data according to eigenface-algorithm.
        if type=="eigenface":
            transformedTrain, transformedTest = transform(trainX, testX, PC, int(x[i]))

        # First transforms train/test to reduced dimension space, then calculates LDA-space
        # and transforms train and test data according to fisherface-algorithm.
        elif type=="fisherface":
            newTrainData, newTestData = transform(trainX, testX, PC, int(x[i]))
            LDASpace = getLDA(newTrainData, trainY)
            transformedTrain, transformedTest = transform(newTrainData, newTestData, LDASpace, classes-1)

        else:
            print("Type should be 'eigenface' or 'fisherface'.")
            return

        # Uses nearest neighbour to predict test data, then calculate test error.
        predictionList = nearestNeighbur(transformedTrain, transformedTest, trainY)
        confMatrix, error = confAndError(predictionList, testY)
        errors[i] = error

    # Prints
    print("Minimum error", min(errors))
    print("Minimum error using d' = ", x[np.argmin(errors)])

    # Plots
    if windowSize>0:
        plt.plot(x, errors, label="k="+str(windowSize))

    else:
        plt.figure(figsize=(8,4))
        plt.title("Fisherface with Yale database")
        plt.xlabel("d'")
        plt.ylabel("error")
        plt.plot(x, errors)
        plt.show()
        #plt.savefig("Yale_fisherface.pdf")