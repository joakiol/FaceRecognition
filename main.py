import functions as fn
import import_data as id
import numpy as np
import matplotlib.pyplot as plt

def main():

    # Import data, ORL is from getFaceData, and Yale is from getDarkFaceData
    #trainX, trainY, testX, testY = id.getFaceData()
    trainX, trainY, testX, testY = id.getDarkFaceData()


    # Perform light normalization
    #sample = fn.normalizeData([trainX[9]], 192, 168, 10)
    #trainX = fn.normalizeData(trainX, 192, 168, 10)
    #testX = fn.normalizeData(testX, 192, 168, 10)


    # Plot normal vs normalized face
    #fig, axs = plt.subplots(1, 2, figsize=(8, 4))
    #axs[0].imshow(trainX[9].reshape((192,168)), cmap='gray')
    #axs[1].imshow(sample[0].reshape((192, 168)), cmap='gray')
    #plt.savefig("Normalization.pdf")
    #plt.show()

    # Perform PCA to get B (here PC is B)
    #PC = fn.getPC(trainX)

    # X-axis with iterations for errorplot
    #x = np.linspace(1, 200, 200)
    #x = np.linspace(1, 50, 50)
    #x = [1,2,3,4,5,7,10,15,20,30,40,50,75,100,150,200,250,300,400,500,600,700,800,900,1000,1500,2000,2500,3000,4000,5000,6000,7000,8000,9000,10000,10304]


    # Error plots that give a singe curve
    #fn.errorPlot(x, trainX, trainY, testX, testY, PC, type="eigenface")
    #fn.errorPlot(x, trainX, trainY, testX, testY, PC, type="fisherface")

    # Error plot that give a curve for each number of k. Takes original data as input.
    fn.errorNorm(trainX, trainY, testX, testY, "fisherface", [3,5,8,10,15,20,30])



main()
