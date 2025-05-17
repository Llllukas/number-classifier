from neuralnetwork import neuralNetworkAnalysis

import numpy as np

def validate(stopTime):
    #load data from test file
    data = np.loadtxt('MNIST_CSV/mnist_test.csv', delimiter=',', skiprows=1)

    #set up count variables
    correctCount = 0
    count = 0

    for row in data:
        print("Sample", count + 1)
        #find the correct label for the image
        trueLabel = row[0]

        #calculate the predicted label for the image
        predictLabel = neuralNetworkAnalysis(row)[0]


        print("Actual value:", trueLabel, "Prediction:", predictLabel)
        if trueLabel == predictLabel:
            correctCount += 1
        count += 1
        if count >= stopTime:
            break
    print(correctCount, "out of", count, "samples were correctly labeled")
    print(f"{100*correctCount/count}% accuracy")


if __name__ == "__main__":
    #the input is the number of samples to use (max is 10k)
    validate(100)