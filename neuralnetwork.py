import numpy as np
import csv

#number of hidden layer neurons
Hidden = 600

#number of features (input neurons)
Feature = 784 # 28 x 28

#RELU
def relu(x):
    if x >= 0:
        return x
    else:
        return 0
    
#derivative of RELU
def delrelu(x):
    if x >= 0:
        return 1
    else:
        return 0
    

#activation function
def activ(x):
    return relu(x)

#derivative of activation function
def delActiv(x):
    return delrelu(x)
    
#output activation function
def outActiv(x):
    return relu(x)

#derivative of output activation function
def delOutActiv(x):
    return delrelu(x)

#neural network function !output the value at every node including hidden layer! interpretation will be done seperately
#input a single vector (x_1, ... x_N)
#output is tuple of (V_2, V_3) where V_2 is the 600-dimensional vector of the values at the hidden layer nodes and V_3 is the 10-dimensional prob. vector
def neuralNetwork(input):
    #load files
    w1 = np.loadtxt('w1.csv', delimiter=',')
    w2 = np.loadtxt('w2.csv', delimiter=',')    

    #turn input string into vector
    Values1 = np.fromstring(input, sep=',')

    #add a 1 to the front of the vector, !remember the first entry of the vector is the label!
    Values1[0] = 1
    
    #Multiply Values1 by w1 matrix to get Values2, proof: trivial
    Values2 = np.dot(Values1, w1)

    #put activation function on Values2
    activ2 = [activ(i) for i in Values2]

    #prepend 1 to activated Hidden layer values vector
    activ2 = np.insert(activ2, 0, 1)
    
    #get the output values from activ2 and the w2 weights
    Values3 = np.dot(activ2,w2)

    #commented out
    #Output = [outActiv(i) for i in Values3]
    #print(Output)

    return (Values2, Values3)
    

if __name__ == "__main__":
    with open("single_example.csv", "r") as csvfile:
        first_line = csvfile.readline()
    neuralNetwork(first_line)
