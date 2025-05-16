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
    print(input)


if __name__ == "__main__":
    with open("single_example.csv", "r") as csvfile:
        first_line = csvfile.readline()
    neuralNetwork(first_line)