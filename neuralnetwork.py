import numpy as np
import csv

#number of hidden layer neurons
Hidden = 600

#number of features (input neurons)
Feature = 784 # 28 x 28

#RELU R^600 -> R^600
def relu(x):
    for i, value in enumerate(x):
        if value < 0:
            x[i] = 0
    return x
            
    
#jacobian of RELU (derivative of i-th entry with respect to j-th entry of V2) R^600 x I x I -> R
def delRelu(V2,i,j):
    if i == j and V2[i] >= 0:
        return V2[i]
    else:
        return 0 

#softmax R^10 -> R^10
def softmax(x):
    out = []
    for i in range(len(x)):
        row = [(v-x[i]) for v in x]
        expon = [(np.exp(v)) for v in row]
        out.append(1/(np.sum(expon)))
    return out
    


#jacobian of softmax (derivative of i-th entry with respect to j-th entry of V3) R^10 x I x I -> R
def delSoftmax(V3,i,j):
    expon = [np.exp(i) for i in V3]
    norm = np.sum(expon)    
    if i == j:
        return ((expon[i]/norm) - ((expon[i]**2)/(norm**2)))
    else:
        return -(expon[i]*expon[j])/(norm**2)


#activation function
def activ(x):
    return relu(x)

#derivative of activation function !it is hard-coded in gradientDescent.py that the jacobian is a diagonal matrix!
def delActiv(x,i,j):
    return delRelu(x,i,j)
    
#output activation function
def outActiv(x):
    return softmax(x)

#derivative of output activation function
def delOutActiv(x,i,j):
    return delSoftmax(x,i,j)

#neural network function !output the value at every node including hidden layer! interpretation will be done seperately
#input a single vector (x_1, ... x_N)
#output is tuple of (V_2, V_3) where V_2 is the 600-dimensional vector of the values at the hidden layer nodes and V_3 is the 10-dimensional prob. vector
def neuralNetwork(input):
    #load files
    w1 = np.loadtxt('w1.csv', delimiter=',')
    w2 = np.loadtxt('w2.csv', delimiter=',')    

    #turn input string into vector
    if isinstance(input, np.ndarray):
        Values1 = input
    else:
        Values1 = np.fromstring(input, sep=',')

    #add a 1 to the front of the vector, !remember the first entry of the vector is the label!
    Values1[0] = 1
    
    #Multiply Values1 by w1 matrix to get Values2, proof: trivial
    Values2 = np.dot(Values1, w1)

    #put activation function on Values2
    activ2 = activ(Values2)

    #prepend 1 to activated Hidden layer values vector
    activ2 = np.insert(activ2, 0, 1)
    
    #get the output values from activ2 and the w2 weights
    Values3 = np.dot(activ2,w2)

    #commented out
    #Output = outActiv(Values3)
    #print(Output)

    return (Values2, Values3)
    

#uses neuralNetwork function to output the chances of each number and the maximum (prediction)
def neuralNetworkAnalysis(input):
    #get output node values from neural network
    Output = neuralNetwork(input)[1]
    
    #softmax output
    Chances = outActiv(Output)

    #find maximum
    Prediction = np.argmax(Chances)

    #print
    #print(Prediction)
    print(' | '.join(f'{i}: {round(100*x,1)}' for i, x in enumerate(Chances)))

    return (Prediction,Chances)


if __name__ == "__main__":
    with open("single_example.csv", "r") as csvfile:
        first_line = csvfile.readline()
    print(neuralNetwork(first_line))
    neuralNetworkAnalysis(first_line)