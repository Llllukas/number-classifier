'''
FILE STRUCTURE:

!the values being taken from the files w1.csv and w2.csv is hard-coded!

where it says {m} it means there are two seperate functions for m = 1 and m = 2

gradient{m}(training): R^N X I -> R^(l x k) 
uses neuralNetwork(Training,m) to compute the gradient (l x k vector) of the error function of w{m}.csv 
iterating over l and k to compute the entire gradient matrix

sumGradient{m}(samples): N -> R^(l x k)
uses gradient{m}(training) to sum over all training samples for w{m}.csv up to the number of samples (the input determines when to stop)

updateValues(batchSize)
uses sumGradient for each m to change w{m}.csv
use 


Missing:
gradient2
updateValues (+ documentation)
'''

from neuralnetwork import activ
from neuralnetwork import delActiv
from neuralnetwork import outActiv
from neuralnetwork import delOutActiv
from neuralnetwork import neuralNetwork
from neuralnetwork import Hidden      #amount of hidden layer neurons H
from neuralnetwork import Feature     #amount of input layer neurons

import numpy as np
from itertools import product

#learning rate for gradient descent (rho or eta)
LearningRate = 0.01

#convergence threshold
ConvergenceThreshold = 0.01


def sumGradient1(samples):

    #load w's to use for each gradient in this iteration
    w1 = np.loadtxt('w1.csv', delimiter=',')
    w2 = np.loadtxt('w2.csv', delimiter=',') 

    sum = 0
    with open("MNIST_CSV/mnist_train.csv", "r") as file:
        for number, training in enumerate(file, 1):  
            if number > samples:
                break
            sum += gradient1(training,w1,w2)
            print("sample", number)
    return sum

#calculate gradient with respect to w2.csv (w2.csv is input when using neuralNetwork())
def gradient2(training):

    #calculate node values
    nodes = neuralNetwork(training)

    #extract V2 and V3
    V2 = nodes[0]
    V3 = nodes[1]

    #matrix of values where delAlpha[i][j] corresponds to δα_i(V³)/δV³_j this is used in the formula for dE/dw
    #why do we call this before the loop? notice in the formula for dE/dw we use every single combination of i,j for every index l,k. this way we then avoid some computation
    delAlpha = [[delOutActiv(V3,i,j) for j in range(10)] for i in range(10)]

    #calculate activation function of nodes
    activatedV3 = outActiv(V3)
    activatedV2 = activ(V2)

    #calculate hot-one vector of training sample
    hotTraining = np.zeros(10)
    if isinstance(training, np.ndarray):
        hotTraining[training[0]] = 1
    else:
       hotTraining[int(training[:1])] = 1 

    #calculating the gradient entries for w2.csv !not w1.csv!

    #create matrix to put entries into:
    gradientMatrix = np.zeros((Hidden+1,10))
    for l,k in product(range(10),range(Hidden + 1)):
        #edit the b_k
        if k == 0:
            
            #calculate derivative vector dV³_j/db_k iterated over j
            dV3 = [indicatorFunc(l,j) for j in range(10)]
            
            #calculate vector of sum_j delAlpha_i(V³)/delV³_j * dV³_j/dw_lk iterated over i
            delSumVector = [np.dot(delAlpha[i],dV3) for i in range(10)]

            #calculate dE(w)/dw_lk by dot multiplying (alpha_i(V³) - t_n) by delSumVector !check paper to notice this is a dot product over i!
            dE = np.dot((activatedV3 - hotTraining), delSumVector)

            gradientMatrix[k][l] = dE
        
        #edit the w_lk
        else:
            
            #calculate derivative vector dV³_j/db_k iterated over j
            dV3 = [indicatorFunc(l,j)*activatedV2[k-1] for j in range(10)]
            
            #calculate vector of sum_j delAlpha_i(V³)/delV³_j * dV³_j/dw_lk iterated over i
            delSumVector = [np.dot(delAlpha[i],dV3) for i in range(10)]

            #calculate dE(w)/dw_lk by dot multiplying (alpha_i(V³) - t_n) by delSumVector !check paper to notice this is a dot product over i!
            dE = np.dot((activatedV3 - hotTraining), delSumVector)

            gradientMatrix[k][l] = dE
        
    return gradientMatrix
            



#calculate gradient with respect to w1.csv (w1.csv is input when using neuralNetwork())
def gradient1(training,w1,w2):

    #reformat training to an array
    if not isinstance(training, np.ndarray):
        training = np.array([int(x) for x in training.split(',')])   

    #calculate node values
    nodes = neuralNetwork(training)

    #extract V2 and V3
    V2 = nodes[0]
    V3 = nodes[1]

    #matrix of values where delAlpha[i][j] corresponds to δα_i(V³)/δV³_j this is used in the formula for dE/dw
    #why do we call this before the loop? notice in the formula for dE/dw we use every single combination of i,j for every index l,k. this way we then avoid some computation
    delAlpha = [[delOutActiv(V3,i,j) for j in range(10)] for i in range(10)]

    #calculate activation function of nodes
    activatedV3 = outActiv(V3)
    activatedV2 = activ(V2)
    
    #derivative of h with respect to same index
    delH = [delActiv(V2,l,l) for l in range(Hidden)]

    #calculate hot-one vector of training sample
    hotTraining = np.zeros(10)
    if isinstance(training, np.ndarray):
        hotTraining[training[0]] = 1
    else:
        hotTraining[int(training[:1])] = 1 

    #calculating the gradient entries for w2.csv !not w1.csv!

    #create matrix to put entries into:
    gradientMatrix = np.zeros((Feature + 1,Hidden))
    for l,k in product(range(Hidden),range(Feature + 1)):
        #edit the b_k
        if k == 0:
            
            #calculate derivative vector dV³_j/db_k iterated over j
            dV3 = [w2[l][j] * delH[l] for j in range(10)]
            
            #calculate vector of sum_j delAlpha_i(V³)/delV³_j * dV³_j/dw_lk iterated over i
            delSumVector = [np.dot(delAlpha[i],dV3) for i in range(10)]

            #calculate dE(w)/dw_lk by dot multiplying (alpha_i(V³) - t_n) by delSumVector !check paper to notice this is a dot product over i!
            dE = np.dot((activatedV3 - hotTraining), delSumVector)

            gradientMatrix[k][l] = dE
        
        #edit the w_lk !we can skip this and see that it is 0 if either delH[l] is zero or if training[k] is zero!
        elif delH[l] != 0 and training[k] != 0:
            
            #calculate derivative vector dV³_j/db_k iterated over j
            dV3 = [w2[l][j] * delH[l] * training[k] for j in range(10)]
            
            #calculate vector of sum_j delAlpha_i(V³)/delV³_j * dV³_j/dw_lk iterated over i
            delSumVector = [np.dot(delAlpha[i],dV3) for i in range(10)]

            #calculate dE(w)/dw_lk by dot multiplying (alpha_i(V³) - t_n) by delSumVector !check paper to notice this is a dot product over i!
            dE = np.dot((activatedV3 - hotTraining), delSumVector)

            gradientMatrix[k][l] = dE
        
        #skip w_lk calculation for trivial zero
        #else:
            #gradientMatrix[k][l] = 0
        #commented out, dont actually need to run this the matrix we initialise already has zeroes here

        #debug 
        #if k == 0 and l >= 570:
            #print("start")
            #print(dE)
            #print(dV3)
            #print("delH")
            #print(delH[l])
            #print([w2[l][j] for j in range(10)])
        
    return gradientMatrix


#returns 1, if both parameters are equal
def indicatorFunc(i, j):
    if i == j:
        return 1
    else:
        return 0




#debug
if __name__ == "__main__":
    with open("single_example.csv", "r") as csvfile:
        first_line = csvfile.readline()    
    w1 = np.loadtxt('w1.csv', delimiter=',')
    w2 = np.loadtxt('w2.csv', delimiter=',')        
    gradientMat = sumGradient1(100)

    np.savetxt("testing.csv", gradientMat, delimiter=",", fmt="%.10f")
