from neuralnetwork import activ
from neuralnetwork import delActiv
from neuralnetwork import outActiv
from neuralnetwork import delOutActiv
from neuralnetwork import Hidden      #amount of hidden layer neurons H

import numpy as np

#learning rate for gradient descent (rho or eta)
LearningRate = 0.01

#convergence threshold
ConvergenceThreshold = 0.01

class GradientDescent:

    #values and activation function values of the neural net for some input TODO: make it dynamic according to neurons per layer (H) and number of layers (L)
    values1 = [] #input values 
    values2 = [] #values of hidden layer
    activ2 = []  #values of hidden layer after activation function
    values3 = [] #values of output layer
    activ3 = []  #values of output layer after activation function

    w1_initial = [] #initial given weights and biases between the input and first hidden layer
    w2_initial = [] #initial given weights and biases between the hidden and output layer

    w1_updated = [] #trained weights and biases between the input and first hidden layer after amountIterations
    w2_updated = [] #trained weights and biases between the hidden and output layer after amountIterations

    w1_working = [] #weights and biases between input and hidden layer for the interim result (x_k+1 in gradient descent formula)
    w2_working = [] #weights and biases between hidden and output layer for the interim result (x_k+1 in gradient descent formula)

    trainingData = 'mnist_train.csv' #training data as a csv file name
    amountTrainingData = 0           #amount of training data (M)

    amountIterations = 0      #amount of iterations of the gradient descent algorithm 
    amountIterations_max = -1 #ampunt of mximal iterations of the gradient descent algorithm. If it is -1, it is equivalent to no maximum
    
    #initialize needed values for the gradient descent algorithm
    def __init__(self, w1, w2, trainingDataFile='mnist_train.csv', maxAmountIterations=-1):
        '''Initialize needed values for the gradient descent algorithm.
        
        Parameters
        ----------
        w1: ndarray
            ndarray read from a csv file containing the weights and biases between the input and first hidden layer
        w2: ndarray
            ndarray read from a csv file containing the weigths and biases between the hidden and output layer
        trainingDataFile: String
            the name of the training data file (must end on '.csv')
        '''
        self.w1_initial = w1
        self.w2_initial = w2
        self.trainingData = trainingDataFile
        self.amountIterations_max = maxAmountIterations

        #create w1_updated
        np.copyto(self.w1_updated, self.w1_initial)
        #create w2_updated
        np.copyto(self.w2_updated, self.w2_initial)

        #initialize w*_working
        np.copyto(self.w1_working, self.w1_initial)
        np.copyto(self.w2_working, self.w2_initial)

        #initialize values of the net and amount of training data
        self.amountTrainingData = 0
        with open(self.trainingData, "r") as csvfile:
            first_line = csvfile.readline()
            self.amountTrainingData += sum(1 for line in csvfile) + 1 #+1 because we already read a line
            
        self.updateValues1(first_line)
        self.calculateNetValues(self.w1_initial, self.w2_initial)


    #updates the input layer neurons (values1) based on a given data point as a csv line
    def updateValues1(self, singleInput):
        '''Updates the input layer neurons (values1) based on a given data point as a csv line

        Attributes
        ----------
        singleInput:
            Datapoint as single line from csv file: label, x1, x2, ... , x784 
        '''
        #turn input string into vector
        self.values1 = np.fromstring(singleInput, sep=',')
        #add a 1 to the front of the vector, !remember the first entry of the vector is the label!
        self.values1[0] = 1
    
    #updates the values of the net based on the input layer (values1) and weights (parameters w1, w2)
    def calculateNetValues(self, w1, w2):
        ''' Updates the values of the net based on the input layer (values1) and weights (parameters w1, w2)
        
        Parameters
        ----------
        w1: ndarray
            ndarray read from a csv file containing the weights and biases between the input and first hidden layer
        w2: ndarray
            ndarray read from a csv file containing the weigths and biases between the hidden and output layer

        '''

        #Multiply Values1 by w1 matrix to get Values2, proof: trivial
        self.values2 = np.dot(self.values1, w1)

        #put activation function on Values2
        self.activ2 = [activ(i) for i in self.values2] #TODO an neue activ anpassen

        #prepend 1 to activated Hidden layer values vector
        self.activ2 = np.insert(self.activ2, 0, 1)
        
        #get the output values from activ2 and the w2 weights
        self.values3 = np.dot(self.activ2, w2)

        #calculate the final output values with the output activation function
        self.activ3 = [outActiv(i) for i in self.values3] #TODO an neue outActiv anpassen

    #performs gradient descent algorithm: minimizes func for initialInput
    #returns argmin of func (for a local minimum)
    def gradientDescent(self, func, initialInput):
        #TODO: Dokumentation und ggf. Überarbeiten!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        #x_k
        last_value = initialInput
        #x_k+1
        result = initialInput
        #for checking, if the the gradient descent algorithm converged
        diff = ConvergenceThreshold + 1

        #while the difference between x_k and x_k+1 is higher than ConvergenceThreshold: Calculate the next x_k+1
        while diff > ConvergenceThreshold:
            #calculate x_k+1
            result = self.calculateNextValue() #TODO: (x_k is in w*_updated, we calculate for w*_working (x_k+1) and copy that to w*_updated, after comparing the result to it)
            #calculate the difference between x_k and x_k+1
            diff = np.abs(last_value - result)

            #TODO: Update weight member attributes!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

            self.amountIterations += 1

            if (self.amountIterations_max != -1) and (self.amountIterations >= self.amountIterations_max):
                break

        return result

    #helper function for gradientDescent, calculating x_k+1
    def calculateNextValue(self):
        #calculate the gradients
        gradients = self.gradient() #uses w*_updated as input x_k from the gradient descent formula
        gradientW1 = gradients[0]
        gradientB2 = gradients[1]
        gradientW2 = gradients[2]
        gradientB3 = gradients[3]

        #gradient descent formula (x_k is in w*_updated, we calculate for w*_working (x_k+1) and copy that to w*_updated, after comparing the result to it)
        # TODO: VLLT MÜSSEN DIE ERGEBNISSE VON DER GRADIENT BERECHNUNG NOCHMAL TRANSPONIERT WERDEN? uhhh
        self.w1_working[1:ende,0:ende] = self.w1_updated[1:ende,0:ende] - LearningRate * gradientW1 #TODO: Hopefully we access the w part correctly and write it in the correct way
        self.w1_working[0] = self.w1_updated[0] - LearningRate * gradientB2                         #TODO: Hopefully we access the b-values correctly and write them in the right way
        self.w2_working[1:ende,0:ende] = self.w2_updated[1:ende,0:ende] - LearningRate * gradientW2 #TODO: Hopefully we access the w part correctly and write it in the correct way
        self.w2_working[0] = self.w2_updated[0] - LearningRate * gradientB3                         #TODO: Hopefully we access the b-values correctly and write them in the right way


        return result #TODO has to be written in a second updated file, because we want to keep the original file and we need two separate working ones...


    def gradient(self):
        gradientW1 = np.zeros(Hidden)
        gradientB2 = np.zeros(Hidden)
        gradientW2 = np.zeros(10)
        gradientB3 = np.zeros(10)

        #calculate gradient for weights w_lk^1 and biases b_l^2
        for l in range(Hidden):
            result = self.partialDerivativeW1B2(self.amountTrainingData, l) #returns partial derivatives for b_l^2 and w_lk^1 for all k
            gradientB2[l] = result[0]
            gradientW1[l] = result[1]
        
        
        #calculate gradient for weights w_lk^2 and biases b_l^3
        for l in range(10):
            result = self.partialDerivativeW2B3(self.amountTrainingData, l) #returns partial derivatives for b_l^3 and w_lk^2 for all k
            gradientB3[l] = result[0]
            gradientW2[l] = result[1]
        
        return np.array(gradientW1, gradientB2, gradientW2, gradientB3)


    def partialDerivativeW1B2(self, l):

        #initialize return values
        result_b_l = 0
        result_w_l_k = np.zeros(self.amountTrainingData)

        with open(self.trainingData, "r") as csvfile:
            for n in range(self.amountTrainingData):
                input_line = csvfile.readline()

                #calculate the net values based on the read input and the current weigths and biases
                self.updateValues1(input_line)
                self.calculateNetValues(self.w1_updated, self.w2_updated)
                
                #get the label t_n
                label = np.fromstring(input_line, sep=',')[0]
            
                for i in range(10):
                    
                    # for calculation of dE / db_l^2
                    result_b_l += (outActiv(self.values3) - label) * self.helperFuncB2(self.values3, i, l, self.values2)

                    for k in range(self.amountTrainingData): #TODO: Das ist glaube ich nicht M sondern Feature = 784??? Vllt. Notation verwechselt oder so, könnte an anderen Stellen dann auch falsch sein
                            # for calculation of dE / dw_l_k^1
                            result_w_l_k[k] += (outActiv(self.values3) - label) * self.helperFuncW1(self.values3, i, l, self.values2, k)
        return np.array(result_b_l, result_w_l_k)

    def partialDerivativeW2B3(self, l):

        #initialize return values
        result_b_l = 0
        result_w_l_k = np.zeros(Hidden)

        with open(self.trainingData, "r") as csvfile:
            for n in range(self.amountTrainingData):
                input_line = csvfile.readline()

                #calculate the net values based on the read input and the current weigths and biases
                self.updateValues1(input_line)
                self.calculateNetValues(self.w1_updated, self.w2_updated)
                
                #get the label t_n
                label = np.fromstring(input_line, sep=',')[0]
            
                for i in range(10):
                    
                    # for calculation of dE / db_l^3
                    result_b_l += (outActiv(self.values3) - label) * self.helperFuncB3(self.values3, i, l)
            
                    for k in range(Hidden):
                        # for calculation of dE / dw_l_k^2
                        result_w_l_k[k] += (outActiv(self.values3) - label) * self.helperFuncW2(self.values3, i, l, self.values2)
        return np.array(result_b_l, result_w_l_k)

    def helperFuncB2(self, v3, i, l, v2):
        result = 0
        for j in range(10):
            result += delOutActiv(v3, i, j) * self.w2_updated[j][l] * delActiv(v2, i, j) #TODO: maybe have to switch indices j and l accessing w
        return result

    def helperFuncW1(self, v3, i, l, v2, k):
        result = 0
        for j in range(10):
            result += delOutActiv(v3, i, j) * self.w2_updated[j][l] * delActiv(v2, i, j) * self.values1[k] #TODO maybe k+1 as index for values1 #TODO: maybe have to switch indices j and l accessing w
        return result
    
    def helperFuncB3(self, v3, i, l):
        result = 0
        for j in range(10):
            result += delOutActiv(v3, i, j) * self.indicatorFunc(l, j) #TODO
        return result
    
    def helperFuncW2(self, v3, i, l, v2):
        result = 0
        for j in range(10):
            result += delOutActiv(v3, i, j) * self.indicatorFunc(l, j) * activ(v2) #TODO
        return result

    #returns 1, if both parameters are equal
    def indicatorFunc(self, i, j):
        if i == j:
            return 1
        else:
            return 0
