from neuralnetwork import Hidden
from neuralnetwork import Feature

import numpy as np


def generateWeights():
    #generate w1 (weights between input and hidden layer), unif(-1,1) distribution
    w1 = np.random.uniform(low=-1, high=1, size=(Feature + 1, Hidden))
    #save to w1.csv with 5 decimal places
    np.savetxt("w1.csv", w1, delimiter=",", fmt="%.5f")

    w2 = np.random.uniform(low=-1, high=1, size=(Hidden + 1, 10))
    np.savetxt("w2.csv", w2, delimiter=",", fmt="%.5f")


if __name__ == "__main__":
    generateWeights()