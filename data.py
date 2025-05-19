"""
Matrix of weights and biases between two layers:
------------------------------------------------
w = [[b_1 , b_2 , b_3 , ... , b_H ]
     [w_11, w_21, w_31, ... , w_H1]
     [w_12, w_22, w_32, ... , w_H2]
     [... , ...., ... , ... , ... ]
     [w_1N, w_2N, w_3N, ... , w_HN]]

with
    H = Amount of neurons of the next layer
    N = Amount of neurons of the previous layer
    w is of size (N+1, H)
"""



def getWeightLK(w, l, k):
    '''Get weigth w_lk from the matrix containing the biases and weights.

    Format of the matrix is described in a comment in "data.py"

    Parameters
    ----------
    w: ndarray
        The matrix containing the biases in the first row and the weights in the remaining rows.
    l: int
        Index l of the neuron of the target layer, the weigth belongs to.
    k: int
        Index k of the neuron of the previous layer, the weigth belongs to.
    
    Returns
    -------
    w_lk: float
        The weight w_lk between neuron k of previous layer and neuron l of next layer
    '''
    return w[k][l-1]

def setWeigthLK(w, l, k, value):
    '''Set weigth w_lk from the matrix containing the biases and weights to value.

    Format of the matrix is described in a comment in "data.py"

    Parameters
    ----------
    w: ndarray
        The matrix containing the biases in the first row and the weights in the remaining rows.
    l: int
        Index l of the neuron of the target layer, the weigth belongs to.
    k: int
        Index k of the neuron of the previous layer, the weigth belongs to.
    value: float
        Value which will be set for w at index l and k.
    '''
    w[k][l-1] = value

def getBiasL(w, l):
    '''Get bias b_l from the matrix containing the biases and weights.

    Format of the matrix is described in a comment in "data.py"

    Parameters
    ----------
    w: ndarray
        The matrix containing the biases in the first row and the weights in the remaining rows.
    l: int
        Index l of the neuron of the target layer, the bias belongs to.
    
    Returns
    -------
    b_l: float
        The bias b_l for the neuron l of the target layer the matrix w belongs to.
    '''
    return w[0][l-1]

def setBiasL(w, l, value):
    '''Sets bias b_l from the matrix containing the biases and weights to value.

    Format of the matrix is described in a comment in "data.py"

    Parameters
    ----------
    w: ndarray
        The matrix containing the biases in the first row and the weights in the remaining rows.
    l: int
        Index l of the neuron of the target layer, the bias belongs to.
    value: float
        The value which will be set for bias of index l.
    '''
    w[0][l-1] = value

def getAllWeights(w, N):
    '''Get all the weights w_lk from the matrix containing the biases and weights as a (N, H) matrix, where N is amount of neurons of previous layer and H amount of neurons of next layer.

    Format of the matrix is described in a comment in "data.py"

    WARNING
    -------
        The functions of "data.py" for getting single values of a weight_and_bias matrix does not work for the returned weight matrix by this function!

    Parameters
    ----------
    w: ndarray
        The matrix containing the biases in the first row and the weights in the remaining rows.
    N: int
        Amount of neurons of the previous layer.
    
    Returns
    -------
    weights: ndarray
        Submatrix (N, H) of the original matrix containing just the weights without the biases.
    '''
    n_plus_one = N+1
    return w[1:n_plus_one]

def getAllBiases(w):
    '''Get all the biases b_l from the matrix containing the biases and weights as a 1D-array of length H, where H is the amount of neurons of next layer.

    Format of the matrix is described in a comment in "data.py"

    WARNING
    -------
        The functions of "data.py" for getting single values of a weight_and_bias matrix does not work for the returned bias vector by this function!

    Parameters
    ----------
    w: ndarray
        The matrix containing the biases in the first row and the weights in the remaining rows.
    
    Returns
    -------
    biases: ndarray
        Subvector (length = H) of the original matrix containing just the biases without the weights.
    '''
    return w[0]