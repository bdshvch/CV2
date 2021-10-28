import doctest
import numpy as np
from numpy import linalg as LA

def Perceptron(TrainX, TrainY):
    """
    >>> Perceptron(np.array([[1, 0, 0, 0, 0, 0]]), np.array([[-1]]))
    array([-1.,  0.,  0.,  1.,  0.,  1.])
    >>> Perceptron(np.array([[1, 1, 0, 0, 0, 0]]), np.array([[-1]]))
    array([-1., -1.,  0.,  1.,  0.,  1.])
    >>> Perceptron(np.array([[1, 1, 1, 0, 0, 0]]), np.array([[-1]]))
    array([-1., -1., -1.,  1.,  0.,  1.])
    """
    condition = True
    alpha = np.zeros(6)
    while(condition):
        i = 0
        while(i != len(TrainX)):
            if (TrainY[i] * np.dot(TrainX[i], alpha) <= 0):
                alpha = alpha + np.multiply(TrainY[i], TrainX[i])
                i = 0
            else:
                i = i + 1
        Matrix = np.array([[alpha[3], alpha[4]/2], [alpha[4]/2, alpha[5]]])
        w, vb = LA.eig(Matrix)
        vb = vb.T
        xTemp = np.array([[0, 0, 0, vb[0][0] * vb[0][0], vb[0][0] * vb[0][1], vb[0][1] * vb[0][1]], [0, 0, 0, vb[1][0] * vb[1][0], vb[1][0] * vb[1][1], vb[1][1] * vb[1][1]]])

        if (np.dot(xTemp[0], alpha) <= 0 or np.dot(xTemp[1], alpha) <= 0):
            if (np.dot(xTemp[0], alpha) <= 0):
                alpha = alpha + xTemp[0]
            if (np.dot(xTemp[1], alpha) <= 0):
                alpha = alpha + xTemp[1]
        else:
            condition = False
    return alpha

if __name__ == "__main__":
    doctest.testmod()