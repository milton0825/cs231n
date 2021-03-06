from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange

def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    dimension = X.shape[1]
    num_train = X.shape[0]
    num_classes = W.shape[1]

    for i in range(num_train):
        exp_s_yi = np.exp(np.dot(W[:,y[i]], X[i]))
        exp_sum_sj = np.sum(np.exp(np.dot(X[i].reshape(dimension, 1).T, W)))
        loss -= np.log(exp_s_yi / exp_sum_sj)

    loss = loss / num_train + reg * np.sum(W * W)


    for i in range(num_train):
        exp_s_yi = np.exp(np.dot(W[:,y[i]], X[i]))
        exp_sum_sj = np.sum(np.exp(np.dot(X[i].reshape(dimension, 1).T, W)))
        for j in range(num_classes):    
            if j == y[i]:
                dW[:,j] += X[i] * (exp_sum_sj - exp_s_yi) / exp_sum_sj
            else:
                exp_s_j = np.exp(np.dot(W[:,j], X[i]))
                dW[:,j] -= X[i] * exp_s_j / exp_sum_sj

    dW = -dW / num_train + 2 * reg * W 

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    num_train, dimension = X.shape
    num_classes = W.shape[1]

    XW = np.dot(X, W)

    assert XW.shape == (num_train, num_classes)

    exp_s_yi = np.exp(XW[np.arange(num_train), y])
    assert exp_s_yi.shape == (num_train,)

    exp_s_ji_sum = np.sum(np.exp(XW), axis=1)

    loss = - np.sum(np.log(exp_s_yi / exp_s_ji_sum)) / num_train + reg * np.sum(W * W)


    a = -np.exp(XW) / np.sum(np.exp(XW), axis=1).reshape(num_train,1)
        
    a[np.arange(num_train), y] += 1

    dW += np.dot(X.T, a)

    dW = -dW / num_train + 2 * reg * W

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
