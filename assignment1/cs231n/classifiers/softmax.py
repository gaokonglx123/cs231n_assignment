from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange
from math import log


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

    num_train = X.shape[0]
    num_class = W.shape[1]
    for i in range(num_train):
        scores = X[i].dot(W)       
        max_score = np.max(scores)
        sum_j = np.sum(np.exp(scores - max_score))
        margin = np.exp(scores[y[i]] - max_score) / sum_j
        for j in range(num_class):            
            if j == y[i]:
                dW[:, y[i]] += (np.exp(scores[j] - max_score) / sum_j - 1) * X[i].T
            else :
                dW[:, j] += (np.exp(scores[j] - max_score) / sum_j) * X[i].T
        
        loss += -log(margin)
        
    
    loss /= num_train
    loss += reg * np.sum(W * W)
    dW /= num_train
    dW += 2 * reg * W

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

    
    num_train = X.shape[0]
    num_class = W.shape[1]
    
    scores = X.dot(W)
    max_scores = np.max(scores, axis=1).reshape(-1, 1)
    true_scores = np.exp(scores[range(num_train), y[range(num_train)]].reshape(-1, 1) - max_scores)
    scores -= max_scores
    scores = np.exp(scores)
    sum_scores = np.sum(scores, axis=1).reshape(-1,1)
    loss = np.sum(-1 * np.log(true_scores / sum_scores))
    
    margin = scores / sum_scores
    margin[range(num_train), y[range(num_train)]] -= 1
    dW = X.T.dot(margin)
    
    loss /= num_train
    loss += reg * np.sum(W * W)
    dW /= num_train
    dW += 2 * reg * W 

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
