from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange

def svm_loss_naive(W, X, y, reg):
    """
    Structured SVM loss function, naive implementation (with loops).

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
    dW = np.zeros(W.shape) # initialize the gradient as zero

    # compute the loss and the gradient
    num_classes = W.shape[1]
    num_train = X.shape[0]
    loss = 0.0
    for i in range(num_train):
        scores = X[i].dot(W) #1x3073 * 3073x10 = 1x10 500表示有500个训练集 10表示有10各类 scores[j]表示第i个训练集在第j个类上的得分
        correct_class_score = scores[y[i]] #y[i]为X[i]的标签，因此scores[y[i]]为X[i]正确的得分       
        count = 0
        for j in range(num_classes):
            if j == y[i]:                
                continue
            margin = scores[j] - correct_class_score + 1  # note delta = 1
            if margin > 0:
                loss += margin
                dW[:,j] += X[i].T
                dW[:,y[i]] -= X[i].T
                #count += 1
        #dW[:,y[i]] += -1 * (X[i].T * count)

    # Right now the loss is a sum over all training examples, but we want it
    # to be an average instead so we divide by num_train.
    loss /= num_train

    # Add regularization to the loss.
    loss += reg * np.sum(W * W)

    #############################################################################
    # TODO:                                                                     #
    # Compute the gradient of the loss function and store it dW.                #
    # Rather than first computing the loss and then computing the derivative,   #
    # it may be simpler to compute the derivative at the same time that the     #
    # loss is being computed. As a result you may need to modify some of the    #
    # code above to compute the gradient.                                       #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    dW /= num_train #loss /= num_train 因此还需除以一个常数
    dW += 2 * W * reg #loss += reg * np.sum(W * W) 对loss求偏导数，偏导数为 2 * W * reg

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    
    return loss, dW



def svm_loss_vectorized(W, X, y, reg):
    """
    Structured SVM loss function, vectorized implementation.

    Inputs and outputs are the same as svm_loss_naive.
    """
    loss = 0.0
    dW = np.zeros(W.shape) # initialize the gradient as zero

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the structured SVM loss, storing the    #
    # result in loss.                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    #scores = np.zeros(X.shape[0], W.shape[1])
    
    num_train = X.shape[0]
    scores = X.dot(W) #500x10
    true_scores = scores[range(num_train), y[range(num_train)]].reshape(-1, 1)
    #scores -= true_scores
    #scores += 1
    #scores = np.maximum(0, scores)
    #Li = np.sum(scores, axis = 1)
    #Li -= 1
    #loss += np.sum(Li)
    #loss /= num_train
    #loss += reg * np.sum(W * W)
    scores = np.maximum(scores - true_scores + 1, 0)
    scores[range(num_train), y[range(num_train)]] = 0
    loss = np.sum(scores) / num_train + reg * np.sum(W * W)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the gradient for the structured SVM     #
    # loss, storing the result in dW.                                           #
    #                                                                           #
    # Hint: Instead of computing the gradient from scratch, it may be easier    #
    # to reuse some of the intermediate values that you used to compute the     #
    # loss.                                                                     #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    scores[scores > 0] = 1 
    scores[range(num_train), y] = -np.sum(scores, axis = 1) 
    dW = (X.T).dot(scores) 
    dW = dW / num_train + 2 * reg * W


    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
