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

    N = X.shape[0]
    D = X.shape[1]
    C = W.shape[1]

    for i in range(N):
        score = np.dot(X[i], W)
        score_exp = np.exp(score)
        score_normalized = score_exp / np.sum(score_exp)
        ans = y[i]
        for j in range(C):
            if j == ans:
                loss -= np.log(score_normalized[j])        
                dW[:, ans] -= X[i]
            
            dW[:, j] +=  X[i] * score_normalized[j]

    loss /= N
    loss += reg * np.sum(W * W)

    dW /= N
    dW += reg * W * 2

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

    N = X.shape[0]
    D = X.shape[1]
    C = W.shape[1]

    score = np.dot(X, W)
    score_exp = np.exp(score)
    score_exp_sum = np.sum(score_exp, axis = 1, keepdims=True)
    # score_exp_sum = np.reshape(score_exp_sum, (N, 1))
    score_normalized = score_exp / score_exp_sum # N x D
    
    correct_scores = - np.log(score_normalized[np.arange(N), y])
    loss += np.sum(correct_scores)
    
    # temp = X[np.arange(N), y]
    score_normalized[np.arange(N), y] -= 1
    dW += np.dot(X.T, score_normalized)

    loss /= N
    loss += reg * np.sum(W * W)

    dW /= N
    dW += reg * W * 2

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
