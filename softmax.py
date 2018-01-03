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
  scores = X.dot(W)  #N*C
  scores_norm = socres - np.max(scores, axis=1).reshape(-1,1)
  scores_exp = np.exp(scores_norm)
  num_train = X.shape[0]
  num_class = W.shape[1]
  loss_matrix = []
  for i in range(num_train):
    loss_matrix[i] = - np.log(scores_exp[i,y[i]] / np.sum(scores_exp[i,:]))
    for j in range(num_class):
        dW[:,j] = - X.T[i,y[i]] + X.T.dot(scores_exp[:, j] / np.sum(socres_exp[:, j])
  loss = np.sum(loss_matrix) / num_train + reg * W * W
  dW = dW / num_train + 2 * reg * W
  #Li=−fyi+log∑jefj
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

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
  scores = X.dot(W)  #N*C
  scores_norm = socres - np.max(scores, axis=1).reshape(-1,1)
  scores_exp = np.exp(scores_norm)
  num_train = X.shape[0]
  num_class = W.shape[1]

  loss_matrix = scores_exp / np.sum(scores_exp,axis=1).reshape(-1,1)
  loss = np.sum(loss_matrix[(range(num_train),y)])
  loss = loss / num_train + reg * W * W
  
  L = np.zeros_like(scores)
  L = 1 * L[(range(num_train),y)]
  dW_term1 = - X.T.dot(L)
  dW_term2 = X.T.dot(loss_matrix)
  dW = (dW_term1 + dW_term2) / num_train + 2 * reg * W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW
