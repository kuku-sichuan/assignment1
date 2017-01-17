import numpy as np
from random import shuffle

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
  train_num = X.shape[0]
  class_num = W.shape[1]
  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  
 
  # compute the np.exp will cause the number become huge big! 
  # we make this: for every sample, we substract the max values
  # this is not influence the final result!
  f = X.dot(W)
  f = f - np.max(f, axis=1,keepdims=True)
  score_matrix = np.exp(f)
  
   # compute the pro of the sample
  scores = score_matrix / np.sum(score_matrix,axis=1,keepdims=True)
  
  # crate the true y of matrix
  true_y = np.zeros_like(score_matrix)
  true_y[np.arange(train_num),y] = 1.0
  
  for num_s in xrange(train_num):
	for num_c in xrange(class_num):
	 loss += -true_y[num_s,num_c] * np.log(scores[num_s,num_c])
	 dW[:,num_c] += -(true_y[num_s,num_c] - scores[num_s,num_c]) * X[num_s,:]
  loss += 0.5 * reg * np.sum(W * W)
  dW += reg * W
  loss /= train_num
  dW /= train_num
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
  train_num = X.shape[0]
  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  f = X.dot(W)
  f = f - np.max(f,axis = 1,keepdims=True)
  scores = np.exp(f)
  scores = scores / np.sum(scores,axis = 1,keepdims=True)
  
  true_y = np.zeros_like(scores)
  true_y[np.arange(train_num),y] = 1.0
  loss = np.sum(-np.log(scores[np.arange(train_num),y]))
  dW = -np.dot(X.T,(true_y - scores)) 
  loss += 0.5 * reg * np.sum(W * W)
  dW += reg * W
  loss /= train_num
  dW /= train_num
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

