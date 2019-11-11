"""
This module implements various modules of the network.
You should fill in code into indicated sections.
"""
import numpy as np

class LinearModule(object):
  """
  Linear module. Applies a linear transformation to the input data. 
  """
  def __init__(self, in_features, out_features):
    """
    Initializes the parameters of the module. 
    
    Args:
      in_features: size of each input sample
      out_features: size of each output sample

    TODO:
    Done - Initialize weights self.params['weight'] using normal distribution with mean = 0 and
    std = 0.0001. Initialize biases self.params['bias'] with 0. 
    
    Done - Also, initialize gradients with zeros.
    """
    
    ########################
    # PUT YOUR CODE HERE  #
    #######################
    self.params = {'weight': None, 'bias': None}
    self.grads = {'weight': None, 'bias': None}
#    raise NotImplementedError
    self.params['weight'] = np.random.normal(loc=0.0, scale=0.0001, size=(out_features, in_features))
    self.params['bias'] = np.zeros((out_features,1))
    self.grads['weight'] = np.zeros(shape=(out_features, in_features))
    self.grads['bias'] = np.zeros((out_features, 1))


    ########################
    # END OF YOUR CODE    #
    #######################

  def forward(self, x):
    """
    Forward pass.
    
    Args:
      x: input to the module
    Returns:
      out: output of the module
    
    TODO:
    Implement forward pass of the module. 
    
    Hint: You can store intermediate variables inside the object. They can be used in backward pass computation.                                                           #
    """
    
    ########################
    # PUT YOUR CODE HERE  #
    #######################
    #raise NotImplementedError
    # x is an S x l-1
    # W is an l x l-1
    # b is an l x 1
    self.act = x
    # print("Forward Linear")
    # print(x.shape)
    # print(self.params['weight'].shape)
    # print(self.params['bias'].shape)
    out = x @ self.params['weight'].T + self.params['bias'].T  # shape (out x 1)

    ########################
    # END OF YOUR CODE    #
    #######################

    return out

  def backward(self, dout):
    """
    Backward pass.

    Args:
      dout: gradients of the previous module
    Returns:
      dx: gradients with respect to the input of the module
    
    TODO:
    Implement backward pass of the module. Store gradient of the loss with respect to 
    layer parameters in self.grads['weight'] and self.grads['bias']. 
    """

    ########################
    # PUT YOUR CODE HERE  #
    #######################
    #raise NotImplementedError
    # print('Weights update time')
    # print(dout.shape)
    # print(self.grads['weight'].shape)
    print('Linear time')
    print(self.act.shape)
    print(self.grads['weight'].shape)
    #dw = np.array([np.eye(self.grads['weight'].shape[1], 1, -i) @ self.act.T for i in range(0, self.grads['weight'].shape[1])])
    dw =
    self.grads['weight'] = dout.dot(dw)
    self.grads['bias'] = dout.dot(np.eye(dout.shape[1]))
    dx = dout.dot(self.params['weight'])
    ########################
    # END OF YOUR CODE    #
    #######################
    
    return dx

class LeakyReLUModule(object):
  """
  Leaky ReLU activation module.
  """
  def __init__(self, neg_slope):
    """
    Initializes the parameters of the module.

    Args:
      neg_slope: negative slope parameter.

    TODO:
    Initialize the module.
    """

    ########################
    # PUT YOUR CODE HERE  #
    #######################
    #raise NotImplementedError
    self.n_slope = neg_slope
    # self.input = None
    # self.output = None
    self.act = None
    ########################
    # END OF YOUR CODE    #
    #######################

  def forward(self, x):
    """
    Forward pass.
    
    Args:
      x: input to the module
    Returns:
      out: output of the module
    
    TODO:
    Implement forward pass of the module. 
    
    Hint: You can store intermediate variables inside the object. They can be used in backward pass computation.                                                           #
    """

    ########################
    # PUT YOUR CODE HERE  #
    #######################
    #raise NotImplementedError
    self.act = x
    out = np.maximum(0,x) + (np.minimum(0,x)*self.n_slope)
    ########################
    # END OF YOUR CODE    #
    #######################

    return out

  def backward(self, dout):
    """
    Backward pass.

    Args:
      dout: gradients of the previous module
    Returns:
      dx: gradients with respect to the input of the module
    
    TODO:
    Implement backward pass of the module.
    """

    ########################
    # PUT YOUR CODE HERE  #
    #######################
    # raise NotImplementedError
    dx = dout * (self.act > 0 + self.n_slope * (self.act <= 0))
    ########################
    # END OF YOUR CODE    #
    #######################    

    return dx


class SoftMaxModule(object):
  """
  Softmax activation module.
  """

  def forward(self, x):
    """
    Forward pass.
    Args:
      x: input to the module
    Returns:
      out: output of the module

    TODO:
    Implement forward pass of the module.
    To stabilize computation you should use the so-called Max Trick - https://timvieira.github.io/blog/post/2014/02/11/exp-normalize-trick/

    Hint: You can store intermediate variables inside the object. They can be used in backward pass computation.
    """

    ########################
    # PUT YOUR CODE HERE  #
    #######################
    # raise NotImplementedError

    mx = x.max(axis = 1, keepdims = True)
    # print(mx)
    y = np.exp(x - mx)
    out = y / y.sum(axis = 1, keepdims = True)
    self.act = out
    # print(out.shape)
    ########################
    # END OF YOUR CODE    #
    #######################

    return out

  def backward(self, dout):
    """
    Backward pass.
    Args:
      dout: gradients of the previous modul
    Returns:
      dx: gradients with respect to the input of the module

    TODO:
    Implement backward pass of the module.
    """

    ########################
    # PUT YOUR CODE HERE  #
    #######################
    # raise NotImplementedError

    dx_dot = self.act[:,:,None] * self.act[:,None]
    dx_diag = np.apply_along_axis(np.diag, axis=1, arr=self.act)
    dxdx = dx_diag - dx_dot
    dx = np.einsum('ij,ijk -> ik', dout, dxdx)

    # breakpoint()
    #######################
    # END OF YOUR CODE    #
    #######################

    return dx

class CrossEntropyModule(object):
  """
  Cross entropy loss module.
  """

  def forward(self, x, y):
    """
    Forward pass.
    Args:
      x: input to the module
      y: labels of the input
    Returns:
      out: cross entropy loss

    TODO:
    Implement forward pass of the module.
    """

    ########################
    # PUT YOUR CODE HERE  #
    #######################
    # raise NotImplementedError
    out = -np.sum(y*np.log(x))/y.shape[0]
    # print("testing the out things")
    # print(out.shape)
    # print(out)
    # out = - sum(y.T.dot(np.log(x)))/y.shape[1]
    self.loss = out
    ########################
    # END OF YOUR CODE    #
    #######################

    return out

  def backward(self, x, y):
    """
    Backward pass.
    Args:
      x: input to the module
      y: labels of the input
    Returns:
      dx: gradient of the loss with the respect to the input x.

    TODO:
    Implement backward pass of the module.
    """

    ########################
    # PUT YOUR CODE HERE  #
    #######################
    # raise NotImplementedError
    # print('Shape y and x')
    # print(y.shape)
    # print(x.shape)
    # test = np.divide(-y,x)
    # print(test.shape)
    # print("the loss shape is:")
    # print(self.loss.shape)
    dx = (-y / x)/y.shape[0]

    #print(dx.shape)
    # this is old and possibly stupid
    # dx = self.loss*np.diag(np.divide(-y,x))
    # dx = np.diag(np.divide(-y, x)) / y.shape[1]
    # print("Loss is back propping")
    # print(dx.shape)
    ########################
    # END OF YOUR CODE    #
    #######################

    return dx