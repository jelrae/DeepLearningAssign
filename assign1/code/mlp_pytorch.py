"""
This module implements a multi-layer perceptron (MLP) in PyTorch.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch.nn as nn
import custom_batchnorm as cbn

class MLP(nn.Module):
  """
  This class implements a Multi-layer Perceptron in PyTorch.
  It handles the different layers and parameters of the model.
  Once initialized an MLP object can perform forward.
  """

  def __init__(self, n_inputs, n_hidden, n_classes, neg_slope):
    """
    Initializes MLP object.

    Args:
      n_inputs: number of inputs.
      n_hidden: list of ints, specifies the number of units
                in each linear layer. If the list is empty, the MLP
                will not have any linear layers, and the model
                will simply perform a multinomial logistic regression.
      n_classes: number of classes of the classification problem.
                 This number is required in order to specify the
                 output dimensions of the MLP
      neg_slope: negative slope parameter for LeakyReLU

    TODO:
    Implement initialization of the network.
    """

    ########################
    # PUT YOUR CODE HERE  #
    #######################
    super(MLP, self).__init__()
    mods = []
    cur_input = n_inputs
    # if we have hidden, create hidden
    if len(n_hidden) > 0:
      for out_lay in n_hidden:
        # Uncomment one of these if you dont want batch norm
        # 3.1
        # mods.append(cbn.CustomBatchNormAutograd(cur_input))
        # 3.2/3
        # mods.append(cbn.CustomBatchNormManualModule(cur_input))
        print('Adding Linear Model')
        mods.append(nn.Linear(in_features=cur_input, out_features=out_lay))
        print('Adding LeakyRElU')
        mods.append(nn.LeakyReLU(neg_slope))
        cur_input = out_lay
    print('Adding output Linear')
    mods.append(nn.Linear(in_features=cur_input, out_features=n_classes))
    print(len(mods))
    self.modls = nn.Sequential(*mods)
    # raise NotImplementedError
    ########################
    # END OF YOUR CODE    #
    #######################

  def forward(self, x):
    """
    Performs forward pass of the input. Here an input tensor x is transformed through
    several layer transformations.

    Args:
      x: input to the network
    Returns:
      out: outputs of the network

    TODO:
    Implement forward pass of the network.
    """

    ########################
    # PUT YOUR CODE HERE  #
    #######################
    # raise NotImplementedError
    out = self.modls(x)
    ########################
    # END OF YOUR CODE    #
    #######################

    return out
