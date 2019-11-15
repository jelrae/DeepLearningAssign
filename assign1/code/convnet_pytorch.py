"""
This module implements a Convolutional Neural Network in PyTorch.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch.nn as nn

class ConvNet(nn.Module):
  """
  This class implements a Convolutional Neural Network in PyTorch.
  It handles the different layers and parameters of the model.
  Once initialized an ConvNet object can perform forward.
  """

  def __init__(self, n_channels, n_classes):
    """
    Initializes ConvNet object.

    Args:
      n_channels: number of input channels
      n_classes: number of classes of the classification problem


    TODO:
    Implement initialization of the network.
    """

    ########################
    # PUT YOUR CODE HERE  #
    #######################
    # raise NotImplementedError
    super(ConvNet, self).__init__()
    mp = {'stride': 2, 'padding': 1, 'kernel': 3}
    conv = {'stride': 1, 'padding': 1, 'kernel': 3}
    size = [64,128,256,512]

    self.mods = nn.Sequential(
    nn.Conv2d(n_channels, size[0],kernel_size= conv['kernel'], stride=conv['stride'], padding=conv['padding']),
    nn.BatchNorm2d(size[0]),
    nn.ReLU(),
    nn.Conv2d(size[0], size[0],kernel_size= mp['kernel'], stride=mp['stride'], padding=mp['padding']),
    nn.Conv2d(size[0], size[1],kernel_size= conv['kernel'], stride=conv['stride'], padding=conv['padding']),
    nn.BatchNorm2d(size[1]),
    nn.ReLU(),
    nn.Conv2d(size[1], size[1],kernel_size= mp['kernel'], stride=mp['stride'], padding=mp['padding']),
    nn.Conv2d(size[1], size[2],kernel_size= conv['kernel'], stride=conv['stride'], padding=conv['padding']),
    nn.BatchNorm2d(size[2]),
    nn.ReLU(),
    nn.Conv2d(size[2], size[2],kernel_size= conv['kernel'], stride=conv['stride'], padding=conv['padding']),
    nn.BatchNorm2d(size[2]),
    nn.ReLU(),
    nn.Conv2d(size[2], size[2],kernel_size= mp['kernel'], stride=mp['stride'], padding=mp['padding']),
    nn.Conv2d(size[2], size[3],kernel_size= conv['kernel'], stride=conv['stride'], padding=conv['padding']),
    nn.BatchNorm2d(size[3]),
    nn.ReLU(),
    nn.Conv2d(size[3], size[3],kernel_size= conv['kernel'], stride=conv['stride'], padding=conv['padding']),
    nn.BatchNorm2d(size[3]),
    nn.ReLU(),
    nn.Conv2d(size[3], size[3],kernel_size= mp['kernel'], stride=mp['stride'], padding=mp['padding']),
    nn.Conv2d(size[3], size[3],kernel_size= conv['kernel'], stride=conv['stride'], padding=conv['padding']),
    nn.BatchNorm2d(size[3]),
    nn.ReLU(),
    nn.Conv2d(size[3], size[3],kernel_size= conv['kernel'], stride=conv['stride'], padding=conv['padding']),
    nn.BatchNorm2d(size[3]),
    nn.ReLU(),
    nn.Conv2d(size[3], size[3],kernel_size= mp['kernel'], stride=mp['stride'], padding=mp['padding']),
    nn.BatchNorm2d(size[3]),
    nn.ReLU()
    )
    self.lin = nn.Linear(size[3], n_classes)

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
    out = self.mods(x)
    out = self.lin(out.squeeze())
    ########################
    # END OF YOUR CODE    #
    #######################

    return out
