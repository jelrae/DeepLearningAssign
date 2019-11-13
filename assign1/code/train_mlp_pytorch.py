"""
This module implements training and evaluation of a multi-layer perceptron in PyTorch.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import os
from mlp_pytorch import MLP
import cifar10_utils
import torch.nn as nn
import torch

import matplotlib.pyplot as plt

# Default constants
DNN_HIDDEN_UNITS_DEFAULT = '100'
LEARNING_RATE_DEFAULT = 2e-3
MAX_STEPS_DEFAULT = 1500
BATCH_SIZE_DEFAULT = 200
EVAL_FREQ_DEFAULT = 100
NEG_SLOPE_DEFAULT = 0.02

# Directory in which cifar data is saved
DATA_DIR_DEFAULT = './cifar10/cifar-10-batches-py'

FLAGS = None

def accuracy(predictions, targets):
  """
  Computes the prediction accuracy, i.e. the average of correct predictions
  of the network.
  
  Args:
    predictions: 2D float array of size [batch_size, n_classes]
    labels: 2D int array of size [batch_size, n_classes]
            with one-hot encoding. Ground truth labels for
            each sample in the batch
  Returns:
    accuracy: scalar float, the accuracy of predictions,
              i.e. the average correct predictions over the whole batch
  
  TODO:
  Implement accuracy computation.
  """

  ########################
  # PUT YOUR CODE HERE  #
  #######################
  # raise NotImplementedError
  accuracy = np.sum(np.argmax(predictions, axis=1) == np.argmax(targets, axis=1)) / targets.shape[0]
  ########################
  # END OF YOUR CODE    #
  #######################

  return accuracy

def train():
  """
  Performs training and evaluation of MLP model. 

  TODO:
  Implement training and evaluation of MLP model. Evaluate your model on the whole test set each eval_freq iterations.
  """

  ### DO NOT CHANGE SEEDS!
  # Set the random seeds for reproducibility
  np.random.seed(42)

  ## Prepare all functions
  # Get number of units in each hidden layer specified in the string such as 100,100
  if FLAGS.dnn_hidden_units:
    dnn_hidden_units = FLAGS.dnn_hidden_units.split(",")
    dnn_hidden_units = [int(dnn_hidden_unit_) for dnn_hidden_unit_ in dnn_hidden_units]
  else:
    dnn_hidden_units = []

  # Get negative slope parameter for LeakyReLU
  neg_slope = FLAGS.neg_slope


  ########################
  # PUT YOUR CODE HERE  #
  #######################
  # raise NotImplementedError
  loss_train = []
  acc_train = []
  acc_test = []

  loss_funct = nn.CrossEntropyLoss()
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  cifar10_set = cifar10_utils.get_cifar10(FLAGS.data_dir)

  x, y = cifar10_set['train'].next_batch(FLAGS.batch_size)
  x = x.reshape(FLAGS.batch_size, -1)
  # print(y.shape)
  out_dim = y.shape[1]
  in_dim = x.shape[1]
  mlp = MLP(in_dim, dnn_hidden_units, out_dim, neg_slope).to(device)
  optimizer = torch.optim.SGD(mlp.parameters(), lr = FLAGS.learning_rate)
  for i in range(0, FLAGS.max_steps + 1):
    x, t = cifar10_set['train'].next_batch(FLAGS.batch_size)
    x = torch.tensor(x.reshape(FLAGS.batch_size, -1), dtype=torch.float32).to(device)
    y = mlp.forward(x)
    loss = loss_funct(y,torch.LongTensor(np.argmax(t, 1)).to(device))
    loss_train.append(loss)
    acc_train.append(accuracy(y.cpu().detach().numpy(), t))
    # breakpoint()
    loss.backward()
    optimizer.step()
    if i % FLAGS.eval_freq == 0:
      x,t = cifar10_set['test'].images, cifar10_set['test'].labels
      x = torch.tensor(x.reshape(x.shape[0], -1), dtype=torch.float32).to(device)
      y = mlp.forward(x)
      # print(t.shape)
      # print(y.shape)
      acc_test.append(accuracy(y.cpu().detach().numpy(),t))
      print("The accuracy at step, " + str(i) + " is : " + str(acc_test[-1]))

  #Plotting the accuracy of test and train:
  # plt.figure(0, figsize = (17,10))
  plt.figure(0)
  plt.plot(np.arange(0, len(acc_train)), acc_train, label = 'Train')
  plt.plot(np.arange(0,len(acc_train), FLAGS.eval_freq), acc_test, label = 'Test')
  plt.xlabel('Epoch')
  plt.ylabel('Accuracy')
  plt.title('Accuracy of Train and Test Set Through Training')
  plt.legend()
  plt.savefig('Accuracy_basic1.png')
  # plt.show()

  # plt.figure(1, figsize=(17,10))
  plt.figure(1)
  plt.plot(np.arange(0, len(loss_train)), loss_train, label = 'Train')
  plt.xlabel('Epoch')
  plt.ylabel('Loss')
  plt.title('Loss Through Training')
  plt.savefig('Loss_basic1.png')
  # plt.show()
  # plt.legend()
  ########################
  # END OF YOUR CODE    #
  #######################

def print_flags():
  """
  Prints all entries in FLAGS variable.
  """
  for key, value in vars(FLAGS).items():
    print(key + ' : ' + str(value))

def main():
  """
  Main function
  """
  # Print all Flags to confirm parameter settings
  print_flags()

  if not os.path.exists(FLAGS.data_dir):
    os.makedirs(FLAGS.data_dir)

  # Run the training operation
  train()

if __name__ == '__main__':
  # Command line arguments
  parser = argparse.ArgumentParser()
  parser.add_argument('--dnn_hidden_units', type = str, default = DNN_HIDDEN_UNITS_DEFAULT,
                      help='Comma separated list of number of units in each hidden layer')
  parser.add_argument('--learning_rate', type = float, default = LEARNING_RATE_DEFAULT,
                      help='Learning rate')
  parser.add_argument('--max_steps', type = int, default = MAX_STEPS_DEFAULT,
                      help='Number of steps to run trainer.')
  parser.add_argument('--batch_size', type = int, default = BATCH_SIZE_DEFAULT,
                      help='Batch size to run trainer.')
  parser.add_argument('--eval_freq', type=int, default=EVAL_FREQ_DEFAULT,
                        help='Frequency of evaluation on the test set')
  parser.add_argument('--data_dir', type = str, default = DATA_DIR_DEFAULT,
                      help='Directory for storing input data')
  parser.add_argument('--neg_slope', type=float, default=NEG_SLOPE_DEFAULT,
                      help='Negative slope parameter for LeakyReLU')
  FLAGS, unparsed = parser.parse_known_args()

  main()