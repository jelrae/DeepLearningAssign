"""
This module implements training and evaluation of a multi-layer perceptron in NumPy.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import os
from mlp_numpy import MLP
from modules import CrossEntropyModule
from modules import LinearModule #TODO IS THIS OK??????
import cifar10_utils

import matplotlib.pyplot as plt
# import seaborn as sns
# sns.set()

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
  # PUT YOUR CODE HERE
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

  ########################
  # PUT YOUR CODE HERE  #
  #######################
  #raise NotImplementedError
  #Load in data set
  cifar10_set = cifar10_utils.get_cifar10(FLAGS.data_dir)

  #Init tracking arrays
  test_acc = []
  train_loss = []
  train_acc = []
  # pull in the first set and get the shapes
  x, t = cifar10_set['train'].next_batch(FLAGS.batch_size)
  x = x.reshape(FLAGS.batch_size, -1)
  out_dim = t.shape[1]
  in_dim = x.shape[1]

  mlp = MLP(in_dim, dnn_hidden_units, out_dim, FLAGS.neg_slope)
  loss_funct = CrossEntropyModule()

  # First pass of the data
  y = mlp.forward(x)
  loss = loss_funct.forward(y, t)
  mlp.backward(loss_funct.backward(y, t))
  for mod in mlp.modules:
    if type(mod) == LinearModule:
      mod.params['weight'] -= FLAGS.learning_rate * mod.grads['weight']
      mod.params['bias'] -= FLAGS.learning_rate * mod.grads['bias']
  train_loss.append(loss)
  train_acc.append(accuracy(y, t))
  x, t = cifar10_set['test'].images, cifar10_set['test'].labels
  x = x.reshape(x.shape[0], -1)
  y = mlp.forward(x)
  test_acc.append(accuracy(y, t))
  print("The accuracy at step, " + str(0) + " is : " + str(test_acc[-1]))

  # loop through till steps are all done
  for i in range(1, FLAGS.max_steps+1):
    x, t = cifar10_set['train'].next_batch(FLAGS.batch_size)
    x = x.reshape(FLAGS.batch_size, -1)
    y = mlp.forward(x)
    loss = loss_funct.forward(y,t)
    mlp.backward(loss_funct.backward(y,t))
    #Update the weights
    for mod in mlp.modules:
      if type(mod) == LinearModule:
        mod.params['weight'] -= FLAGS.learning_rate * mod.grads['weight']
        mod.params['bias'] -= FLAGS.learning_rate * mod.grads['bias']
    # Evaluation and acc/loss saving
    if i % FLAGS.eval_freq == 0:
      train_loss.append(loss)
      train_acc.append(accuracy(y,t))
      x,t = cifar10_set['test'].images, cifar10_set['test'].labels
      x = x.reshape(x.shape[0], -1)
      y = mlp.forward(x)
      test_acc.append(accuracy(y,t))
      print("The accuracy at step, " + str(i) + " is : " + str(test_acc[-1]))

  #Plotting the accuracy of test and train:
  plt.figure(0)
  plt.plot(
    np.arange(0, len(train_acc) * FLAGS.eval_freq * FLAGS.batch_size, FLAGS.eval_freq * FLAGS.batch_size) / cifar10_set[
      'train'].num_examples, train_acc, label='Train')
  plt.plot(
    np.arange(0, len(train_acc) * FLAGS.eval_freq * FLAGS.batch_size, FLAGS.eval_freq * FLAGS.batch_size) / cifar10_set[
      'train'].num_examples, test_acc, label='Test')
  plt.xlabel('Epoch')
  plt.ylabel('Accuracy')
  plt.title('Accuracy of Train and Test Set Through Training')
  plt.legend()
  plt.savefig('Accuracy_basic1.png')
  # plt.show()

  plt.figure(1)
  plt.plot(np.arange(0, len(train_loss)*FLAGS.eval_freq* FLAGS.batch_size, FLAGS.eval_freq* FLAGS.batch_size)/cifar10_set['train'].num_examples, train_loss, label = 'Train')
  plt.xlabel('Epoch')
  plt.ylabel('Loss')
  plt.title('Loss Through Training')
  plt.savefig('Loss_basic1.png')
  # plt.show()

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
