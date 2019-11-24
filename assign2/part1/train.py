################################################################################
# MIT License
# 
# Copyright (c) 2019
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to conditions.
#
# Author: Deep Learning Course | Fall 2019
# Date Created: 2019-09-06
################################################################################

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import time
from datetime import datetime
import numpy as np

import sys
sys.path.append("..")

import torch
from torch.utils.data import DataLoader

from part1.dataset import PalindromeDataset
from part1.vanilla_rnn import VanillaRNN
from part1.lstm import LSTM
import matplotlib.pyplot as plt


# You may want to look into tensorboard for logging
# from torch.utils.tensorboard import SummaryWriter

################################################################################

def train(config):

    assert config.model_type in ('RNN', 'LSTM')

    # Set the model
    # config.model_type = 'RNN'
    config.model_type = 'LSTM'

    # Initialize the device which to run the model on
    device = torch.device(config.device)

    # p_len_list = [4,5,6,7,8,9,10,15,20,25,30,35,40,45,50]
    p_len_list = [4, 9, 14, 15, 16, 17, 18, 19, 24, 29, 34, 39, 44, 49]
    p_acc = []
    # config.batch_size = 150
    for in_len in p_len_list:
        print("The Palendrom length is: " + str(in_len+1))
        config.input_length = in_len

        np.random.seed(42)

        if config.model_type == 'RNN':
            # Initialize the RNN model that we are going to use
            model = VanillaRNN(config.input_length, config.input_dim, config.num_hidden, config.num_classes, device)
        elif config.model_type == 'LSTM':
            # Initialize the LSTM model that we are going to use
            model = LSTM(config.input_length, config.input_dim, config.num_hidden, config.num_classes, device)

        # Initialize the dataset and data loader (note the +1)
        dataset = PalindromeDataset(config.input_length+1)
        data_loader = DataLoader(dataset, 4000, num_workers=1)

        (test_inputs, test_targets) = next(iter(data_loader))

        data_loader = DataLoader(dataset, config.batch_size, num_workers=1)

        #Test set ~ 4000

        # Setup the loss and optimizer
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.RMSprop(model.parameters(), lr=config.learning_rate)

        #Data Storage
        train_acc = []
        test_acc = []
        # test_one = False
        for step, (batch_inputs, batch_targets) in enumerate(data_loader):

            # Only for time measurement of step through network
            t1 = time.time()
            model_out = model.forward(batch_inputs)
            loss = criterion(model_out, batch_targets)
            optimizer.zero_grad()
            loss.backward()

            ############################################################################
            # QUESTION: what happens here and why?  It seems that its giving the gradient an upper limit so that there arent exploading gradients
            ############################################################################
            torch.nn.utils.clip_grad_norm(model.parameters(), max_norm=config.max_norm)
            ############################################################################

            # Add more code here ...

            optimizer.step()

            loss = loss.item()
            # loss_over_time.append(loss)

            accuracy = np.average((torch.max(model_out, 1)[1] == batch_targets))
            train_acc.append(accuracy)
            # Just for time measurement
            t2 = time.time()
            examples_per_second = config.batch_size/float(t2-t1)

            if step % 10 == 0:

                # print("[{}] Train Step {:04d}/{:04d}, Batch Size = {}, Examples/Sec = {:.2f}, "
                #       "Accuracy = {:.2f}, Loss = {:.3f}".format(
                #         datetime.now().strftime("%Y-%m-%d %H:%M"), step,
                #         config.train_steps, config.batch_size, examples_per_second,
                #         accuracy, loss
                # ))

                model_out = model.forward(test_inputs)
                accuracy = np.average((torch.max(model_out, 1)[1] == test_targets))
                # if accuracy == 1 and not test_one:
                #     test_one = True
                # print("The currecnt test set accuracy is: " + str(accuracy))
                if (step > 2500 and in_len < 9) or (step > 4000 and in_len >= 9):
                    if accuracy == 1:
                        print(str(step))
                        print("We have convergence due to 1,  accuracy is: " + str(accuracy))
                        p_acc.append(accuracy)
                        test_acc = []
                        break
                    elif not all(x <= accuracy for x in test_acc[-5:]):
                        print(str(step))
                        print("We have convergence due to being worse than last 5, accuracy is: " + str(accuracy) + ". Best is:  " + str(max(test_acc)))
                        p_acc.append(max(test_acc))
                        test_acc = []
                        break
                    elif np.var(test_acc[-5:]) < 0.001:
                        print(str(step))
                        print("We have convergence due to variance low, accuracy is: " + str(accuracy) + ". Best is:  " +  str(max(test_acc)))
                        p_acc.append(max(test_acc))
                        test_acc = []
                        break
                    else:
                        test_acc.append(accuracy)
                else:
                    test_acc.append(accuracy)
            if step == config.train_steps:
                print("We havent converged, but we ran out of time")
                p_accc.append(max(test_acc))
                test_acc = []
            ## Another stopping could be loss < 0.015?
            ## This is stopping after training acc is 1 for 50 steps.
            # if step % 50 == 0:
            #     if sum(train_acc) == len(train_acc) and train_acc[-1] == 1 and step > 1000:
            #         print("We have convergence" + str(sum(train_acc)))
            #         model_out = model.forward(test_inputs)
            #         accuracy = np.average((torch.max(model_out, 1)[1] == test_targets))
            #         print(accuracy)
            #         p_acc.append(accuracy)
            #         train_acc = []
            #         break
            #     train_acc = []
            if step == config.train_steps:
                # If you receive a PyTorch data-loader error, check this bug report:
                # https://github.com/pytorch/pytorch/pull/9655
                break
        # break

    print('Done training.')
    plt.plot(p_len_list, p_acc)
    plt.title("Accuracy for Different Lengths of Palindrome for Test Set")
    plt.xlabel("Palindrome Length")
    plt.ylabel("Accuracy")
    plt.savefig('figs/LSTM_Acc_basic.png')
    plt.show()



 ################################################################################
 ################################################################################

if __name__ == "__main__":

    # Parse training configuration
    parser = argparse.ArgumentParser()

    # Model params
    parser.add_argument('--model_type', type=str, default="RNN", help="Model type, should be 'RNN' or 'LSTM'")
    parser.add_argument('--input_length', type=int, default=10, help='Length of an input sequence')
    parser.add_argument('--input_dim', type=int, default=1, help='Dimensionality of input sequence')
    parser.add_argument('--num_classes', type=int, default=10, help='Dimensionality of output sequence')
    parser.add_argument('--num_hidden', type=int, default=128, help='Number of hidden units in the model')
    parser.add_argument('--batch_size', type=int, default=128, help='Number of examples to process in a batch')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--train_steps', type=int, default=10000, help='Number of training steps')
    parser.add_argument('--max_norm', type=float, default=10.0)
    parser.add_argument('--device', type=str, default="cuda:0", help="Training device 'cpu' or 'cuda:0'")

    config = parser.parse_args()

    # Train the model
    train(config)