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

import torch
import torch.nn as nn

################################################################################

class VanillaRNN(nn.Module):

    def __init__(self, seq_length, input_dim, num_hidden, num_classes, device='cpu'):
        super(VanillaRNN, self).__init__()

        # Things to save(may be useful later)
        self.seq_len = seq_length
        self.dev = device
        self.s_o = num_classes
        self.num_hidden = num_hidden
        initial_param = math.sqrt(6/(num_hidden + input_dim))

        # Trainable Parameters
        self.b_h = nn.Parameter(torch.zeros(num_hidden,))
        self.W_hx = nn.Parameter(torch.Tensor(num_hidden, input_dim).uniform_(0,0.001))
        self.W_hh = nn.Parameter(torch.Tensor(num_hidden, num_hidden).uniform_(0,0.001))
        self.b_p = nn.Parameter(torch.zeros(num_classes,))
        self.W_ph = nn.Parameter(torch.Tensor(num_classes, num_hidden).uniform_(0,0.001))

    def forward(self, x):
        # print(x.size(0))
        h_t = torch.autograd.Variable(torch.zeros(size = (x.size(0), self.num_hidden)), requires_grad = True)
        self.h_list = []
        self.h_list.append(h_t)
        self.h_list[-1].retain_grad()
        for i in range(0,self.seq_len):
            h_t = torch.tanh((x[:,i, None] @ self.W_hx.t()) + (self.h_list[-1] @ self.W_hh) + self.b_h)
            self.h_list.append(h_t)
            self.h_list[-1].retain_grad()
        p = h_t @ self.W_ph.t() + self.b_p
        return p