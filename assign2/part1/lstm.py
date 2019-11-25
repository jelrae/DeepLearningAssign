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

class LSTM(nn.Module):

    def __init__(self, seq_length, input_dim, num_hidden, num_classes, device='cpu'):
        super(LSTM, self).__init__()

        # Things to save(may be useful later)
        self.seq_len = seq_length
        self.dev = device
        self.s_o = num_classes
        self.num_hidden = num_hidden

        # Input Modulation gate
        self.W_gx = nn.Parameter(torch.Tensor(num_hidden, input_dim).normal_(0,0.001))
        self.W_gh = nn.Parameter(torch.Tensor(num_hidden, num_hidden).normal_(0,0.001))
        self.b_g = nn.Parameter(torch.zeros(num_hidden,))

        # Input gate
        self.W_ix = nn.Parameter(torch.Tensor(num_hidden, input_dim).normal_(0,0.001))
        self.W_ih = nn.Parameter(torch.Tensor(num_hidden, num_hidden).normal_(0,0.001))
        self.b_i = nn.Parameter(torch.zeros(num_hidden,))

        # Forget gate
        self.W_fx = nn.Parameter(torch.Tensor(num_hidden, input_dim).normal_(0,0.001))
        self.W_fh = nn.Parameter(torch.Tensor(num_hidden, num_hidden).normal_(0,0.001))
        self.b_f = nn.Parameter(torch.zeros(num_hidden,))

        # Output gate
        self.W_ox = nn.Parameter(torch.Tensor(num_hidden, input_dim).normal_(0,0.001))
        self.W_oh = nn.Parameter(torch.Tensor(num_hidden, num_hidden).normal_(0,0.001))
        self.b_o = nn.Parameter(torch.zeros(num_hidden,))

        # Prediction parameters
        self.W_ph = nn.Parameter(torch.Tensor(num_classes, num_hidden).normal_(0,0.001))
        self.b_p = nn.Parameter(torch.zeros(num_classes,))

    def forward(self, x):

        h_t = torch.autograd.Variable(torch.zeros(x.size(0), self.num_hidden), requires_grad = True)
        c_t = torch.zeros(x.size(0), self.num_hidden)
        self.h_list = []
        self.h_list.append(h_t)
        self.h_list[-1].retain_grad()

        for itter in range(0,self.seq_len):

            g = torch.tanh((x[:,itter, None] @ self.W_gx.t()) + (self.h_list[-1] @ self.W_gh) + self.b_g)
            i = torch.sigmoid((x[:, itter, None] @ self.W_ix.t()) + (self.h_list[-1] @ self.W_ih) + self.b_i)
            f = torch.tanh((x[:, itter, None] @ self.W_fx.t()) + (self.h_list[-1] @ self.W_fh) + self.b_f)
            o = torch.tanh((x[:, itter, None] @ self.W_ox.t()) + (self.h_list[-1] @ self.W_oh) + self.b_o)
            c_t = g * i + c_t * f
            h_t = torch.tanh(c_t) * o
            self.h_list.append(h_t)
            self.h_list[-1].retain_grad()
            p_t = h_t @ self.W_ph.t() + self.b_p

        return p_t