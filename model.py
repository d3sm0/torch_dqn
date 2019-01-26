#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import torch.nn.functional
import numpy as np


def conv2d_size_out(size, kernel_size=4, stride=2):
    return (size - (kernel_size - 1) - 1) // stride + 1


class QNetwork(torch.nn.Module):
    def __init__(self, obs_shape, act_shape):
        super(QNetwork, self).__init__()
        h, w, c = obs_shape
        self.conv_0 = torch.nn.Conv2d(in_channels=c, out_channels=16, kernel_size=4, stride=2, padding=0)
        self.conv_1 = torch.nn.Conv2d(in_channels=16, out_channels=32, kernel_size=2, stride=2, padding=0)
        # conv_w = conv2d_size_out(conv2d_size_out(w))
        # conv_h = conv2d_size_out(conv2d_size_out(h))
        # input_shape = conv_w * conv_h * 32
        self.fc_0 = torch.nn.Linear(32, 64)
        # self.fc_0 = torch.nn.Linear(obs_shape, 64)
        # self.fc_1 = torch.nn.Linear(64, 64)

        # out_shape = conv_w * conv_h * 32
        self.out = torch.nn.Linear(64, act_shape)

    def forward(self, x):
        # x = x.view((x.size(0), -1))
        b, h, w, c = x.shape
        x = x.view((b, c, h, w))
        x = torch.nn.functional.relu(self.conv_0(x))
        x = torch.nn.functional.relu(self.conv_1(x))
        x = x.view((-1, x.shape[1]))
        x = torch.nn.functional.relu(self.fc_0(x))
        # x = torch.nn.functional.relu(self.fc_0(x))
        # x = torch.nn.functional.relu(self.fc_1(x))
        x = self.out(x)
        return x
