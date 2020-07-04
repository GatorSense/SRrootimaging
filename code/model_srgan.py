#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul  4 11:40:05 2020

@author: jruizmunoz
"""

import torch.nn as nn

##############################
#        Generator
##############################

class AutoGen2(nn.Module):
    # nc: Number of channels in the training images
    # ndf: Size of feature maps in discriminator
    def __init__(self, ngpu = 1, nc_in = 1, nc_out = 1, scale_factor = 4):
        super(AutoGen2, self).__init__()
        self.ngpu = ngpu
        # input is (nc) x 64 x 64
        """
        self.layers = nn.Sequential(nn.Conv2d(nc_in, 64, 9, 1, 4, bias=False),
                                    nn.ReLU(),
                                    nn.Conv2d(64, 32, 5, 1, 2, bias=False),
                                    nn.ReLU(),
                                    nn.Conv2d(32, nc_out, 5, 1, 2, bias=False),
                                    nn.ReLU())
        """
        self.layers = nn.Sequential(nn.Conv2d(nc_in, 64, 9, 1, 4, bias=False),
                                    nn.ReLU(),
                                    nn.Conv2d(64, 32, 5, 1, 2, bias=False),
                                    nn.ReLU(),
                                    nn.Conv2d(32, nc_out, 5, 1, 2, bias=False))
        # self.layer3 = nn.Sequential(nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False))
        # nn.ConvTranspose2d(32, 1, kernel_size=9, stride=scale_factor, padding=9//2,
        #                                    output_padding=scale_factor-1)

    def forward(self, input):
        x = self.layers(input)
        return x

##############################
#        Discriminator
##############################
    
class Discriminator2(nn.Module):
    def __init__(self, nc =1, crop_size = 64):
        super(Discriminator2, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(nc, 64, 9, 1, 4, bias=False),
            nn.ReLU(),
            nn.Conv2d(64, 32, 5, 1, 2, bias=False),
            nn.MaxPool2d((2,2)),
            nn.ReLU(),
            nn.Conv2d(32, nc, 5, 1, 2, bias=False),
            nn.MaxPool2d((2,2)),
            nn.ReLU()
        )
        # self.fc = nn.Linear((crop_size//16)**2,2)
        self.fc = nn.Conv2d(1, 1,crop_size//4, bias=False)
    def forward(self, x):
        batch_size = x.size(0)
        x = self.net(x)
        # x = x.view(batch_size,-1)
        x = self.fc(x)
        x = x.view(batch_size,-1)
        return x

