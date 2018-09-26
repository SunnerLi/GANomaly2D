from lib.module import Encoder, Decoder, Discriminator
from lib.loss import GANLoss
from torch.optim import Adam
import torch.nn.functional as F
import torch.nn as nn
import itertools
import torch
import os

"""
    This script defines the structure of GANomaly2D 

    Author: SunnerLi
"""

class GANomaly2D(nn.Module):
    def __init__(self, r = 1, device = 'cpu'):
        super().__init__()
        # Store the variable
        self.r = r
        self.device = device

        # Define the hyper-parameters
        self.w_adv = 1.0
        self.w_con = 10.0
        self.w_enc = 10.0

        # Define the network structure
        self.G_E = Encoder(in_channels  = 3, nef = 64 // r)
        self.G_D = Decoder(out_channels = 3, ndf = 64 // r)
        self.E = Encoder(in_channels = 3, nef = 64 // r)
        self.f = Discriminator(in_channels = 3, out_channels = 1, nef = 64 // r, last = nn.Sigmoid)
        self.l1l_criterion = nn.L1Loss(reduction='sum')
        # self.l1l_criterion = nn.L1Loss(reduction='elementwise_mean')
        self.l2l_criterion = nn.MSELoss(reduction='sum')
        self.bce_criterion = GANLoss(use_lsgan = False)
        self.optim_G = Adam(itertools.chain(self.G_E.parameters(), self.G_D.parameters(), self.E.parameters()), lr = 0.0001)
        self.optim_D = Adam(self.f.parameters(), lr = 0.0001)
        self.to(self.device)

    def IO(self, path, direction = 'load'):
        """
            This function deal with input/output toward the hardware
            
            Arg:    path        (Str)   - The path you want to store/load
                    direction   (Str)   - The action you want to do, and the candidate is ['save', 'load']
                                          Default is load
        """
        if direction == 'load':
            if os.path.exists(path):
                self.load_state_dict(torch.load(path))  
        elif direction == 'save':
            torch.save(self.state_dict(), path)
        else:
            raise Exception("Unknown direction: {}".format(direction))

    def forward(self, x):
        self.x = x.to(self.device)
        self.z = self.G_E(self.x)
        self.x_ = self.G_D(self.z)
        self.z_ = self.E(self.x_)
        return self.z, self.z_

    def backward(self):
        # Update discriminator
        self.optim_D.zero_grad()
        true_pred = self.f(self.x)
        fake_pred = self.f(self.x_.detach())
        self.loss_D = self.bce_criterion(true_pred, True) + self.bce_criterion(fake_pred, False)
        self.loss_D.backward()
        self.optim_D.step()

        # Update encoder-decoder-encoder generator
        self.optim_G.zero_grad()
        fake_pred = self.f(self.x_)
        self.loss_G = self.bce_criterion(fake_pred, True) * self.w_adv + \
                        self.l2l_criterion(self.x_, self.x) * self.w_con + \
                        self.l1l_criterion(self.z_, self.z) * self.w_enc
        self.loss_G.backward()
        self.optim_G.step()

    def getLoss(self):
        return round(self.loss_G.item(), 5), round(self.loss_D.item(), 5)

    def getImg(self):
        return self.x, self.x_