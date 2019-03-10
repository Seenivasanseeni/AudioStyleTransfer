import  torch.nn as nn
import  torch.nn.functional as F
import torchvision.transforms.functional as TF
import os
import matplotlib.pyplot as plt
import  torch
import torch.optim as optim
import torch.nn.modules.distance as distance
import datadriver
import pdb
import librosa
from scipy.io import wavfile
import random

class AutoEncoder(nn.Module):
    '''
    a convolutional Autoencoder class
    '''
    def __init__(self):
        '''
        Initializes the autoencoder. Initializes layers of encoders and decoders.
        '''
        super(AutoEncoder,self).__init__()
        self.conv1 = nn.Conv2d(1, 3, kernel_size=(8,8))
        self.conv2 = nn.Conv2d(3, 5, kernel_size=(2, 2),stride=2)
        self.conv2_t = nn.ConvTranspose2d(5,3,kernel_size=(2,2),stride=2)
        self.conv1_t = nn.ConvTranspose2d(3,1,  kernel_size=(8,8))

    def forward(self, x,debug=False):
        '''
        gets the input and pass it along different convolutional encoders and decoders and returns the tensor result.
        :param x:
        :param debug:
        :return:
        '''
        xshape = list(x.shape)
        inputshape = [1,1]
        inputshape.extend(xshape)
        if(debug):
            print("Reshaping to ",inputshape)
        x= torch.from_numpy(x).float()
        x=x.view(inputshape)

        if(debug):
            print(x.shape)
        x =self.conv1(x)
        if (debug):
            print(x.shape)
        x = self.conv2(x)
        if (debug):
            print(x.shape)

        x = self.conv2_t(x)
        if (debug):
            print(x.shape)

        x = self.conv1_t(x)
        if (debug):
            print(x.shape)
        return x.view(xshape)

model,optimizer,pairwiseDistance=None,None,None

def setupModel():
    '''
    This method will be called by the driver to setup a model,optimizer, loss functions which can be used to train the model
    :return:
    '''
    model=AutoEncoder()
    optimizer = optim.Adam(params=model.parameters(),lr=0.5)
    pairwiseDistance=distance.PairwiseDistance(p=2)
    return model,optimizer,pairwiseDistance


