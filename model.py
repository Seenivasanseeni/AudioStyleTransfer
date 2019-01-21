import torch
from torch import nn, optim
import torch.functional as F

from torch.autograd.variable import Variable
from torchvision import transforms, datasets
import os
import matplotlib.pyplot as plt

DATA_FOLDER = 'Data/Spectrogram/HumanAudio'


class GeneratorNet(torch.nn.Module):
    '''A Gerenator net that combines produces a distribution AB where A is real and B is fake distribution.
    First reduce the size and bring back the size'''

    def __init__(self):
        super(GeneratorNet, self).__init__()
        n_features = 640*480
        n_out = 640*480

        self.hidden0 = nn.Sequential( #todo 1 change the neural networks to convolutional
            nn.Linear(n_features, 256),
            nn.ReLU(0.2)
        )

        self.hidden1 = nn.Sequential(
            nn.Linear(256, 512),
            nn.ReLU(0.2)
        )
        self.hidden2 = nn.Sequential(
            nn.Linear(512, 1024),
            nn.ReLU(0.2)
        )

        self.out = nn.Sequential(
            nn.Linear(1024, n_out),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.hidden0(x)
        x = self.hidden1(x)
        x = self.hidden2(x)
        x = self.out(x)
        return x


class DiscriminatorNet(torch.nn.Module):
    """
        A discrimnator network that accepts a spectrogram and outputs whether that is a valid image or not
    """

    def __init__(self):
        super(DiscriminatorNet, self).__init__()
        n_features =640*480
        n_out = 1

        self.hidden0 = nn.Sequential(
            nn.Linear(n_features, 1024),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3)
        )
        self.hidden1 = nn.Sequential(
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3)
        )
        self.hidden2 = nn.Sequential(
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3)
        )
        self.out = nn.Sequential(
            torch.nn.Linear(256, n_out),
            torch.nn.Sigmoid()
        )

    def forward(self, x):
        x = self.hidden0(x)
        x = self.hidden1(x)
        x = self.hidden2(x)
        x = self.out(x)
        return x


discriminator = DiscriminatorNet()
generator = GeneratorNet()


d_optimizer= optim.Adam(discriminator.parameters(),lr=0.001)
g_optimizer=optim.Adam(generator.parameters(),lr=0.001)
loss=nn.BCELoss()


def fetch_data():
    human_audio_data=[]
    tts_data=[]
    for file in os.listdir("Data/Spectrogram/HumanAudio"):
        human_audio_path=os.path.join("Data/Spectrogram/HumanAudio",file)
        tts_path=os.path.join("Data/Spectrogram/TTS",file)
        human_audio_data.append(plt.imread(human_audio_path))
        tts_data.append(plt.imread(tts_path))
    return human_audio_data,tts_data

human_audio_data,tts_data=fetch_data()

epoch=3

def train():
    for e in range(epoch):
        d_optimizer.zero_grad()
        g_optimizer.zero_grad()
        generated_data=generator(tts_data)
        discriminator_generator_data = discriminator(generated_data)
        discriminator_real_data=discriminator(human_audio_data)
        loss_generator= nn.BCELoss() # log(1-discriminator_generator_data) todo
        loss_discriminator = nn.BCELoss() # - log( discriminator_real_data) -log(1-discriminator_generator_data) todo
        loss_generator.backward()
        loss_discriminator.backward()
        g_optimizer.step()
        d_optimizer.step()
        print(loss_generator,loss_discriminator)

train()
