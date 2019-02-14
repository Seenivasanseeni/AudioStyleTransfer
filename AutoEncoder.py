import  torch.nn as nn
import  torch.functional as F
import torchvision.transforms.functional as TF
import os
import matplotlib.pyplot as plt
import  torch
import torch.optim as optim
import torch.nn.modules.distance as distance

class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder,self).__init__()
        #images will be of shape (480, 640, 3)
        self.conv1= nn.Conv2d(3,10,kernel_size=(50,50))
        self.conv2=nn.Conv2d(10,20,kernel_size=(30,30))
        self.conv3 = nn.Conv2d(20, 30, kernel_size=(10, 10))

        self.conv3_t=nn.ConvTranspose2d(30,20,kernel_size=(10,10))
        self.conv2_t = nn.ConvTranspose2d(20,10, kernel_size=(30,30))
        self.conv1_t = nn.ConvTranspose2d(10,3,  kernel_size=(50,50))

    def forward(self, x):
        x=TF.to_tensor(x)
        x=x.view([1,3,480,640])
        print(x.shape)
        x=self.conv1(x)
        print(x.shape)
        x=self.conv2(x)
        print(x.shape)
        x = self.conv3(x)
        print(x.shape)
        x = self.conv3_t(x)
        print(x.shape)
        x = self.conv2_t(x)
        print(x.shape)
        x = self.conv1_t(x)
        print(x.shape)
        return x

def get_input_pair():
    for file in os.listdir("Data/Spectrogram/HumanAudio/"):
        human_audio_path = "Data/Spectrogram/HumanAudio/" + file
        tts_audio_path = "Data/Spectrogram/TTS/" + file
        human_audio_data=plt.imread(human_audio_path)
        tts_audio_data=plt.imread(tts_audio_path)
        yield  human_audio_data,tts_audio_data


model=AutoEncoder()
optimizer = optim.Adam(params=model.parameters(),lr=0.01)

def train(epoch=10):
    for e in range(epoch):
        for index,(human_audio,tts_audio) in enumerate(get_input_pair()):
            model.zero_grad()
            generated_audio = model(tts_audio)
            reconstruction_loss_euclidean=distance.PairwiseDistance(generated_audio,human_audio)
            loss= torch.mean(torch.pow(reconstruction_loss_euclidean,2))
            loss.backward()
            optimizer.step()
            input()

if __name__ == '__main__':
    train()
