import  torch.nn as nn
import  torch.functional as F
import torchvision.transforms.functional as TF
import os
import matplotlib.pyplot as plt
import  torch
import torch.optim as optim
import torch.nn.modules.distance as distance
import datadriver

class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder,self).__init__()
        #images will be of shape (480, 640, 3)
        self.conv1= nn.Conv2d(1,3,kernel_size=(20,20))
        self.conv1_t = nn.ConvTranspose2d(3,1,  kernel_size=(20,20))

    def forward(self, x,debug=False):
        xshape = list(x.shape)
        inputshape = [1,1]
        inputshape.extend(xshape)
        if(debug):
            print("Reshaping to ",inputshape)
        x= torch.from_numpy(x).float()
        x=x.view(inputshape) # find shape

        if(debug):
            print(x.shape)
        x=self.conv1(x)
        if (debug):
            print(x.shape)
        x = self.conv1_t(x)
        if (debug):
            print(x.shape)
        return x.view(xshape)

def get_input_pair():
    for file in os.listdir("Data/Spectrogram/HumanAudio/"):
        human_audio_path = "Data/Spectrogram/HumanAudio/" + file
        tts_audio_path = "Data/Spectrogram/TTS/" + file
        human_audio_data=plt.imread(human_audio_path)
        tts_audio_data=plt.imread(tts_audio_path)
        yield  human_audio_data,tts_audio_data


model=AutoEncoder()
optimizer = optim.Adam(params=model.parameters(),lr=0.5)
pairwiseDistance=distance.PairwiseDistance(p=2)
def train(epoch=10,debug=False):
    for e in range(epoch):
        data = datadriver.CustomDataset()
        for index,(human_audio,tts_audio) in enumerate(data):
            if(debug):
                print("Inputs are of shape",human_audio.shape,tts_audio.shape)
            model.zero_grad()
            generated_audio = model(tts_audio,debug=debug)
            reconstruction_loss_euclidean=pairwiseDistance(torch.Tensor(generated_audio),torch.Tensor(human_audio))
            loss=torch.mean(torch.pow(reconstruction_loss_euclidean,2))
            loss.backward()
            optimizer.step()
        print(loss.data)

if __name__ == '__main__':
    train(debug=False)
