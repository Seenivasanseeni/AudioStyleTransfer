import  torch.nn as nn
import  torch.functional as F
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

def to_numpy(t):
    '''
    convert the tensor t into numpy
    :param t: tensor
    :return:
    '''
    return torch.Tensor.numpy(t.data)


def get_input_pair():
    for file in os.listdir("Data/Spectrogram/HumanAudio/"):
        human_audio_path = "Data/Spectrogram/HumanAudio/" + file
        tts_audio_path = "Data/Spectrogram/TTS/" + file
        human_audio_data=plt.imread(human_audio_path)
        tts_audio_data=plt.imread(tts_audio_path)
        yield  human_audio_data,tts_audio_data

model,optimizer,pairwiseDistance=None,None,None

def setupModel():
    model=AutoEncoder()
    optimizer = optim.Adam(params=model.parameters(),lr=0.5)
    pairwiseDistance=distance.PairwiseDistance(p=2)
    return model,optimizer,pairwiseDistance


def train(epoch=10,debug=False):
    lossData=[]
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
            lossData.append(loss.data.numpy())
        print("Epoch e:{} Loss:{}".format(e,loss.data))
        plt.plot([i for i in range(len(lossData))], lossData)
        lossData=[]
    plt.savefig("lossGraph.jpg")
    plt.show()

def convert_to_wav(Zxx):
    '''
    given a fourier matrix , find the inverse matrix to find the source
    :param Zxx:
    :return:
    '''
    #todo not working all the audio params are not in sync
    wav = librosa.istft(Zxx)
    wavfile.write("out.wav",16000,wav)
    return

def test(debug=False):
    data = datadriver.CustomDataset(test=True)
    generated_audio_=None
    tts_audio_ = None
    for index,(human_audio,tts_audio) in enumerate(data):
        convert_to_wav(human_audio)
        generated_audio = model(tts_audio, debug=debug)
        generated_audio_,tts_audio_ = to_numpy(generated_audio),tts_audio
        break;
    '''plot the figures on matplotlib'''

    return

if __name__ == '__main__':
    model, optimizer, pairwiseDistance = setupModel()
    train(epoch=10)
    test(debug=False)
