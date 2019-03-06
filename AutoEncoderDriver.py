import argparse
import datadriver
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
import random
from config import size
import numpy as np

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

def train(model,optimizer,loss,args,epoch=10,debug=False):
    lossData=[]
    total_lossData=[]
    for e in range(epoch):
        data = datadriver.CustomDataset()
        total_loss=0
        for index,(human_audio,tts_audio) in enumerate(data):
            if(debug):
                print("Inputs are of shape",human_audio.shape,tts_audio.shape)
            model.zero_grad()
            generated_audio = model(tts_audio,debug=debug)
            reconstruction_loss_euclidean=loss(torch.Tensor(generated_audio),torch.Tensor(human_audio))
            reconstruction_loss=torch.mean(torch.pow(reconstruction_loss_euclidean,2))
            reconstruction_loss.backward()
            optimizer.step()
            lossData.append(reconstruction_loss.data.numpy())
            total_loss+=reconstruction_loss.data
        print("Epoch e:{} Loss:{}".format(e,total_loss))
        total_loss=0 #reset total loss
        total_lossData.append(total_loss)
        plt.plot([i for i in range(len(lossData))], lossData)
        lossData=[] #reset lossData for the next epoch
    plt.plot([i for i in range(len(total_lossData))],total_lossData,color="red")
    plt.savefig("Graphs/lossGraph"+args.model+"-"+args.name+".jpg")
    plt.show()

def reconstruct_wav(Zxx,name):
    '''
    given a fourier matrix , find the inverse matrix to find the source
    :param Zxx:
    :return:
    '''
    global size
    xn=np.random.rand(size)
    no_iterations=200
    for i in range(no_iterations):
        stft_xn=librosa.stft(xn)
        angle_xn = np.angle(stft_xn)
        xn=librosa.istft(Zxx*np.exp(1j*angle_xn))

    wavfile.write(name,16000,xn)
    return

def test(model,debug=False):
    data = datadriver.CustomDataset(test=True)
    generated_audio_=None
    tts_audio_ = None
    for index,(human_audio,tts_audio) in enumerate(data):
        generated_audio = model(tts_audio, debug=debug)
        generated_audio = to_numpy(generated_audio)
        reconstruct_wav(generated_audio, name="gen.wav")
        reconstruct_wav(tts_audio, name="tts.wav")
        reconstruct_wav(human_audio, name="human.wav")
        break;

    '''plot the figures on matplotlib'''
    return


def load_model(args):
    '''
    :returns the model based on passed arguments
    :param args:
    :return:
    '''
    if(args.model=="basic"):
        import AutoEncoder
        return AutoEncoder.setupModel()
    raise Exception("Model:{} not specified or doesn't exist".format(args.model))






if __name__ == '__main__':
    argParse = argparse.ArgumentParser()
    argParse.add_argument("--model", type=str, nargs="?")
    argParse.add_argument("--name", type=str, nargs="?")
    args = argParse.parse_args()
    # load the model
    model,optimizer,loss=load_model(args)
    #train(model,optimizer,loss,args,epoch=3)
    test(model,debug=False)
