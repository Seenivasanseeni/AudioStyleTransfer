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

def train(model,optimizer,loss,epoch=10,debug=False):
    lossData=[]
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
        plt.plot([i for i in range(len(lossData))], lossData)
        lossData=[] #reset lossData for the next epoch
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

def test(model,debug=False):
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


def load_model(args):
    '''
    :returns the model based on passed arguments
    :param args:
    :return:
    '''
    if(args.model=="basic"):
        import AutoEncoder
        return AutoEncoder.setupModel()
    raise Exception("ModelL:{} not specified or doesn't exist".format(args.model))






if __name__ == '__main__':
    argParse = argparse.ArgumentParser()
    argParse.add_argument("--model", type=str, nargs="?")
    args = argParse.parse_args()
    # load the model
    model,optimizer,loss=load_model(args)
    train(model,optimizer,loss,epoch=10)
    test(model,debug=False)
