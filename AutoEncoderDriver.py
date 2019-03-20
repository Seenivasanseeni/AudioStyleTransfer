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


def plot_learning_speed_graph(lossData,args):
    '''
    This algorithm plots the speed in which loss is progressing
    :param lossData: aggreated sum of losses, in which each loss is for one whole epoch
    :return:
    '''
    if(len(lossData)==0):
        return
    speedData=[]
    previousLoss=lossData[0]
    for l in lossData[1:]:
        if(l==0):
            continue
        speedData.append((previousLoss-l)/previousLoss*100)
        previousLoss=l
    plt.plot([i for i in range(len(speedData))],speedData,color="red")
    plt.savefig("Graphs/speedGraph-"+args.model+"-"+args.name+".jpg")
    plt.show()

def train(model,optimizer,loss,args,epoch=10,debug=False,onlyHuman=False):
    '''
        This function trains the model based on the passed parameters
    :param model: nn.Module
    :param optimizer: which optimizers the weights of @model
    :param loss: loss function based on which model's output is compared with its input
    :param args: system command line arguments parsed by argparse.ArgumentParser
    :param epoch: number of times the model has to be retrained on the model
    :param debug: If True, it periodically print debugging information
    :return: plotted graphs in Graphs/ directory
    '''
    lossData=[]
    total_lossData=[]
    for e in range(epoch):
        data = datadriver.CustomDataset()
        total_loss=0
        for index,(human_audio,tts_audio) in enumerate(data):
            if onlyHuman:
                tts_audio = human_audio
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
        total_lossData.append(total_loss)
        plt.plot([i for i in range(len(lossData))], lossData)
        lossData=[] #reset lossData for the next epoch
    plt.savefig("Graphs/lossGraph"+args.model+"-"+args.name+".jpg")
    plt.show()
    plot_learning_speed_graph(total_lossData,args)
    return

def reconstruct_wav(Zxx,name):
    '''
    given a fourier matrix , find the inverse matrix to find the source
    :param Zxx:
    :return:
    '''
    global size
    xn=np.random.rand(size)
    no_iterations=200
    if(name=="gen.wav"):
        no_iterations = 10
    for i in range(no_iterations):
        stft_xn=librosa.stft(xn)
        angle_xn = np.angle(stft_xn)
        xn=librosa.istft(Zxx*np.exp(1j*angle_xn))


    wavfile.write(name,16000,xn)
    return

def test(model,debug=False):
    '''
    tests the passed model and writes the files 1.gen.wav 2.human.wav 3.tts.wav
    :param model: a trained model
    :param debug: if True, prints debugging information.
    :return:
    '''
    data = datadriver.CustomDataset(test=True)
    generated_audio_=None
    tts_audio_ = None
    for index,(human_audio,tts_audio) in enumerate(data):
        generated_audio = model(tts_audio, debug=debug)
        generated_audio = to_numpy(generated_audio)
        reconstruct_wav(human_audio, name="gen.wav")
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
    if (args.model == "basic"):
        import BasicAutoEncoder
        return BasicAutoEncoder.setupModel()
    if (args.model == "mod"):
        import AutoEncoder
        return AutoEncoder.setupModel()
    raise Exception("Model:{} not specified or doesn't exist".format(args.model))


if __name__ == '__main__':
    argParse = argparse.ArgumentParser()
    argParse.add_argument("--model", type=str, nargs="?")
    argParse.add_argument("--name", type=str, nargs="?")
    argParse.add_argument("--epoch", type=int, nargs="?",default=10)
    args = argParse.parse_args() #parsing the arguments
    # load the model
    model,optimizer,loss=load_model(args)
    train(model, optimizer, loss, args, epoch=args.epoch, onlyHuman=True)
    print("Training with TTS")
    train(model, optimizer, loss, args, epoch=args.epoch, onlyHuman=False)

    test(model,debug=False)
