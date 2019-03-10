import os
import librosa
import pdb
import numpy as np
from librosa import stft
import matplotlib.pyplot as plt
from scipy.io import wavfile
import random
from config import size
maxlength = 0


def make_fixed_audio_size(a,b):
    '''
    make two arrays to be same size by appending 0
    :param a: numpy array
    :param b: numpy array
    :return:
    '''
    global maxlength,size
    maxlength = max(maxlength,size) # 279344 is the maximum number of samples in this dataset
    na = list(a)
    nb = list(b)
    for _ in range(len(a), size):
        na.append(0)
    for _ in range(len(b), size):
        nb.append(0)
    return np.array(na),np.array(nb)

class CustomDataset():
    '''
    Custom dataset class for fetching pair at at time
    '''

    def __init__(self,test=False):
        self.list = os.listdir("Data/HumanAudio")
        self.test = test
        if(test):
            random.shuffle(self.list)

    def __getitem__(self, ind):
        '''
        Get the data read the file and send it as a fourier transform matrix
        :param ind: item no
        :return:
        '''
        file = self.list[ind]
        human_audio_path = "Data/HumanAudio/" + file
        tts_path = "Data/TTS/" + file
        human_audio = librosa.load(human_audio_path,sr = 16000)
        tts = librosa.load(tts_path,sr = 16000)
        h,t= make_fixed_audio_size(human_audio[0],tts[0]) #tested it didnt scramble the audio
        wavfile.write("h.wav", rate=16000, data=h)
        wavfile.write("t.wav", rate=16000, data=t)
        h_mag,h_phase = librosa.magphase(stft(h)) #split the magnitude and phase from fourier matrix
        t_mag,t_phase = librosa.magphase(stft(t))
        return h_mag,t_mag


def test():
    c = CustomDataset()
    for i,(h_z,t_z) in enumerate(c):
        print(i,h_z.shape,t_z.shape)

def testVariousIterations():
    for j in range(10):
        for i,(h_z,t_z) in enumerate(CustomDataset()):
            if i==0:
                print(j,h_z.shape,t_z.shape)


if __name__ == '__main__':
    testVariousIterations()
