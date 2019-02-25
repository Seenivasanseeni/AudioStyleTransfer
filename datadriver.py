import os
import librosa
import pdb
import numpy as np
from scipy.signal import stft
import matplotlib.pyplot as plt
from scipy.io import wavfile

maxlength = 0


def make_fixed_audio_size(a,b):
    '''
    make two arrays to be same size by appending 0
    :param a: numpy array
    :param b: numpy array
    :return:
    '''
    global maxlength
    size = 280000
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

    def __init__(self):
        self.list = os.listdir("Data/HumanAudio")
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
        h,t= make_fixed_audio_size(human_audio[0],tts[0])
        #wavfile.write("h.wav", rate=16000, data=h)
        #wavfile.write("t.wav", rate=16000, data=t)
        h_f, h_t, h_z = stft(h)
        t_f, t_t, t_z = stft(t)
        return np.real(h_z),np.real(t_z)


def test():
    c = CustomDataset()
    for i,(h_z,t_z) in enumerate(c):
        print(i,h_z.shape,t_z.shape)


if __name__ == '__main__':
    test()
