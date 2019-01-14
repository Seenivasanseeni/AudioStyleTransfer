from scipy.io import wavfile
import matplotlib.pyplot as plt
import os

def spectrogram_1(source_path,destination_path):
    '''Read the media file wav file and store its spectroggram in destination_path'''
    samplingFrequency, signalData = wavfile.read(source_path)
    plt.specgram(signalData,Fs=samplingFrequency)
    plt.savefig(destination_path)
    return

def spectrogram_tts():
    '''Read all the media files in Data/TTS and save the spcectrograms in Data/Spectrogram/TTS'''
    for file in os.listdir("Data/TTS"):
        file_path=os.path.join("Data/TTS",file)
        destination_path="Data/Spectrogram/TTS/"+file+".jpg"
        spectrogram_1(file_path,destination_path)
    return
