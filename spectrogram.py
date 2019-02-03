from scipy.io import wavfile
import matplotlib.pyplot as plt
import os
import librosa

def spectrogram_1(source_path,destination_path):
    '''Read the media file wav file and store its spectrogram in destination_path'''
    signalData, samplingFrequency = librosa.load(source_path,sr=16000)
    plt.specgram(signalData,Fs=samplingFrequency)
    plt.savefig(destination_path)
    plt.close()
    return

def spectrogram_tts():
    '''Read all the media files in Data/TTS and save the spectrogram in Data/Spectrogram/TTS'''
    for file in os.listdir("Data/TTS"):
        file_path=os.path.join("Data/TTS",file)
        destination_path="Data/Spectrogram/TTS/"+file+".jpg"
        spectrogram_1(file_path,destination_path)
    return

def spectrogram_human_audio():
    '''Read all the media files in Data/HumanAudio and save the spectrogram in Data/Spectrogram/HumanAudio'''
    for file in os.listdir("Data/HumanAudio"):
        file_path=os.path.join("Data/HumanAudio",file)
        destination_path="Data/Spectrogram/HumanAudio/"+file+".jpg"
        spectrogram_1(file_path,destination_path)
    return
