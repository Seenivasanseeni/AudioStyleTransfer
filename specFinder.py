from scipy.io import wavfile
import os
import matplotlib.pyplot as plt


def tts_spec():
    data=[]
    for file in os.listdir("Data/TTS/"):
        sF,sD=wavfile.read("Data/TTS/"+file)
        data.append([sF,len(sD)])
    return data

def human_audio_spec():
    data=[]
    for file in os.listdir("Data/HumanAudio/"):
        sF,sD=wavfile.read("Data/HumanAudio/"+file)
        data.append([sF, len(sD)])
    return data

a=tts_spec()
b=human_audio_spec()
print("Min","Max")
print(min(a),max(a))
print(min(b),max(b))
