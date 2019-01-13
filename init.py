import os
import config

'''The Dataset Directory should be as follows

Data
|
---->Text
---->TTS
---->HumanAudioFlac
---->HumanAudio
---->Spectrogram
     |
      ---->TTS
      ---->HumanAudio
    

'''
def initialize_dataset_dir():
    '''Initialize the directory as per the above template '''
    os.makedirs("Data", exist_ok=True)
    os.makedirs("Data/Text", exist_ok=True)
    os.makedirs("Data/TTS", exist_ok=True)
    os.makedirs("Data/HumanAudioFlac", exist_ok=True)
    os.makedirs("Data/HumanAudio", exist_ok=True)
    os.makedirs("Data/Spectrogram/TTS", exist_ok=True)
    os.makedirs("Data/Spectrogram/HumanAudio", exist_ok=True)
    return

