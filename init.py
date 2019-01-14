import os
import config
import shutil
import mediaconvert,spectrogram,tts

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
    os.makedirs("Data/Text", exist_ok=True)
    os.makedirs("Data/TTS", exist_ok=True)
    os.makedirs("Data/HumanAudioFlac", exist_ok=True)
    os.makedirs("Data/HumanAudio", exist_ok=True)
    os.makedirs("Data/Spectrogram/TTS", exist_ok=True)
    os.makedirs("Data/Spectrogram/HumanAudio", exist_ok=True)
    return


def transfer_arrange(source_location):
    '''Copy the files from Dataset location into our data store as per the template
        This should separate the .flac files not .txt files
    '''
    source_location = source_location.strip()
    if(source_location is None or len(source_location) == 0):
        raise Exception("source location is an Invalid Path to dataset")

    #create directories if not already present
    os.makedirs("Data/Text", exist_ok=True)
    os.makedirs("Data/HumanAudioFlac", exist_ok=True)

    all_files=os.listdir(source_location)
    for file in all_files:
        source_file=os.path.join(source_location,file)
        destination_file=os.path.join("Data/HumanAudioFlac/",file) # 1 special case is with .txt file which is dealt below
        if (".trans.txt" in file):
            '''This is a subtitle file .txt'''
            destination_file=os.path.join("Data/Text",file)
        shutil.copy(source_file,destination_file)

    return

def setup():
    '''Copy, convert, TTS and spectrogram'''
    print("Setup Starting")
    initialize_dataset_dir()
    transfer_arrange(config.dataset_location)
    mediaconvert.convert_all("Data/HumanAudioFlac")
    tts.make_tts("Data/Text/3081-166546.trans.txt")
    spectrogram.spectrogram_tts()
    spectrogram.spectrogram_human_audio()
    print("Setup Done")

if __name__ == '__main__':
    setup()
