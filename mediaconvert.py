import pydub
import os


def convert_1_flac_wav(source_path,destination_path):
    '''Convert the  flac file in source_path and save it as a wav file in destination_path'''
    source_path=source_path.strip()
    destination_path=destination_path.strip()
    if(source_path is None or len(source_path) == 0):
        raise Exception("source path is an Invalid Path to media file")
    if(destination_path is None or len(destination_path) == 0):
        raise Exception("destination path is an Invalid Path to a media file")

    mediaObject=pydub.AudioSegment.from_file(source_path,"flac") #todo 1: Find the API method to do this
    mediaObject.export(destination_path,format="wav")
    return

def convert_all(source_dir):
    '''Convert all the flac files in source_dir into wav files in data/HumanAudio'''
    source_dir=source_dir.strip()
    if(source_dir is None or len(source_dir)==0):
        raise Exception("source dir is an invalid path")

    for file in os.listdir(source_dir):
        source_path=os.path.join(source_dir,file)
        file=file.replace("flac","wav")
        destination_path=os.path.join("Data/HumanAudio",file)
        convert_1_flac_wav(source_path,destination_path)

    return
