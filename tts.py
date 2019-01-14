import subprocess
import os

def make_tts_1(text,destination_path):
    '''Convert the text into audio file and save it in destination path. Note that it uses espeak command in background'''
    text=text.strip()
    destination_path = destination_path.strip()
    if (len(text) == 0):
        raise Exception("Text is Empty")
    if (len(destination_path) == 0):
        raise Exception("destination path is an Invalid Path to a media file")
    subprocess.call(["espeak","-w "+destination_path,"-g 50",text]) #convert the file using espeak subprocess
    return

def make_tts(file_path):
    '''Read the file and convert the text into media files and save it in Data/TTS'''
    file_path=file_path.strip()
    if(len(file_path)==0):
        raise Exception("filepath is invalid")
    file=open(file_path,"r")
    for line in file.readlines():
        offset= len("3081-166546-0000")
        line_no=line[:offset] # avoid the whitespace between the number and offset
        text=line[offset+1:]
        destination_path=os.path.join("Data/TTS",line_no+".wav")
        make_tts_1(text,destination_path)

    return
