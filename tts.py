import subprocess

def make_tts_1(text,destination_path):
    '''Convert the text into audio file and save it in destination path. Note that it uses espeak command in background'''
    text=text.strip()
    destination_path = destination_path.strip()
    if (len(text) == 0):
        raise Exception("Text is Empty")
    if (len(destination_path) == 0):
        raise Exception("destination path is an Invalid Path to a media file")
    subprocess.call(["espeak","-w "+destination_path,text]) #convert the file using espeak subprocess
    return
