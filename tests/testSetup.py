import unittest
import os
import init
import mediaconvert
import tts
import spectrogram

class TestInit(unittest.TestCase):
    '''To test the initializing of the project'''

    def test_dirstructure(self):
        '''check whether the dir is as per the template'''
        init.initialize_dataset_dir()
        assert os.path.exists("Data")
        assert os.path.exists("Data/Text")
        assert os.path.exists("Data/TTS")
        assert os.path.exists("Data/HumanAudioFlac")
        assert os.path.exists("Data/HumanAudio")
        assert os.path.exists("Data/Spectrogram")
        assert os.path.exists("Data/Spectrogram/TTS")
        assert os.path.exists("Data/Spectrogram/HumanAudio")

    def test_transfer(self):
        '''Check the copying works and it is as per the template '''
        dataset_location="/media/seeni/Backup/Dataset/Extract/LibriSpeech/dev-clean/3081-largeFile/166546"
        init.transfer_arrange(dataset_location)
        assert os.path.exists("Data/Text/3081-166546.trans.txt")
        assert os.path.exists("Data/HumanAudioFlac/3081-166546-0000.flac") #first file
        assert os.path.exists("Data/HumanAudioFlac/3081-166546-0089.flac") #second file

    def test_convert_wav_1(self):
        '''Convert all the files in HumanAudioFlac/*.flac into HumanAudio/*.wav '''
        assert os.path.exists("Data/HumanAudioFlac/3081-166546-0000.flac")
        mediaconvert.convert_1_flac_wav("Data/HumanAudioFlac/3081-166546-0000.flac","Data/HumanAudio/3081-166546-0000.wav")
        assert os.path.exists("Data/HumanAudio/3081-166546-0000.wav")

    @unittest.skip("Not Implemented")
    def test_convert_wav_all(self):
        '''Convert all the files in HumanAudioFlac/*.flac into HumanAudio/*.wav '''
        assert os.path.exists("Data/HumanAudioFlac/3081-166546-0000.flac")
        assert os.path.exists("Data/HumanAudioFlac/3081-166546-0089.flac")
        mediaconvert.convert_all("Data/HumanAudioFlac")
        assert os.path.exists("Data/HumanAudio/3081-166546-0000.wav")
        assert os.path.exists("Data/HumanAudio/3081-166546-0089.wav")

    @unittest.skip("Not Implemented")
    def test_make_tts(self):
        '''test the generation of tts wav files'''
        assert os.path.exists("Data/Text/3081-166546.trans.txt")
        tts.make_tts("Data/Text/3081-166546.trans.txt")
        assert os.path.exists("Data/TTS/3081-166546-0000.wav")
        assert os.path.exists("Data/TTS/3081-166546-0089.wav")

    @unittest.skip("Not Implemented")
    def test_spectrogram_tts(self):
        '''Check whethet spectrogram is created for all files in Data/TTS'''
        assert os.path.exists("Data/TTS/3081-166546-0000.wav")
        assert os.path.exists("Data/TTS/3081-166546-0089.wav")
        spectrogram.spectrogram_tts()
        assert os.path.exists("Data/Spectrogram/TTS/3081-166546-0000.wav.jpg")
        assert os.path.exists("Data/Spectrogram/TTS/3081-166546-0089.wav.jpg")

    @unittest.skip("Not Implemented")
    def test_spectrogram_humanaudio(self):
        '''Check whethet spectrogram is created for all files in Data/HumanAudio'''
        assert os.path.exists("Data/HumanAudio/3081-166546-0000.wav")
        assert os.path.exists("Data/HumanAudio/3081-166546-0089.wav")
        spectrogram.spectrogram_human_audio()
        assert os.path.exists("Data/Spectrogram/HumanAudio/3081-166546-0000.wav.jpg")
        assert os.path.exists("Data/Spectrogram/HumanAudio/3081-166546-0089.wav.jpg")


if __name__ == '__main__':
    unittest.main()
