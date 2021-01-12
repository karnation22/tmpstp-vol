## input audio file
## (maybe) input text
## output words with timestamps..
from pydub import AudioSegment
import argparse

def main():
    parser = argparse.ArgumentParser(description="input .wav file")
    parser.add_argument('wav-file',help="Please input a wav file")
    args = vars(parser.parse_args())
    if(args['wav-file']==None or not args['wav-file'].endswith(".wav")):
        return
    filename = args['wav-file']
    sound = AudioSegment.from_wav(args['wav-file'])
    sound = sound.set_channels(1)
    sound = sound.set_frame_rate(16000)
    sound.export("{}_mono.wav".format(filename[:filename.index(".wav")]), format="wav")

main()