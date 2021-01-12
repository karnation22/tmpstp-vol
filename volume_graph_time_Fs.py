import librosa
from matplotlib import pyplot as plt
import argparse
import requests
import numpy as np
import os
def main():
    parser = argparse.ArgumentParser(description="Please input filename..")
    parser.add_argument("filename",help=".wav or .mp3 filename (local only, same directory)...")
    args = vars(parser.parse_args())
    if(not(args['filename'].endswith(".wav") or args['filename'].endswith(".mp3"))
        or args['filename'] not in os.listdir()):
        print("Please input a valid .wav or .mp3 file from local directory")
        return 
    fileNameRaw = args['filename']
    x, Fs = librosa.load(fileNameRaw, sr=None)
    time = librosa.get_duration(filename=fileNameRaw)
    plt.plot(x,color="gray")
    plt.xlim([0, x.shape[0]])
    plt.xlabel("Time (samples)")
    plt.ylabel("Amplitude")
    plt.title("Amplitude vs Time")
    if(fileNameRaw.endswith(".wav")): fileNameRaw2 = fileNameRaw[:fileNameRaw.index(".wav")]
    else: fileNameRaw2 = fileNameRaw[:fileNameRaw.index(".mp3")]
    plt.savefig("{}_vol_plot.png".format(fileNameRaw2))
    with(open("{}_time_Fs.txt".format(fileNameRaw2), 'w')) as f_tim_Fs:
        f_tim_Fs.writelines(["Time: {}\n".format(time), "Fs: {}".format(Fs)])
    np.savetxt("{}_amp_vals.txt".format(fileNameRaw2),x)
main()