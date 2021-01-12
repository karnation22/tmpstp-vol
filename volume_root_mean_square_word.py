## split volume data - find root mean squared volume of each word
import librosa 
import argparse
import numpy as np
import os

def run_output_vol(TimeFsDict, args, rawFileName):
    if(args['volume-data'] not in os.listdir()):
        print("Please have all numeric volume data in current working directory")
        return
    audio_numeric_data = np.loadtxt(args['volume-data'])
    output_list = list()
    time, Fs = TimeFsDict['Time'], TimeFsDict['Fs']
    with open(args['text-dump'],'r') as f_text_dump:
        text_dump_list = f_text_dump.readlines()
    counter = 0
    root_mean_square_max, root_mean_square_min = None,None
    root_mean_square_list = []
    for item in text_dump_list[1:]:
        if(item.startswith("<")): continue
        else: 
            try:
                itemSplit = item.strip().split()
                word, startTime, endTime = itemSplit[0],float(itemSplit[1]), float(itemSplit[2])
                if("(" in word): word = word[:word.index("(")]
                rel_audio_data = list(audio_numeric_data[int(float(startTime/time)*Fs):int(float(endTime/time)*Fs)])
                root_mean_square = np.sqrt(sum([i**2 for i in rel_audio_data])/len(rel_audio_data)) 
                assert(root_mean_square>=0)
                if(root_mean_square_max!=None): root_mean_square_max = max(root_mean_square_max, root_mean_square)
                else: root_mean_square_max = root_mean_square
                if(root_mean_square_min!=None): root_mean_square_min = min(root_mean_square_min, root_mean_square)
                else: root_mean_square_min = root_mean_square
                counter += 1
                output_list.append((counter, word, root_mean_square))
                root_mean_square_list.append(root_mean_square)
            except: continue
    output2_list = []
    root_mean_square_std, root_mean_square_mean = np.std(root_mean_square_list), np.mean(root_mean_square_list)
    for (counter, word, root_mean_square) in output_list:
        assert(root_mean_square>=root_mean_square_min)
        assert(root_mean_square<=root_mean_square_max)
        z = float(root_mean_square-root_mean_square_mean)/root_mean_square_std
        norm = float(1) / (1 + np.exp(-z))
        ## norm = float(root_mean_square-root_mean_square_min)/(root_mean_square_max-root_mean_square_min)
        output2_list.append("{}_{} {}\n".format(counter,word, round(norm,4)))
    with open("{}_RMS_vol_by_word.txt".format(rawFileName),'w') as f_RMS:
        f_RMS.writelines(output2_list)
    with open("rms_vol_min_max.txt","w") as f_min_max:
        f_min_max.write("RMS_MAX: {}; RMS_MIN: {}; RMS_STD: {}; RMS_MEAN: {}"
            .format(root_mean_square_max, root_mean_square_min, root_mean_square_std, root_mean_square_mean))
    return


def main():
    parser = argparse.ArgumentParser(description="Parse the volume data - root mean squared volume by word..")
    parser.add_argument('file-name',help="Input filename")
    parser.add_argument('text-dump',help="Please input pocketsphinx_continuous .txt dump for now")
    parser.add_argument('time-Fs',help="Please input the .txt file with the total time and Fs of audio file")
    parser.add_argument('volume-data',help="Please input relevant volume data")
    args = vars(parser.parse_args())
    rawFileName = args['file-name'][:args['file-name'].index(".wav")]
    if(not(args['file-name'].endswith(".wav") or args['file-name'].endswith("mp3")) 
            or args['file-name'] not in os.listdir()):
        print("Please input valid .wav or .mp3 file in current working directory")
        return
    with open(args['time-Fs'], 'r') as f_time_Fs:
        TimeFsDict = dict()
        try:
            for item in f_time_Fs.readlines():
                itemSplit = item.strip().split()
                lab, val = itemSplit[0], itemSplit[1]
                TimeFsDict[lab[:lab.index(":")]] = float(val)
        except:
            print("Please input time-Fs file in format 'Time: <time>\nFs: <Fs>' ")
            return
    run_output_vol(TimeFsDict, args, rawFileName)



main()