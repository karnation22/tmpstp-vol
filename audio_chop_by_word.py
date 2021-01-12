from pydub import AudioSegment
import os
import argparse
def main():
    parser = argparse.ArgumentParser(description="Parse audio files into multiple snippets")
    parser.add_argument("audio-file",help="relevant .mp3 or .wav file")
    parser.add_argument("text-dump", help="text dump of words with start and end times")
    args = vars(parser.parse_args())
    isWave = False
    if(not(args['audio-file'].endswith(".mp3") or args['audio-file'].endswith(".wav")) 
            or args['audio-file'] not in os.listdir() or args['text-dump'] not in os.listdir()):
        print("please input valid .mp3 or .wav file and text dump file in current working directory")
        return
    if(args['audio-file'].endswith(".wav")):
        isWave = True
        newAudio = AudioSegment.from_wav(args['audio-file'])
    else:
        assert(args['audio-file'].endswith(".mp3"))
        newAudio = AudioSegment.from_mp3(args['audio-file'])  
    with open(args['text-dump'], 'r') as f_text_dump:
        text_dump_data = f_text_dump.readlines()
    counter = 0
    aud_files_bn_str = ""
    for item in text_dump_data:
        if(item.startswith("<")): continue
        else:
            try:
                itemSplit = item.strip().split()
                word, startTime, endTime = itemSplit[0], float(itemSplit[1]), float(itemSplit[2])
                if("(" in word): word = word[:word.index("(")]
                t_start, t_end = int(startTime*1000), int(endTime*1000)
                counter += 1
                curAudio = newAudio[t_start:t_end]
                if(isWave): 
                    relStr = "{}_{}.wav".format(counter, word)
                    curAudio.export(relStr, format="wav") 
                else: 
                    relStr = "{}_{}.mp3".format(counter, word)
                    curAudio.export(relStr, format="mp3")
                aud_files_bn_str += relStr+"\n"
            except: continue
    with open("audio_splits_text_data.txt","w") as f_aud_txt:
        f_aud_txt.write(aud_files_bn_str)

main()

