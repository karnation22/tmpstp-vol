import os
import shutil
import argparse
def main():
    parser = argparse.ArgumentParser("parse relevant data files")
    parser.add_argument("stemword")
    args = vars(parser.parse_args())
    stemword = args['stemword']
    with open("audio_splits_text_data.txt","r") as f_audio:
        audio_data = f_audio.readlines()
        audio_data = [audio.strip() for audio in audio_data]
    ## print(audio_data)
    with open("rms_data.txt") as f_rms:
        rms_data = f_rms.readlines()
        rms_data = [rms.strip() for rms in rms_data]
        rms_data.append("{}_mono_RMS_vol_by_word.txt".format(stemword))
    ## print(rms_data)
    other_data = ["{}_mono.wav".format(stemword), "rms_vol_min_max.txt",
        "{}_mono_amp_vals.txt".format(stemword), "{}_mono_time_Fs.txt".format(stemword),
        "{}_mono_vol_plot.png".format(stemword), "{}textDump.txt".format(stemword),
        "audio_splits_text_data.txt", "rms_data.txt", "rms_pitch_max_min_std_mean.txt"]
    if("audio_snippets_by_word" not in os.listdir()):
        os.mkdir("audio_snippets_by_word")
    if("rms_voice_stress_and_vol_by_word" not in os.listdir()):
        os.mkdir("rms_voice_stress_and_vol_by_word") 
    if("{}_misc_data".format(stemword) not in os.listdir()):
        os.mkdir("{}_misc_data".format(stemword))
    for audio in audio_data:
        if(audio in os.listdir()):
            shutil.move(os.path.join(os.getcwd(), audio), os.path.join("audio_snippets_by_word", audio))
    for rms in rms_data:
        if(rms in os.listdir()):
            shutil.move(os.path.join(os.getcwd(), rms), os.path.join("rms_voice_stress_and_vol_by_word", rms))
    for other in other_data:
        if(other in os.listdir()):
            shutil.move(os.path.join(os.getcwd(), other), os.path.join("{}_misc_data".format(stemword), other))

main()