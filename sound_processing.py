import librosa
import os
import numpy as np
import soundfile as sf
import numpy as np
def main():
    with open("audio_splits_text_data.txt","r") as f_aud_txt:
        list_aud = f_aud_txt.readlines()
        list_aud = [itm.strip() for itm in list_aud]
    ## print(list_aud)
    overall_string = ""
    rms_pitch_max, rms_pitch_min = None,None
    rms_pitch_list = []
    for item in list_aud:
        data, sr = sf.read(item)
        ## print(data, type(data), len(data))
        ## print(sr)
        pitch, mag = librosa.piptrack(y=data, sr=sr)
        pitch1 = pitch.mean(axis=1) #take mean along column axis..
        rms_pitch = np.sqrt(sum([itm**2 for itm in pitch1])/len(pitch1)) ##rms along row..
        rms_pitch_list.append(rms_pitch)
        if(rms_pitch_max!=None): rms_pitch_max = max(rms_pitch_max, rms_pitch)
        else: rms_pitch_max = rms_pitch
        if(rms_pitch_min!=None): rms_pitch_min = min(rms_pitch_min, rms_pitch)
        else: rms_pitch_min = rms_pitch
    rms_pitch_std, rms_pitch_mean = np.std(rms_pitch_list), sum(rms_pitch_list)/len(rms_pitch_list)
    for item,rms_pitch in zip(list_aud, rms_pitch_list):
        item_raw = item[:item.index(".wav")]
        filename = "{}_rms.txt".format(item_raw)
        overall_string += filename+"\n"
        with open(filename,'w') as f_rms:
            z = float(rms_pitch-rms_pitch_mean)/ (rms_pitch_std)
            norm = 1 / (1 + np.exp(-z))
            f_rms.write("{}: {}".format(item_raw, round(norm,5)))

    with open("rms_data.txt","w") as f_rms_data:
        f_rms_data.write(overall_string)
    with open("rms_pitch_max_min_std_mean.txt","w") as f_max_min:
        f_max_min.write("PitchMax: {}; PitchMin:{}; PitchSTD: {}; PitchMean: {}"
            .format(rms_pitch_max,rms_pitch_min, rms_pitch_std,rms_pitch_mean))
main()
