speech_processing:


ifdef wav 
ifdef pip
	pip install argparse numpy pandas librosa soundmeter SpeechRecognition pydub soundfile
else
	@echo 'skipping over the pip install; please ensure you have the relevant dependencies installed..'
endif
	@echo 'found a wav - noted below'
	@echo $(wav)
	@echo $(subst .wav,, $(wav))
	$(eval stem = $(subst .wav,, $(wav)))
	@echo $(stem)
	python wav_preprocessing.py $(wav)
	pocketsphinx_continuous -infile $(stem)_mono.wav -time yes > $(stem)textDump.txt
	python volume_graph_time_Fs.py $(stem)_mono.wav
	python volume_root_mean_square_word.py $(stem)_mono.wav $(stem)textDump.txt $(stem)_mono_time_Fs.txt $(stem)_mono_amp_vals.txt
	python audio_chop_by_word.py $(stem)_mono.wav $(stem)textDump.txt
	python sound_processing.py 	
	python place_data_into_folders.py $(stem)
	python extractTextFromDump.py
	python triples_from_text.py --text_file=$(stem)textString.txt
else
	@echo "no wav file inputted; please type 'wav=<wavfile>' after make, with <wavfile> in cwd; also, please flag you have pip and python 3 installed.."
endif 

