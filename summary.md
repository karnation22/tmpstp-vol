Achievments:

 - Can generate timestamps using 'pocketsphinx_continuous...' command..
 - Can find root-mean-squared volume for each word after segmenting by timestamp..
 - Can segregate audio by word given timestamps in separate audio files.
 - Full Makefile that does the entire process (using 'pocketsphinx_continuous..')
 - Places all relevant output data neatly into different folders.
 - Maps each individual word to voice stress.

Work that needs to be done:

 - Jiachen-1: Train ML model on audio sample of each word using emotion recognition; 
 - Jiachen-2: Extract relative weights of each emotion and map to colour.
 - Karn-1a: Map root-mean-squared volume to diameter of word bubble; might have to apply a log transformation on this...?
 - Karn-1b: Actually integrate color/diameter bubble changes to the knowledge-graph once done.
