# Copyright 2016    Vincent Renkens
#                   Gaofeng Cheng
##@package kaldiInterface
#contains the functionality to interface with Kaldi

import numpy as np
import gzip
from collections import OrderedDict
import os
from shutil import copyfile
import subprocess
from random import shuffle
## shuffle the utterances and put them in feats_shuffled.scp
#
#@param featdir the directory containing the features in feats.scp
def shuffle_examples(featdir):
	#read feats.scp
	featsfile = open(featdir + '/feats.scp', 'r')
	feats = featsfile.readlines()
	
	#shuffle feats randomly
	shuffle(feats)

	#wite them to feats_shuffled.scp
	feats_shuffledfile = open(featdir + '/feats_shuffled.scp', 'w')
	feats_shuffledfile.writelines(feats)

##read the alignment file generated by kaldi
#
#@param filename path to alignment file
#
#@return a dictionary containing the alignments with the utterance IDs as keys      
def read_alignments(filename):
	print(("alifile_name: "+str(filename)))
	with gzip.open(filename, 'rb') as f:
		alignments = {}
		for line in f:
			data = line.decode().strip().split(' ')
			alignments[data[0]] = np.asarray(list(map(int,data[1:len(data)]))) #segment:alignment
	return alignments

## read a segment file that is used in kaldi
#
#@param filename path to segment file
#
#@return a dictionary containing the utterance IDs teh begining of the utterance and the end of the utterance with the name of the recording as key
def read_segments(filename):
	with open(filename) as f:
		segments = OrderedDict()
		for line in f:
			data = line.decode().strip().split(' ') #seg utt begin end
			if data[1] not in segments:
				segments[data[1]] = [(data[0], float(data[2]), float(data[3]))] #utt: [(seg , begin, end)]
			else:
				segments[data[1]].append((data[0], float(data[2]), float(data[3])))
	return segments

## read the wav.scp file used in kaldi	
#
#@param filename path to the wav scp file
#
#@return a dictionary containing the filenames and bools that determine if the filenames are extended (with a read command) or not with the utterance IDs as keys
def read_wavfiles(filename):
	with open(filename) as f:
		wavfiles = OrderedDict()
		for line in f:
			data = line.decode().strip().split(' ')
			if len(data) == 2: #wav.scp contains filenames
				wavfiles[data[0]] = (data[1], False) #utterance:(filename, not extended)
			else: #wav.scp contains extended filenames
				wavfiles[data[0]] = (line[len(data[0])+1:len(line)-1], True) #utterance: (extended filename, extended)
	return wavfiles
	
#Read the utt2spk file used in kaldi
#
#@param filename path to the utt2spk file
#
#@return a dictionary containing the speaker names with the utterance IDs as keys
def read_utt2spk(filename):
	with open(filename) as f:
		utt2spk = {}
		for line in f:
			data = line.replace('\n','').split(' ')
			utt2spk[data[0]] = data[1]
	return utt2spk
