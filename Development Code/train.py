# Max Wenzel

from seqlearn.datasets import load_conll
from seqlearn.perceptron import StructuredPerceptron
from seqlearn.evaluation import bio_f_score
from seqlearn.hmm import MultinomialHMM
import numpy as np
import pandas as pd
import sklearn as sk
import nltk as nl
import os
import re
import sys

def load_dat(filename="gene-trainF18.txt", tra=True): # This function allows the sentences in a file
# to be treated as a single string making POS tagging easier 
	data = []
	with open(filename) as f:
		temp = []
		for line in f:
			if line.strip() == "": # Start new sentence
				data.append(temp)
				temp = []
			else: # build up sentence
				l = line.split()
				if tra:
					temp.append([l[1], l[2]])
				else:
					temp.append(l[1])
	data.append(temp) # add that last one in there 
	return data

def load_con(data, features, tts=False): # loads the training data and adds in POS tags can do test split too  
	np.random.shuffle(data)
	
	if tts: # Code for doing train-test split
		split = int(len(data) * tts)
	
		train = data[:split]
		test = data[split:]


		if os.path.exists("test.txt"): # replace files instead of adding on 
			os.remove("test.txt")
		if os.path.exists("gs.txt"):
			os.remove("gs.txt")


		with open("test.txt", "a") as f: # Create teh fhe file for testing
			for s in test:
				pos = nl.pos_tag([w[0] for w in s])
				for ii in range(len(s)):
					f.write("{}\t{}\t{}\n".format(s[ii][0],pos[ii][1],s[ii][1]))
				f.write("\n")

		with open("gs.txt", "a") as f: #create the gold standard file for comparison
			for s in test:
				for ii in range(len(s)):
					f.write("{}\t{}\n".format(s[ii][0],s[ii][1]))
				f.write("\n")
		X_test, y_test, l_test = load_conll("test.txt", features)
	else:
		train = data # if not doing the tts then just use all the data to train
	
	if os.path.exists("train.txt"): # replace the training fil 
		os.remove("train.txt")

	with open("train.txt", "a") as f: # Create the train file and add in the POS
		for s in train:
			pos = nl.pos_tag([w[0] for w in s])
			for ii in range(len(s)):
				f.write("{}\t{}\t{}\n".format(s[ii][0],pos[ii][1],s[ii][1]))
			f.write("\n")

	X_train, y_train, l_train = load_conll("train.txt", features) # use the load_conll function on the generated file
	
	if tts: # variable return based on if you want to have a test-train split
		return X_train, X_test, y_train, y_test, l_train, l_test
	else:
		return X_train, y_train, l_train


def features(seq, i): # The Feature Extractor! aka 90% of this project
	p = seq[i].split()[0]
	pos = seq[i].split()[1]
	yield "word=" + p # Current word and POS
	yield "pos=" + pos
	if i == 0: # add info for the previous word
		yield "preword=" + "START"
		yield "prepos=" + "START"
	else:
		pp = seq[i-1].split()[0]
		ppos = seq[i-1].split()[1]
		yield "prepos=" + ppos
		if pp.isupper() and len(pp) == 3: # check if previous word is acronym 
			yield "preUpper"
		if re.search(r"\d", pp.lower()): # check if prev word has number
			yield "preNumber"
		yield "preword=" + pp.lower()
	if (i + 1) == len(seq):
		yield "folword=" + "END"
		yield "folpos=" + "END"
	else: # check the same for the next word
		nnp = seq[i+1].split()[0]
		nnpos = seq[i+1].split()[1]
		yield "folpos=" + nnpos
		if nnp.isupper():
			yield "folUpper"
		if re.search(r"\d", nnp.lower()):
			yield "folNumber"
		yield "folword=" + nnp.lower()
	if p.isupper() and len(p) == 3:
		yield "Uppercase"
	if re.search(r"\d", p.lower()):
		yield "Number"
	if len(p) > 8: # check if current word is unusually long 
		yield "Long"

def create_eval_file(y_p):  # File to produce the output from the predicted y 
	ind = 0 # counters for keeping track of words and sentences
	c = 1

	if os.path.exists("output.txt"):
		os.remove("output.txt")
	with open("output.txt", "a") as f: 
		with open("test.txt", "r") as t: # use info from the test file to generate output
			for line in t:
				if line.strip() == "": # similar proc as above file 
					f.write("\n")
					c = 1
				else:
					l = line.split()
					f.write("{}\t{}\t{}\n".format(c,l[0], y_p[ind]))
					c += 1
					ind += 1

def addcol(data, filename): # yet ANOTHER function for file writing 
# I probably could have made this a bit less bloaty 
# this function both adds in a dummy BOI column and adds POS that is used for final pipeline
	d = load_dat(filename, tra=False)
	sent_num = 0
	if os.path.exists("test.txt"):
		os.remove("test.txt")
	with open("test.txt", "a") as f:
		ind = 0
		pos = nl.pos_tag(d[0])
		for line in data:
			if line.strip() == "":
				f.write("\n")
				ind = 0
				sent_num += 1
				if(sent_num != len(d)): # generate the new POS for each new sentence
					pos = nl.pos_tag(d[sent_num])
			else: # write the test file 
				f.write(line.strip().split()[1] + "\t" + pos[ind][1] + "\tO\n")
				ind += 1



def main():
	print("Loading data") #Useful messages 
	dat = open(sys.argv[1]) # get filename and open the correct file 
	addcol(dat, sys.argv[1])
	X_test, y_test, l_test = load_conll("test.txt", features) # load the test set created by addcol
	data = load_dat() # yet another file loading function!
	X_train, y_train, l_train = load_con(data, features) # the big loading file 
	per = StructuredPerceptron(lr_exponent=0.35,max_iter=20,verbose=1) # Some trial and error found that
	# a lr of .35 and 20 iters worked best
	print("Fitting")
	per.fit(X_train, y_train, l_train)  # fit and predict
	y_p = per.predict(X_test, l_test)
	create_eval_file(y_p) # save

	print("Done!")



if __name__ == '__main__':
	main()