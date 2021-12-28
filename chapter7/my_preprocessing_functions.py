from nltk.tokenize import TreebankWordTokenizer
import glob
import os
from random import shuffle
from gensim.models.keyedvectors import KeyedVectors

word_vectors = KeyedVectors.load_word2vec_format('/Users/velazquezerick/Documents/NLP_in_action/GoogleNews-vectors-negative300.bin',binary=True)


def pre_process_data(filepath):
	positive_path = os.path.join(filepath,'pos')
	negative_path = os.path.join(filepath,'neg')
	pos_label = 1
	neg_label = 0
	dataset = []

	for filename in glob.glob(os.path.join(positive_path,'*.txt')):
		with open(filename,'r') as f:
			dataset.append((pos_label,f.read()))
	for filename in glob.glob(os.path.join(negative_path,'*.txt')):
		with open(filename,'r') as f:
			dataset.append((neg_label, f.read()))
	shuffle(dataset)
	return dataset

def tokenize_and_vectorize(dataset):
	tokenizer = TreebankWordTokenizer()
	vectorized_data = []
	expected = [] # this variable seems to have no use 
	for sample in dataset:
		tokens = tokenizer.tokenize(sample[1])
		sample_vecs = []
		for token in tokens:
			try:
				sample_vecs.append(word_vectors[token])
			except KeyError:
				pass # no matching token in the embeddings
		vectorized_data.append(sample_vecs)
	return vectorized_data 

def collect_expected(dataset):
	expected = []
	for sample in dataset:
		expected.append(sample[0])
	return expected

def pad_trunc(data, maxlen):
	new_data = []
	zero_vector = []
	for _ in range(len(data[0][0])):
		zero_vector.append(0.0)
	for sample in data:
		if len(sample) > maxlen:
			temp = sample[:maxlen]
		elif len(sample) < maxlen:
			temp = sample
			additional_elements = maxlen - len(sample)
			for _ in range(additional_elements):
				temp.append(zero_vector)
		else:
			temp = sample
		new_data.append(temp)
	return new_data
