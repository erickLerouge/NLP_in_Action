from keras.models import Sequential
from keras.layers import SimpleRNN, Dense, Dropout, Flatten
from keras.layers.wrappers import Bidirectional

import glob
import os
from random import shuffle
from nltk.tokenize import TreebankWordTokenizer
from nlpia.loaders import get_data
from gensim.models.keyedvectors import KeyedVectors
import numpy as np

word_vectors = KeyedVectors.load_word2vec_format('/Users/velazquezerick/Documents/NLP_in_action/GoogleNews-vectors-negative300.bin',binary=True)
####### preprocessing functions
def pre_process_data(filepath):
	positive_path = os.path.join(filepath,'pos')
	negative_path = os.path.join(filepath,'neg')
	pos_label = 1
	neg_label = 0
	dataset = []
	for filename in glob.glob(os.path.join(positive_path,'*.txt')):
		with open(filename,'r') as f:
			dataset.append((pos_label, f.read())) 
	for filename in glob.glob(os.path.join(negative_path,'*.txt')):
		with open(filename,'r') as f:
			dataset.append((neg_label, f.read()))
	shuffle(dataset)
	return dataset
 
def tokenize_and_vectorize(dataset):
	tokenizer = TreebankWordTokenizer()
	vectorized_data = []
	for sample in dataset:
		tokens = tokenizer.tokenize(sample[1])
		sample_vecs = []
		for token in tokens:
			# check if the token is in our word embeddings
			try:
				sample_vecs.append(word_vectors[token])
			except KeyError:
				pass
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


dataset = pre_process_data('../chapter7/aclImdb/train')
vectorized_data = tokenize_and_vectorize(dataset)
expected = collect_expected(dataset)
split_point = int(len(vectorized_data) * 0.8)
x_train = vectorized_data[:split_point]
y_train = expected[:split_point]

x_test = vectorized_data[split_point:]
y_test = expected[split_point:]

num_neurons = 5
maxlen = 100
embedding_dims = 300
batch_size = 32
epochs = 8

x_train = pad_trunc(x_train, maxlen)
x_test = pad_trunc(x_test, maxlen)

x_train = np.reshape(x_train,(len(x_train),maxlen, embedding_dims))
y_train = np.array(y_train)
x_test = np.reshape(x_test,(len(x_test), maxlen, embedding_dims))
y_test =  np.array(y_test)


model = Sequential()
model.add(Bidirectional(SimpleRNN(num_neurons, return_sequences =True),input_shape=(maxlen, embedding_dims)))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(1,activation='sigmoid'))

model.compile('rmsprop','binary_crossentropy',metrics=['accuracy'])
model.fit(x_train,y_train, batch_size = batch_size, epochs = epochs, validation_data = (x_test,y_test))

print(model.summary())
