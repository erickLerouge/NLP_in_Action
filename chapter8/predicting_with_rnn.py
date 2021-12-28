from nltk.tokenize import TreebankWordTokenizer
from keras.models import model_from_json
import numpy as np
from gensim.models.keyedvectors import KeyedVectors

word_vectors = KeyedVectors.load_word2vec_format('/Users/velazquezerick/Documents/NLP_in_action/GoogleNews-vectors-negative300.bin',binary=True)


maxlen = 400
embedding_dims = 300

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

sample_1 = "I hate that the dismal weather had me down for so long, when will it break! Ugh, when does happines return? The sun is blinding and the puffy clouds are too thin. I can't wait for the weekend"

with open('simpleRNN_model.json','r') as json_file:
	json_string = json_file.read()
model = model_from_json(json_string)
model.load_weights('simpleRNN_weights.h5')

vec_list = tokenize_and_vectorize([(1,sample_1)])
test_vec_list = pad_trunc(vec_list, maxlen)
test_vec = np.reshape(test_vec_list, (len(test_vec_list), maxlen, embedding_dims))
prediction = model.predict(test_vec)
print(np.round(prediction).astype(int))
