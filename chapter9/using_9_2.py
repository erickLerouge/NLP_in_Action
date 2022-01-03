from gensim.models.keyedvectors import KeyedVectors
import numpy as np
from nltk.tokenize import TreebankWordTokenizer

from keras.models import model_from_json

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



maxlen = 400
embedding_dims = 300

with open('lstm_model2.json','r') as json_file:
	json_string = json_file.read()

model = model_from_json(json_string)
model.load_weights('lstm_weights1.h5')

sample_1 = """ I hate that the dismal weather had me down for so long, when will it break! Ugh, when does happiness return
the sun is blinging and the puffy clouds are to thin. I can't wait for the weekend"""
# you pass a dummy value in the first element of the tuple, because your helper expects it from the way yu processed the initial data. 
vec_list = tokenize_and_vectorize([(1, sample_1)])
test_vec_list = pad_trunc(vec_list, maxlen)
test_vec = np.reshape(test_vec_list, (len(test_vec_list),maxlen,embedding_dims))
predict_x = model.predict(test_vec)
class_p = np.argmax(predict_x, axis = 1)
print('Sample sentiment, 1 - pos 2 - neg : {}'.format(class_p))
