from keras.models import model_from_json
import my_preprocessing_functions as my_p
import numpy as np

with open('cnn_model.json', 'r') as json_file:
	json_string = json_file.read()

model = model_from_json(json_string)
model.load_weights('cnn_weights.h5')

maxlen = 400
embedding_dims = 300
sample1 = 'I hate that the dismal weather had me down for so long, when will it break!'
vec_list = my_p.tokenize_and_vectorize([(1, sample1)])
test_vec_list = my_p.pad_trunc(vec_list,maxlen) 

test_vec = np.reshape(test_vec_list,(len(test_vec_list),maxlen,embedding_dims))
print(model.predict(test_vec))
# the predict_classes have been removed since version 2.6
#print(model.predict_classes(test_vec))
print(np.round(model.predict(test_vec)).astype(int))
