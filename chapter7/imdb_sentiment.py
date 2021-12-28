import numpy as np
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import Conv1D, GlobalMaxPooling1D

# I separate the model instruction from the preprocessing tasks

import my_preprocessing_functions as my_p

dataset = my_p.pre_process_data('/Users/velazquezerick/Documents/NLP_in_action/chapter7/aclImdb/train/')

vectorized_data = my_p.tokenize_and_vectorize(dataset)
expected = my_p.collect_expected(dataset)


#print(dataset[0])
#splitting the dataset into train and test sets

split_point = int(len(vectorized_data)*0.8)
x_train = vectorized_data[:split_point]
y_train = expected[:split_point]
x_test = vectorized_data[split_point:]
y_test = expected[split_point:]

### hyperparameters:

maxlen = 400
batch_size = 32
embedding_dims = 300
filters = 250
kernel_size = 3
hidden_dim = 250
epochs = 2

### padding the dataset:

x_train = my_p.pad_trunc(x_train, maxlen)
x_test = my_p.pad_trunc(x_test, maxlen)
### need to do the test too?

x_train = np.reshape(x_train, (len(x_train), maxlen, embedding_dims))
print('maxlen {} embedding_dims {}'.format(maxlen,embedding_dims))
y_train = np.array(y_train)
print('len x_train', len(x_train))
print('len x_test ',len(x_test))
print('types: x_train {} , x_test'.format(type(x_train),type(x_test)))
x_test = np.reshape(x_test,(len(x_test), maxlen, embedding_dims))
y_test = np.array(y_test)

print('type of training and testing sets:')
print('x_train {} /n x_test {} \n y_train {} \n y_test {}'.format(type(x_train),type(x_test),type(y_train),type(y_test)))
print('Building model')
model = Sequential()

model.add(Conv1D(filters, kernel_size,padding='valid',activation='relu',strides=1, input_shape=(maxlen, embedding_dims)))
model.add(GlobalMaxPooling1D()) # pooling is used to dimensionality reduction
model.add(Dense(hidden_dim))
model.add(Dropout(0.2))
model.add(Activation('relu'))
model.add(Dense(1))
model.add(Activation('sigmoid'))
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
print(model.summary())
#lets train the hell out of this!
model.fit(x_train,y_train, batch_size=batch_size, epochs=epochs, validation_data=(x_test,y_test))
#saving the model
model_structure = model.to_json()
with open('cnn_model.json','w') as json_file:
	json_file.write(model_structure)
model.save_weights("cnn_weights.h5")


print('finish')
