import numpy as np
from keras.models import Sequential # this is the base keras model
from keras.layers import Dense, Activation # Dense is a fully connected layer of neurons
from tensorflow.keras.optimizers import SGD
#from keras.optimizers import SGD # stands for Stochastic gradient descent. Other more exist
import h5py

x_train = np.array([[0,0],[0,1],[1,0],[1,1]])
y_train = np.array( [0,1,1,0])

model = Sequential()
num_neurons = 10
model.add(Dense(num_neurons,input_dim = 2))
model.add(Activation('tanh'))#this is the activation function
model.add(Dense(1)) # this represents the ouput layer with one neuron to output a single binary classification value 
model.add(Activation('sigmoid'))
print(model.summary())

sgd = SGD(lr=0.1)

model.compile(loss='binary_crossentropy',optimizer=sgd, metrics = ['accuracy'])

print(model.predict(x_train))

model.fit(x_train,y_train, epochs=100)

#print(model.predict_classes(x_train)) the function predict_classes was removed from the library 
#the equivalent is:
predict_x = model.predict(x_train)
classes = np.argmax(predict_x,axis=1)

print(classes)

print(predict_x)
model_structure = model.to_json()

with open('basic_model.json','w') as json_file:
	json_file.write(model_structure)
model.save_weights('basic_weights.h5')
