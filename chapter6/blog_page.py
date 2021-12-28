import numpy as np
from tensorflow.keras.datasets import imdb
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Embedding
from tensorflow.keras.preprocessing import sequence


np.random.seed(7)

top_words = 5000
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words = top_words)
#truncating and padding the input sequences
max_review_length = 500
X_train = sequence.pad_sequences(X_train,maxlen=max_review_length)
X_test = sequence.pad_sequences(X_test, maxlen=max_review_length)

# creating the model
embedding_vector_length = 32

model = Sequential()
model.add(Embedding(top_words, embedding_vector_length, input_length = max_review_length))
model.add(LSTM(100)) # 100 is the number of neuros we want in the structure
model.add(Dense(1,activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())
model.fit(X_train, y_train, validation_data = (X_test,y_test), epochs = 3, batch_size =64)

scores = model.evaluate(X_test, y_test, verbose=0)
print('Accuracy: %.2f%%'%(scores[1]*100))
