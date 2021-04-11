import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

data = keras.datasets.reuters
datalen = 10000
categories = 46

(train_data, train_labels), (test_data, test_labels) = data.load_data(num_words=datalen)

word_index = data.get_word_index()

reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

decoded_newswire = ' '.join([reverse_word_index.get(i - 3, '?') for i in train_data[0]])

def vectorize_sequences(sequences, dimention=datalen):
	results = np.zeros((len(sequences), dimention))
	for i, sequence in enumerate(sequences):
		results[i, sequence] = 1.
	return results

x_train = vectorize_sequences(train_data)
x_test = vectorize_sequences(test_data)

def to_one_hot(labels, dimention=categories):
	results = np.zeros((len(labels), dimention))
	for i, label in enumerate(labels):
		results[i, label] = 1.
	return results

one_hot_train_labels = to_one_hot(train_labels)
one_hot_test_labels = to_one_hot(test_labels)

model = keras.Sequential([
	keras.layers.Dense(64, activation='relu', input_shape=(10000, )),
	keras.layers.Dense(64, activation='relu'),
	keras.layers.Dense(46, activation='softmax')
	])

model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

x_val = x_train[:1000]
partial_x_train = x_train[1000:]

y_val = one_hot_train_labels[:1000]
partial_y_train = one_hot_train_labels[1000:]

history = model.fit(partial_x_train, partial_y_train, epochs=9, batch_size=512, validation_data=(x_val, y_val))

results = model.evaluate(x_test, one_hot_test_labels)

predictions = model.predict(x_test)

np.argmax(predictions[0])

