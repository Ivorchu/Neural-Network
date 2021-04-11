from tensorflow import keras

dinner = [0, 1, 1, 2, 0, 1, 1, 2]
dinner = keras.utils.to_categorical(dinner)

data_gen = keras.preprocessing.sequence.TimeseriesGenerator(dinner, dinner, 
	length=2, sampling_rate=1, stride=1, batch_size=2)

model = keras.Sequential()
model.add(keras.layers.SimpleRNN(10, input_shape=(2, 3)))
model.add(keras.layers.Dense(3, activation='softmax'))

model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['acc'])

train_history = model.fit(data_gen, epochs=50)

prediction = model.predict_classes(data_gen)
print(prediction)

