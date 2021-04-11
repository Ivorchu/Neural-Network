from tensorflow import keras

data = [0,1,1,1,1,1,1,2,2,2,2,3,3] * 5
data = keras.utils.to_categorical(data)

data_gen = keras.preprocessing.sequence.TimeseriesGenerator(data, data, length=1, batch_size=1)

model = keras.Sequential()
model.add(keras.layers.SimpleRNN(10, stateful=True, batch_input_shape=(1, None, 4)))
model.add(keras.layers.Dense(4, activation='softmax'))
model.compile(optimizer='rmsprop', loss='categorical_crossentropy',metrics=['acc'])

model.reset_states()
out_put = data_gen[0][0]

for i in range(50):
	prediction = model.predict_classes(out_put, batch_size=1)
	print(prediction)
	out_put = keras.utils.to_categorical(prediction, num_classes=4).reshape(1, -1, 4)

