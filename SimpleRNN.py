from tensorflow import keras
import numpy as np

sample = np.array([[[0],[1]], [[1],[1]],[[1],[2]]])
label = np.array([1, 2, 0])

sample = keras.utils.to_categorical(sample)
label = keras.utils.to_categorical(label)

model = keras.Sequential()
model.add(keras.layers.SimpleRNN(10, input_shape=(2, 3)))
model.add(keras.layers.Dense(3, activation='softmax'))

model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['acc'])

history = model.fit(sample, label, epochs=100)

predict = model.predict_classes(sample)
print(predict)

