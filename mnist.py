from tensorflow import keras
import matplotlib.pyplot as plt

data = keras.datasets.mnist

(train_images, train_label), (test_images, test_label) = data.load_data()

train_images = train_images.astype('float32') / 255
test_images = test_images.astype('float32') / 255

train_label = keras.utils.to_categorical(train_label)
test_label = keras.utils.to_categorical(test_label)

model = keras.Sequential()
model.add(keras.layers.Flatten(input_shape=(28,28)))
model.add(keras.layers.Dense(128, activation='relu'))
model.add(keras.layers.Dense(10, activation='softmax'))

model.compile(optimizer='rmsprop', loss='categorical_crossentropy',metrics=['acc'])

model.fit(train_images, train_label,epochs=5,batch_size=64)

test_loss, test_acc = model.evaluate(test_images, test_label)

print('Test Loss: ', test_loss)
print('Test Accuracy: ', test_acc)