from tensorflow import keras

# load data
data = keras.datasets.mnist

(train_images, train_labels), (test_images, test_labels) = data.load_data()

# preprocess data
train_images = train_images.reshape((60000, 28, 28, 1))
train_images = train_images.astype('float32') / 255

test_images = test_images.reshape((10000, 28, 28, 1))
test_images = test_images.astype('float32') / 255

train_labels = keras.utils.to_categorical(train_labels)
test_labels = keras.utils.to_categorical(test_labels)

# model
model = keras.models.Sequential([
	keras.layers.Conv2D(32, (3, 3), input_shape=(28, 28, 1)),
	keras.layers.MaxPooling2D((2, 2)),
	keras.layers.Conv2D(64, (3, 3), activation='relu'),
	keras.layers.MaxPooling2D((2, 2)),
	keras.layers.Conv2D(64, (3, 3), activation='relu')
	])

# classifier
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(64, activation='relu'))
model.add(keras.layers.Dense(10, activation='softmax'))

model.compile(optimizer='rmsprop',loss='categorical_crossentropy',metrics=['accuracy'])

model.fit(train_images, train_labels, epochs=5, batch_size=64)

test_loss, test_acc = model.evaluate(test_images, test_labels)

print('Test Loss: ', test_loss)
print('Test Accuracy: ', test_acc)

