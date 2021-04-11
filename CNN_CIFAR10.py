from tensorflow import keras
import matplotlib.pyplot as plt

data = keras.datasets.cifar10

(x_train, y_train), (x_test, y_test) = data.load_data()

x_train_norm = x_train.astype('float32') / 255
x_test_norm = x_test.astype('float32') / 255

y_train_onehot = keras.utils.to_categorical(y_train, 10)
y_test_onehot = keras.utils.to_categorical(y_test, 10)

'''
model = keras.models.Sequential()
model.add(keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(32, 32, 3)))
model.add(keras.layers.Dropout(0.25))
model.add(keras.layers.MaxPooling2D((2, 2)))

model.add(keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(keras.layers.Dropout(0.25))
model.add(keras.layers.MaxPooling2D((2, 2)))

model.add(keras.layers.Flatten())
model.add(keras.layers.Dropout(0.25))
model.add(keras.layers.Dense(1024, activation='relu'))
model.add(keras.layers.Dropout(0.25))
model.add(keras.layers.Dense(10, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
history = model.fit(x=x_train_norm, y=y_train_onehot, batch_size=128, epochs=20, validation_split=0.1)

model.save('CNN_CIFAR10.h5')
'''

old_model = keras.models.load_model('CNN_CIFAR10.h5')

test_loss, test_val = old_model.evaluate(x_test_norm, y_test_onehot)
print('測試資料損失值：', test_loss)
print('測試資料準確率：', test_val)

predict_prop = old_model.predict(x_test_norm)
print('第一筆測試資料的預測機率：', predict_prop[0])

