from tensorflow import keras
from dogs_vs_cats_path import *
import matplotlib.pyplot as plt

conv_base = keras.applications.VGG16(weights='imagenet', include_top=False, input_shape=(150,150,3))

model = keras.Sequential()
model.add(conv_base)
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(256, activation='relu'))
model.add(keras.layers.Dense(1, activation='sigmoid'))

print('This is the number of trainable weights before freezing the conv base:', len(model.trainable_weights))
conv_base.trainable = False
print('This is the number of trainable weights after freezing the conv base:', len(model.trainable_weights))

train_datagen = keras.preprocessing.image.ImageDataGenerator(
	rescale=1./255, rotation_range=40, width_shift_range=0.2, height_shift_range=0.2, shear_range=0.2, zoom_range=0.2, horizontal_flip=True, fill_mode='nearest')

test_datagen = keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
	train_dir, target_size=(150,150), batch_size=20, class_mode='binary')

validation_generator = test_datagen.flow_from_directory(
	validation_dir, target_size=(150,150), batch_size=20, class_mode='binary')

conv_base.trainable = True

set_trainable = False
for layer in conv_base.layers:
	if layer.name == 'block5_conv1':
		set_trainable = True
	if set_trainable:
		layer.trainable = True
	else:
		layer.set_trainable = False

model.compile(
	loss='binary_crossentropy', optimizer=keras.optimizers.RMSprop(lr=1e-5), metrics=['acc'])

history = model.fit_generator(
	train_generator, steps_per_epoch=100, epochs=100, validation_data=validation_generator, validation_steps=50)

model.save('freeze_convolutional_base.h5')

def smooth_curve(points, factor=0.8):
	smoothed_points = []
	for point in points:
		if smoothed_points:
			previous = smoothed_points[-1]
			smoothed_points.append(previous * factor + point * (1 - factor))
		else:
			smoothed_points.append(point)
	return smoothed_points

plt.plot(epochs, smooth_curve(acc), 'bo', label='Smoothed training acc')
plt.plot(epochs, smooth_curve(val_acc), 'b', label='Smoothed validation acc')
plt.title('Training and validatoin accuracy')
plt.legend()

plt.figure()

plt.plot(epochs, smooth_curve(loss), 'bo', label='Smoothed training loss')
plt.plot(epochs, smooth_curve(val_loss), 'b', label='Smoothed validation loss')
plt.title('Training and validatoin loss')
plt.legend()

plt.show()

test_generator = test_datagen.flow_from_directory(
	test_dir, target_size=(150,150), batch_size=20, class_mode='binary')

test_loss, test_acc = model.evaluate_generator(test_generator, steps=50)
print('test acc:', test_acc)


