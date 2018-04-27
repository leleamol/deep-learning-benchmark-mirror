'''Train a simple deep CNN on the CIFAR10 small images dataset.

It gets to 75% validation accuracy in 25 epochs, and 79% after 50 epochs.
(it's still underfitting at that point, though).
'''

from __future__ import print_function
import keras
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
import os
import time
import logging, pdb
logging.basicConfig(level=logging.INFO)
# logging.basicConfig(filename='example.log', filemode='w',
#                     format='%(message)s', level=logging.INFO)


class TimeHistory(keras.callbacks.Callback):
    def __init__(self):
        super(TimeHistory, self).__init__()
        self.times = []

    def on_train_begin(self, logs={}):
        self.times = []

    def on_epoch_begin(self, batch, logs={}):
        self.epoch_time_start = time.time()

    def on_epoch_end(self, batch, logs={}):
        self.times.append(time.time() - self.epoch_time_start)

    def reformat(self, var):
        if var >= 1:
            var = '%.2f ' % var
            time_format = 'sec'
        elif var >= 1e-3:
            var = '%.2f ' % (var * 1e3)
            time_format = 'msec'
        else:
            var = '%.2f ' % (var * 1e6)
            time_format = 'usec'
        return var, time_format

    def get_speed(self, epoch=-1):
        if epoch >= self.params['samples']:
            raise ValueError("Epoch out of limit Error")
        if len(self.times) != 0:
            speed = self.times[epoch] / self.params['samples']
            speed, time_format = self.reformat(speed)
            return speed + time_format + '/step'
        else:
            raise ValueError()

    def get_total_time(self):
        total_time = sum(self.times)
        total_time, time_format = self.reformat(total_time)
        return total_time + time_format

    def get_epoch_time(self, epoch=-1):
        if epoch >= self.params['samples']:
            raise ValueError("Epoch out of limit Error")
        time, time_format = self.reformat(self.times[epoch])
        return time + time_format

    def get_sample_per_sec(self, epoch=-1):
        if epoch >= self.params['samples']:
            raise ValueError("Epoch out of limit Error")
        sample_sec = self.params['samples'] / self.times[epoch]
        sample_sec, time_format = self.reformat(sample_sec)
        return sample_sec + 'samples/' + time_format


def logging_method(history_callback, time_callback):
    if not isinstance(history_callback, keras.callbacks.History):
        raise TypeError('`history_callback` should be an instance of '
                        '`keras.callbacks.History`')
    if not isinstance(time_callback, keras.callbacks.Callback):
        raise TypeError('`time_callback` should be an instance of '
                        '`keras.callbacks.Callback`')
    for e in history_callback.epoch:
        logging.info('[Epoch %d] - time: %s - time_step: %s - speed: %s - '
                     'train_loss: %.4f - train_acc: %.4f - val_loss: %.4f - '
                     'val_acc: %.4f'
                     %(e+1, time_callback.get_epoch_time(e), time_callback.get_speed(e), time_callback.get_sample_per_sec(e),
                       history_callback.history['loss'][e],
                       history_callback.history['acc'][e],
                       history_callback.history['val_loss'][e],
                       history_callback.history['val_acc'][e]))



batch_size = 32
num_classes = 10
epochs = 200
data_augmentation = False
num_predictions = 20
save_dir = os.path.join(os.getcwd(), 'saved_models')
model_name = 'keras_cifar10_trained_model.h5'

# The data, split between train and test sets:
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train = x_train[:1000]
y_train = y_train[:1000]
x_test = x_test[:1000]
y_test = y_test[:1000]
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# Convert class vectors to binary class matrices.
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

model = Sequential()
model.add(Conv2D(32, (3, 3), padding='same',
                 input_shape=x_train.shape[1:]))
model.add(Activation('relu'))
model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes))
model.add(Activation('softmax'))

# initiate RMSprop optimizer
opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)

# Let's train the model using RMSprop
model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
pdb.set_trace()
if not data_augmentation:
    print('Not using data augmentation.')
    time_callback = TimeHistory()
    history_callback = model.fit(x_train, y_train,
                                 batch_size=batch_size,
                                 epochs=epochs,
                                 validation_data=(x_test, y_test),
                                 shuffle=True, verbose=0,
                                 callbacks=[time_callback])
    speed = time_callback.get_speed()
    total_training_time = time_callback.get_total_time()
    logging_method(history_callback, time_callback)
    print(history_callback)
else:
    print('Using real-time data augmentation.')
    # This will do preprocessing and realtime data augmentation:
    datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=True,  # randomly flip images
        vertical_flip=False)  # randomly flip images

    # Compute quantities required for feature-wise normalization
    # (std, mean, and principal components if ZCA whitening is applied).
    datagen.fit(x_train)

    # Fit the model on the batches generated by datagen.flow().
    model.fit_generator(datagen.flow(x_train, y_train,
                                     batch_size=batch_size),
                        epochs=epochs,
                        validation_data=(x_test, y_test),
                        workers=4)

# Save model and weights
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
model_path = os.path.join(save_dir, model_name)
model.save(model_path)
print('Saved trained model at %s ' % model_path)

# Score trained model.
scores = model.evaluate(x_test, y_test, verbose=1)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])

