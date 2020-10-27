import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.optimizers import RMSprop
import numpy as np

boadsize = 10
batch_size = 128
num_classes = boadsize*boadsize
epochs = 20


# the data, split between train and test sets
#(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = np.load('savedata\\bord_histry.npy')
y_train = np.load('savedata\\input_history.npy')

x_test = np.load('savedata\\bord_histryt.npy')
y_test = np.load('savedata\\input_historyt.npy')

x_train = x_train.reshape(int(x_train.size/(boadsize*boadsize)), (boadsize*boadsize))
x_test = x_test.reshape(int(x_test.size/(boadsize*boadsize)), (boadsize*boadsize))

x_train += 1
x_test += 1

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 2
x_test /= 2

print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices

y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)


model = Sequential()

model.add(Dense(512, activation='relu', input_shape=(100,)))
model.add(Dropout(0.2))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.2))

model.add(Dense(num_classes, activation='softmax'))

model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer=RMSprop(),
              metrics=['accuracy'])

history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_data=(x_test, y_test))
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
model.save('model.h5', include_optimizer=False)
