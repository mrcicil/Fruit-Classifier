from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing import image
from keras.layers import Dense, Dropout, Flatten
from keras.models import Sequential,Input,Model
from tensorflow.keras.optimizers import RMSprop
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import to_categorical
from sklearn.metrics import confusion_matrix
from keras.models import load_model
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
import numpy as np
import inspect
import random
import keras
import glob

convert = {"a": 'apple', "b": 'banana', "m":'mixed', "o":'orange'}
train = glob.glob('./fruit/train/*.*')

train_X = []
train_Y = []

for i in train:
    if (i[-1]=='g'):
        image=tf.keras.preprocessing.image.load_img(i, color_mode='rgb', target_size= (300,300))
        image=np.array(image)
        train_X.append(image)
        train_Y.append(i[14:-7])
        
train_X = np.array(train_X)
train_Y = np.array(train_Y)

train_Y[train_Y=='appl']='apple'
train_Y[train_Y=='banan']='banana'
train_Y[train_Y=='mixe']='mixed'
train_Y[train_Y=='orang']='orange'

print(train_X.shape)
print(train_Y.shape)
print(np.unique(train_Y))

test = glob.glob('./fruit/test/*.*')

test_X = []
test_Y = []

for i in test:
    if (i[-1]=='g'):
        image=tf.keras.preprocessing.image.load_img(i, color_mode='rgb', target_size= (300,300))
        image=np.array(image)
        test_X.append(image)
        test_Y.append(i[13:-7])

test_X = np.array(test_X)
test_Y = np.array(test_Y)

test_Y[test_Y=='appl']='apple'
test_Y[test_Y=='banan']='banana'
test_Y[test_Y=='mixe']='mixed'
test_Y[test_Y=='orang']='orange'

print(test_X.shape)
print(test_Y.shape)
print(np.unique(train_Y))


# Random Checking
for i in range(9):
    plt.subplot(3,3,i+1)
    n = random.randint(0,239)
    plt.imshow(train_X[n,:,:])
    plt.title("Fruit: {}".format(train_Y[n]))
plt.tight_layout()

for i in range(9):
    plt.subplot(3,3,i+1)
    n = random.randint(0,59)
    plt.imshow(test_X[n,:,:])
    plt.title("Class {}".format(test_Y[n]))
plt.tight_layout()

# Data Preprocessing
train_X = train_X.astype('float32')
test_X = test_X.astype('float32')
train_X = train_X / 255.
test_X = test_X / 255.

train_Y[train_Y=='apple']=0
train_Y[train_Y=='banana']=1
train_Y[train_Y=='mixed']=2
train_Y[train_Y=='orange']=3

train_Y = train_Y.astype(int)
train_Y_one_hot = to_categorical(train_Y)

n = random.randint(0,239)
print('Original label:', train_Y[n])
print('After conversion to one-hot:', train_Y_one_hot[n])

test_Y[test_Y=='apple']=0
test_Y[test_Y=='banana']=1
test_Y[test_Y=='mixed']=2
test_Y[test_Y=='orange']=3

test_Y = test_Y.astype(int)
test_Y_one_hot = to_categorical(test_Y)

n = random.randint(0,59)
print('Original label:', test_Y[n])
print('After conversion to one-hot:', test_Y_one_hot[n])

train_X, valid_X, train_label, valid_label = train_test_split(train_X, train_Y_one_hot, test_size=0.2, random_state=45)

train_X.shape,valid_X.shape,train_label.shape,valid_label.shape

batch_size = 40
epochs = 10
num_classes = 4

fruit_model = Sequential()
fruit_model.add(Conv2D(32, kernel_size=(3, 3),activation='relu',input_shape=(300,300,3)))
fruit_model.add(Conv2D(32, (3, 3), activation='relu'))
fruit_model.add(MaxPooling2D(pool_size=(2, 2)))
fruit_model.add(Dropout(0.25))
fruit_model.add(Flatten())
fruit_model.add(Dense(128, activation='relu'))
fruit_model.add(LeakyReLU(alpha=0.1))
fruit_model.add(Dropout(0.5))
fruit_model.add(Dense(num_classes, activation='softmax'))

fruit_model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(),metrics=['accuracy'])

fruit_model.summary()

fruit_train = fruit_model.fit(train_X, train_label, batch_size=batch_size,epochs=epochs,verbose=1,validation_data=(valid_X, valid_label))

test_eval = fruit_model.evaluate(test_X, test_Y_one_hot, verbose=0)
print('Test loss:', test_eval[0])
print('Test accuracy:', test_eval[1])

accuracy = fruit_train.history['accuracy']
val_accuracy = fruit_train.history['val_accuracy']
loss = fruit_train.history['loss']
val_loss = fruit_train.history['val_loss']
epochs = range(len(accuracy))
plt.plot(epochs, accuracy, 'bo', label='Training accuracy')
plt.plot(epochs, val_accuracy, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()

# Prediction
predicted_classes = fruit_model.predict(test_X)

predicted_classes = np.argmax(np.round(predicted_classes),axis=1)

predicted_classes.shape, test_Y.shape

print('Predicted: ', predicted_classes)
print('Actual: ', test_Y)

correct = np.where(predicted_classes==test_Y)[0]
incorrect = np.where(predicted_classes!=test_Y)[0]
print('Predict correct:',len(correct))
print('Predict incorrect:',len(incorrect))

confusion_matrix(test_Y, predicted_classes)

fruit_dict = {0:'apple', 1:'banana', 2:'mixed', 3:'orange'}
for i in range(9):
    plt.subplot(3,3,i+1)
    n = random.randint(0,59)
    plt.imshow(test_X[n,:,:])
    plt.title("Predict: {} \n Act: {}".format(fruit_dict[predicted_classes[n]], fruit_dict[test_Y[n]]))
plt.tight_layout()
































