# -*- coding: utf-8 -*-
"""
Created on Fri Aug  5 11:13:30 2022

@author: Firas
"""

import tensorflow as tf

from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
import cv2
from imutils import paths
import os
import random
import numpy as np
from sklearn.preprocessing import LabelBinarizer
from tensorflow.keras.utils import to_categorical

from sklearn.preprocessing import LabelEncoder

def folder_to_f_l(folder):
    
    imagePaths = list(paths.list_images(folder))
    
    data = []
    labels = []
    
    random.shuffle(imagePaths)

    for imagePath in imagePaths[:1000]:
        label = imagePath.split(os.path.sep)[-2]
        image = cv2.imread(imagePath)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) 
        image = cv2.resize(image, (32, 32), interpolation = cv2.INTER_AREA)

        labels.append(label)
        data.append(image)
    
    
    labels = np.array(labels)    
    data = np.array(data)    
    le = LabelEncoder()
    labels = le.fit_transform(labels)
    
    return data, labels
    
train_images, train_labels = folder_to_f_l(r'Vegetable Images\train')

test_images, test_labels = folder_to_f_l(r'Vegetable Images\test')


print(train_images.shape)
print(train_labels.shape)

train_images, test_images = train_images / 255.0, test_images / 255.0

plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i])
    plt.xlabel(train_labels[i])
plt.show()


model = models.Sequential()
model.add(layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=(32, 32, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(units=64, activation='relu'))
model.add(layers.Dense(units=15))

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

history = model.fit(train_images, train_labels, epochs=60, 
                    validation_data=(test_images, test_labels))

plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
#plt.ylim([0.5, 1])
plt.legend(loc='lower right')

test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)


model.summary()