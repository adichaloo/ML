# -*- coding: utf-8 -*-
"""Number_recognition.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1dPo9UuzPm11pMvczdrWkWYqXTS0MaCR6
"""

import tensorflow as tf
import matplotlib.pyplot as plt

mn=tf.keras.datasets.mnist
(X_train,Y_train),(X_test,Y_test)=mn.load_data()
X_train=tf.keras.utils.normalize(X_train,axis=1)
X_test=tf.keras.utils.normalize(X_test,axis=1)
model=tf.keras.models.Sequential([
tf.keras.layers.Flatten(input_shape=(28,28)),
tf.keras.layers.Dense(128,activation=tf.nn.relu),
tf.keras.layers.Dense(128,activation=tf.nn.relu),
tf.keras.layers.Dense(10,activation=tf.nn.softmax)])

model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
model.fit(X_train,Y_train,epochs=3)

loss,acc=model.evaluate(X_test,Y_test)
print(loss,acc)

plt.imshow(X_train[0],cmap=plt.cm.binary)
print(X_train[0])

model.save('epic_num_reader.model')
X_test.shape

new_model = tf.keras.models.load_model('epic_num_reader.model')

predictions=new_model.predict([X_test])
#print(predictions)
print(predictions.shape)
import numpy as np 
for i in range(0,5):
  pred=np.argmax(predictions[i])
  print(pred)

plt.imshow(X_test[0])

