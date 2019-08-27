# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout,Activation,Conv2D,MaxPooling2D,Flatten
import cv2
import matplotlib.pyplot as plt

import glob

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


datadir="/kaggle/input/dogs-vs-cats/dataset/dataset/training_set"
category=["dogs","cats"]
for c in category:
    path1=os.path.join(datadir,c)
    for img in os.listdir(path1):
        
        img_array=cv2.imread(os.path.join(path1,img),cv2.IMREAD_GRAYSCALE)
        plt.imshow(img_array,cmap="gray")
        plt.show()


new_array=cv2.resize(img_array,(50,50))
plt.imshow(new_array,cmap="gray")
plt.show()

train=[]
def training():
    for c in category:
        path2=os.path.join(datadir,c)
        class_num=category.index(c)
        for i in os.listdir(path2):
            try:
                im_arr=cv2.imread(os.path.join(path2,i),cv2.IMREAD_GRAYSCALE)
                new_arr=cv2.resize(im_arr,(50,50))
                train.append([new_arr,class_num])
            except Exception as e:
                pass
training()

import random
random.shuffle(train)

X=[]
Y=[]
for features ,label in train:
    X.append(features)
    Y.append(label)
X=np.array(X).reshape(-1,50,50,1)

X=X/255.0
model=Sequential()

model.add(Conv2D(64,(3,3),input_shape=X.shape[1:]))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(64,(3,3)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())
model.add(Dense(64))

model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss="binary_crossentropy",optimizer="adam",metrics=['accuracy'])
model.fit(X,Y,batch_size=32,epochs=3,validation_split=0.1,)

