# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as py
from sklearn.model_selection import train_test_split

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

iris=pd.read_csv("../input/Iris.csv")
iris.describe()

iris.head()

iris.drop('Id',axis=1,inplace=True)
iris.head()

iris.plot(kind='scatter',x='SepalLengthCm',y='SepalWidthCm')
py.xlabel('Length')
py.ylabel('Width')
py.show()

train,test=train_test_split(iris, test_size=0.3)
train_X=train[['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']]
train_Y=train.Species
test_X=test[['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']]
test_Y=test.Species

train_X.head()
test_X.head()

from sklearn.svm import SVC
clf=SVC()
clf.fit(train_X,train_Y)
pred=clf.predict(test_X)

from sklearn.metrics import accuracy_score
acc=accuracy_score(pred,test_Y)
print(acc)
