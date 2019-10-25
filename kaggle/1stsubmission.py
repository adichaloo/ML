#!/usr/bin/env python
# coding: utf-8

# In[402]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory
'''
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
        '''

# Any results you write to the current directory are saved as output.


# In[403]:


df1=pd.read_csv("/home/aditya/kaggle/train.csv")
df2=pd.read_csv("/home/aditya/kaggle/test.csv")


# In[404]:


df1.isnull().sum()


# In[405]:


df1.describe()


# In[406]:


df1["location"].dtype


# In[407]:


df1.dtypes


# In[408]:


df1.info()


# In[409]:


df1.drop("vendor",axis=1,inplace=True)


# In[410]:


df1.info()


# In[411]:


df1.head()


# In[412]:


df1["opened_by"].describe()


# In[413]:


df1["opened_by"].fillna(df1["opened_by"].value_counts().idxmax(),inplace=True)


# In[414]:


df1.isnull().sum()


# In[415]:


df1["location"].describe()


# In[416]:


df1["location"].unique()


# In[417]:


df1.isnull().sum()


# In[418]:


df1["location"].describe()


# In[419]:


df1["location"].fillna(df1["location"].value_counts().idxmax(),inplace=True)


# In[420]:


df1.isnull().sum()


# In[421]:


df1["category"].describe()


# In[422]:


df1["category"].fillna(df1["category"].value_counts().idxmax(),inplace=True)


# In[423]:


df1.isnull().sum()


# In[424]:


df1["subcategory"].describe()


# In[425]:


df1["subcategory"].fillna(df1["subcategory"].value_counts().idxmax(),inplace=True)


# In[426]:


df1.head()


# In[427]:


df1.isnull().sum()


# In[428]:


df1["assigned_to"].describe()


# In[429]:


df1["assigned_to"].fillna(df1["assigned_to"].value_counts().idxmax(),inplace=True)


# In[430]:


df1.isnull().sum()


# In[431]:


df1.columns


# In[432]:


df1.head()


# In[ ]:





# In[433]:


df1["update_count"].unique()


# In[434]:


df1.drop(["reassignment_count","reopen_count"],axis=1,inplace=True)


# In[435]:


df1.head()


# In[436]:


df1["notify"].unique()


# In[437]:


df2.head()


# In[438]:


df2.isnull().sum()


# In[439]:


df1["location"].unique()


# In[440]:


len(df1["assigned_to"].unique())


# In[441]:


df1.head()


# In[442]:


testdf1=[df1]
for data in testdf1:
    data["location"]=data["location"].str.extract('(\d+)',expand=False)


# In[443]:


df1.head()


# In[444]:


testdf1=[df1]
for data in testdf1:
    data["category"]=data["category"].str.extract('(\d+)',expand=False)


# In[445]:


df1.head()


# In[446]:


testdf1=[df1]
for data in testdf1:
    data["subcategory"]=data["subcategory"].str.extract('(\d+)',expand=False)


# In[447]:


df1.head()


# In[ ]:





# In[448]:


df1.head()


# In[449]:


df1.info()


# In[450]:


df1["location"]=df1["location"].astype(int)


# In[451]:


df1.info()


# In[452]:


df1["category"]=df1["category"].astype(int)


# In[453]:


df1["subcategory"]=df1["subcategory"].astype(int)


# In[454]:


df1.head()


# In[455]:


df1.info()


# In[456]:


df1["impact"].unique()


# In[457]:


df1.head()


# In[458]:


len(df1["opened_at"].values)


# In[459]:


df1["urgency"].unique()


# In[460]:


df1["priority"].unique()


# In[461]:


testdf1=[df1]
for data in testdf1:
    data["opened_by"]=data["opened_by"].str.extract('(\d+)',expand=False)


# In[462]:


df1.head()


# In[463]:


df1["opened_by"]=df1["opened_by"].astype(int)


# In[464]:


df1.info()


# In[465]:


df1["assigned_to"].unique()


# In[466]:


testdf1=[df1]
for data in testdf1:
    data["assigned_to"]=data["assigned_to"].str.extract('(\d+)',expand=False)


# In[467]:


df1["assigned_to"]=df1["assigned_to"].astype(int)


# In[468]:


df1.head()


# In[469]:


testdf1=[df1]
for data in testdf1:
    data["impact"]=data["impact"].str.extract('(\d+)',expand=False)


# In[470]:


df1["impact"]=df1["impact"].astype(int)


# In[471]:


df1.head()


# In[472]:


df1.info()


# In[473]:


testdf1=[df1]
for data in testdf1:
    data["urgency"]=data["urgency"].str.extract('(\d+)',expand=False)


# In[474]:


df1["urgency"]=df1["urgency"].astype(int)


# In[475]:


testdf1=[df1]
for data in testdf1:
    data["priority"]=data["priority"].str.extract('(\d+)',expand=False)


# In[476]:


df1["priority"]=df1["priority"].astype(int)


# In[477]:


df1.head()


# In[478]:


df1["priority"].unique()


# In[479]:


df1.info()


# In[480]:


df1["knowledge"].unique()


# In[481]:


maptf = {False: 0, True: 1}
for dataset in testdf1:
    dataset['knowledge'] = dataset['knowledge'].map(maptf)


# In[482]:


df1.head()


# In[483]:


df1.info()


# In[484]:


df1["made_sla"].unique()


# In[485]:


maptf = {False: 0, True: 1}
for dataset in testdf1:
    dataset['made_sla'] = dataset['made_sla'].map(maptf)


# In[486]:


df1.head()


# In[487]:


df1["notify"].unique()


# In[488]:


maptf = {"Do Not Notify": 0,  "Send Email":1}
for dataset in testdf1:
    dataset['notify'] = dataset['notify'].map(maptf)


# In[489]:


df1.head()


# In[ ]:





# In[490]:



df1.drop("update_count",axis=1,inplace=True)


# In[491]:


df1.head()


# In[492]:


df1.plot(kind="line",x="Id",y="location")
plt.show()


# In[493]:


import numpy as np
from sklearn.linear_model import LinearRegression


# In[532]:


df1=df1.sample(frac=1)

X=df1[["location","category","subcategory","impact","priority","urgency","assigned_to","opened_by","priority",]]
Y=df1["target_days"]


reg = LinearRegression().fit(X, Y)
reg.score(X, Y)
pred=reg.predict(X)


# In[533]:


from sklearn.metrics import mean_absolute_error
print(mean_absolute_error(Y,pred))


# In[534]:


print(pred)


# In[535]:


from sklearn.model_selection import train_test_split


# In[536]:


X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3)


# In[537]:


reg = LinearRegression().fit(X_train, y_train)
reg.score(X_train, y_train)
pred1=reg.predict(X_test)


# In[538]:


from sklearn.metrics import mean_absolute_error
print(mean_absolute_error(y_test,pred1))


# In[539]:


from sklearn.svm import SVR
clf = SVR( C=1.0, epsilon=0.1)
clf.fit(X_train, y_train)
pred2=clf.predict(X_test)


# In[540]:


from sklearn.metrics import mean_absolute_error
print(mean_absolute_error(y_test,pred2))


# In[541]:


import seaborn as sns


# In[542]:


from sklearn import ensemble
params = {'n_estimators': 800, 'max_depth': 6, 'min_samples_split': 2,
          'learning_rate': 0.03, 'loss': 'ls'}
clf = ensemble.GradientBoostingRegressor(**params)
clf.fit(X_train, y_train)
mse = mean_absolute_error(y_test, clf.predict(X_test))


# In[543]:


print(mse)


# In[544]:


sns.boxplot(df1["location"],Y)


# In[545]:


df2


# In[546]:


df2.info()


# In[547]:


df2.isnull().sum()


# In[548]:


df2.dtypes


# In[549]:


df2["update_count"].unique()


# In[550]:


df2.drop(["reassignment_count","reopen_count","update_count","vendor"],axis=1,inplace=True)


# In[551]:


df2.head()


# In[552]:


df2.drop("made_sla",axis=1,inplace=True)


# In[553]:


df2.isnull().sum()


# In[554]:



df2["category"].describe()


# In[555]:


df2["category"].fillna(df2["category"].value_counts().idxmax(),inplace=True)
df2["subcategory"].fillna(df2["subcategory"].value_counts().idxmax(),inplace=True)
df2["assigned_to"].fillna(df2["assigned_to"].value_counts().idxmax(),inplace=True)


# In[556]:


df2["category"].describe()


# In[557]:


testdf2=[df2]
for data in testdf2:
    data["category"]=data["category"].str.extract('(\d+)',expand=False)
    data["location"]=data["location"].str.extract('(\d+)',expand=False)
    data["subcategory"]=data["subcategory"].str.extract('(\d+)',expand=False)
    data["opened_by"]=data["opened_by"].str.extract('(\d+)',expand=False)
    data["urgency"]=data["urgency"].str.extract('(\d+)',expand=False)
    data["priority"]=data["priority"].str.extract('(\d+)',expand=False)
    data["impact"]=data["impact"].str.extract('(\d+)',expand=False)
    data["assigned_to"]=data["assigned_to"].str.extract('(\d+)',expand=False)


maptf = {False: 0, True: 1}
for dataset in testdf2:
    dataset['knowledge'] = dataset['knowledge'].map(maptf)
    


# In[558]:


df2.head()


# In[559]:


df1.head()


# In[560]:


df2["notify"].unique()


# In[561]:


df1.drop("notify",axis=1,inplace=True)


# In[562]:


df2.drop("notify",axis=1,inplace=True)


# In[563]:


df2.head()


# In[ ]:





# In[564]:


df1.head()


# In[565]:


df2.head()


# In[566]:


df2.info()


# In[567]:


df2["location"]=df2["location"].astype(int)
df2["category"]=df2["category"].astype(int)
df2["opened_by"]=df2["opened_by"].astype(int)
df2["impact"]=df2["impact"].astype(int)
df2["urgency"]=df2["urgency"].astype(int)
df2["priority"]=df2["priority"].astype(int)
df2["subcategory"]=df2["subcategory"].astype(int)
df2["assigned_to"]=df2["assigned_to"].astype(int)


# In[568]:


df2.info()


# In[569]:


df2.isnull().sum()


# In[574]:


df2=df2.sample(frac=1)

X_tester1=df2[["location","category","subcategory","impact","urgency","priority","assigned_to","opened_by","priority",]]
#Y=df2["target_days"]


# In[575]:


from sklearn.svm import SVR
clf1 = SVR( C=1.0, epsilon=0.1)
clf1.fit(X_train, y_train)
prediction=clf1.predict(X_tester1)


# In[605]:


print(prediction)


# In[606]:


r=pd.DataFrame()


# In[ ]:





# In[615]:


r["Id"]=df2["Id"]
r["target_days"]=prediction


# In[ ]:





# In[616]:


r.info()


# In[617]:


r["target_days"]=r["target_days"].astype(int)


# In[618]:


r.head()


# In[619]:


r.drop("target",axis=1,inplace=True)


# In[620]:


r.head()


# In[621]:


r.to_csv('mycsvfile.csv',index=False)


# In[622]:


r.info()


# In[ ]:




