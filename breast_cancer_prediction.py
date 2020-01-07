


import numpy as np # linear algebra
import os
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns # data visualization library  
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression # to apply the Logistic regression
from sklearn.model_selection import train_test_split # to split the data into two parts
# use for cross validation
from sklearn.model_selection import GridSearchCV# for tuning parameter
from sklearn.ensemble import RandomForestClassifier # for random forest classifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm # for Support Vector Machine
from sklearn import metrics
from sklearn.externals.six import StringIO  
from IPython.display import Image 
from sklearn.tree import export_graphviz
from IPython.display import SVG
import pickle
from flask import request,jsonify
import json



pwd



df = pd.read_csv(r"C:\Users\Senali Perera\Downloads\breast-cancer-wisconsin-data\data.csv")



df.head()



df.columns



list = ['Unnamed: 32','id']
df= df.drop(list,axis = 1 )
df.head()



y = df.diagnosis
ax = sns.countplot(y,label="Count")       # M = 212, B = 357
B, M = y.value_counts()
print('Number of Benign: ',B)
print('Number of Malignant : ',M)




df=df.replace('M',1)
df=df.replace('B',0)

df.head()


df.describe()


f,ax = plt.subplots(figsize=(18, 18))
sns.heatmap(df.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)



drop_list1 = ['perimeter_mean','radius_mean','compactness_mean','concave points_mean','radius_se','perimeter_se','radius_worst','perimeter_worst','compactness_worst','concave points_worst','compactness_se','concave points_se','texture_worst','area_worst']
df = df.drop(drop_list1,axis = 1 )        # do not modify x, we will use it later 
df.head()




df.columns



f,ax = plt.subplots(figsize=(14, 14))
sns.heatmap(df.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)


train, test = train_test_split(df, test_size = 0.3)
print(train.shape)
print(test.shape)




train_y=train.diagnosis
list = ['diagnosis']
train_X= train.drop(list,axis = 1 )

test_y =test.diagnosis  
list = ['diagnosis']
test_X= test.drop(list,axis = 1 )

train_y1=train.diagnosis
list = ['diagnosis','texture_se', 'area_se', 'smoothness_se', 'concavity_se', 'symmetry_se',
       'fractal_dimension_se', 'smoothness_worst', 'concavity_worst',
       'symmetry_worst', 'fractal_dimension_worst']
train_X1= train.drop(list,axis = 1 )

test_y1 =test.diagnosis  
list = ['diagnosis','texture_se', 'area_se', 'smoothness_se', 'concavity_se', 'symmetry_se',
       'fractal_dimension_se', 'smoothness_worst', 'concavity_worst',
       'symmetry_worst', 'fractal_dimension_worst']
test_X1= test.drop(list,axis = 1 )



model=RandomForestClassifier(n_estimators=100)
model.fit(train_X,train_y)


model1=RandomForestClassifier(n_estimators=100)
model1.fit(train_X1,train_y1)



prediction=model.predict(test_X)
metrics.accuracy_score(prediction,test_y)



prediction=model1.predict(test_X1)
metrics.accuracy_score(prediction,test_y1)




model = svm.SVC()
model.fit(train_X,train_y)

prediction=model.predict(test_X)
metrics.accuracy_score(prediction,test_y)




with open('model1.pkl', 'wb') as handle:
    pickle.dump(model1, handle, pickle.HIGHEST_PROTOCOL)





with open('model.pkl', 'wb') as handle:
    pickle.dump(model, handle, pickle.HIGHEST_PROTOCOL)




LR=LogisticRegression(random_state=0, solver='lbfgs', multi_class='ovr').fit(train_X,train_y)




prediction=LR.predict(test_X)
metrics.accuracy_score(prediction,test_y)




clf=DecisionTreeClassifier()
clf=clf.fit(train_X,train_y)
prediction=clf.predict(test_X)
metrics.accuracy_score(prediction,test_y)



import requests
url = 'http://localhost:5000/api'
data=json.dumps({'texture_mean':10.38,'area_mean':1001.0,'smoothness_mean':0.11840,'concavity_mean':0.3001,'symmetry_mean':0.2419,'fractal_dimension_mean':0.07871,})
r = requests.post(url,data)
print(r.json())






