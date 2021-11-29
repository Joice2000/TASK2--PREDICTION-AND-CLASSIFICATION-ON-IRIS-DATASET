#!/usr/bin/env python
# coding: utf-8

# # TASK- 2 PREDICTION AND CLASSIFICATION OF IRIS DATASET

# In[1]:


## IMPORTING THE NECESSARY LIBRARIES AND DATASET

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


df= pd.read_csv(r'C:\Users\joice mary\Downloads\iris.data')


# In[3]:


## SIMPLE ANALYSIS OF DATASET
df.head()


# In[4]:


print(df.columns)


# In[5]:


df.shape


# In[6]:


df.columns= ['sepal length','sepal widhth','petal length','petal width','class']
df


# In[7]:


df.head()


# In[8]:


df.tail()


# In[9]:


df.columns = df.columns.str.upper()
df.columns


# In[10]:


df.dtypes


# In[11]:


df.info()


# In[12]:


## CHECKING FOR NULL VALUES

df.isnull().sum()


# In[13]:


df.nunique()


# In[14]:


## CHECKING FOR DUPLICATED VALUES

df.duplicated().sum()


# In[15]:


df.drop_duplicates()


# In[16]:


##STATISTICAL ANALYSIS ON DATASET

df.describe()


# In[17]:


df.hist()


# In[18]:


sns.pairplot(df)


# In[19]:


correlation= df.corr()
sns.heatmap(correlation,xticklabels = correlation.columns,yticklabels= correlation.columns,annot= True)


# In[20]:


## DATA VISUALIZATION
 
## BOXPLOT BETWEEN PETAL LENGTH AND CLASS

sns.boxplot(x="PETAL LENGTH",y="CLASS",data=df)


# In[21]:


## BOXPLOT BETWEEN PETAL WIDTH AND CLASS

sns.boxplot(x="PETAL WIDTH",y="CLASS",data=df)


# In[22]:


## BOXPLOT BETWEEN SEPAL WIDTH AND CLASS

sns.boxplot(x="SEPAL WIDHTH",y="CLASS",data=df)


# In[23]:


## BOXPLOT BETWEEN SEPAL LENGTH AND CLASS

sns.boxplot(x="SEPAL LENGTH",y="CLASS",data=df)


# In[24]:


## SPLITTING THE DATASET FOR TRAINING AND TESTING


feature_columns =['SEPAL LENGTH', 'SEPAL WIDHTH', 'PETAL LENGTH','PETAL WIDTH']
x = df[feature_columns].values
y = df['CLASS'].values


# In[25]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test= train_test_split(x,y,test_size=0.2,random_state=4)


# In[26]:


print(x_train.shape)
print(x_test.shape)


# In[27]:


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)


# In[28]:


#import the KNeighborsClassifier class from sklearn

from sklearn.neighbors import KNeighborsClassifier

from sklearn import metrics
k_range= range(1,20)
scores = {}
scores_list = []
for k in k_range:
    knn= KNeighborsClassifier(n_neighbors=k)
    knn.fit(x_train,y_train )
    y_pred=knn.predict(x_test)
    scores[k] = metrics.accuracy_score(y_test,y_pred)
    scores_list.append(metrics.accuracy_score(y_test,y_pred))


# In[29]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import cross_val_score

# Instantiate learning model (k = 3)
classifier = KNeighborsClassifier(n_neighbors=3)

# Fitting the model
classifier.fit(x_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(x_test)


# In[30]:


from sklearn import metrics
acc_knn = round(metrics.accuracy_score(y_test,y_pred),k)
print(acc_knn)


# In[31]:


cm = confusion_matrix(y_test, y_pred)
cm


# In[33]:


# creating list of K for KNN
k_list = list(range(1,50,2))
# creating list of cv scores
cv_scores = []

# perform 10-fold cross validation
for k in k_list:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, x_train, y_train, cv=10, scoring='accuracy')
    cv_scores.append(scores.mean())


# In[34]:


MSE = [1 - x for x in cv_scores]

plt.figure()
plt.figure(figsize=(15,10))
plt.title('The optimal number of neighbors', fontsize=20, fontweight='bold')
plt.xlabel('Number of Neighbors K', fontsize=15)
plt.ylabel('Misclassification Error', fontsize=15)
sns.set_style("whitegrid")
plt.plot(k_list, MSE)

plt.show()


# In[38]:


best_k = k_list[MSE.index(min(MSE))]
print("The optimal number of neighbors is %d." % best_k)


# In[35]:


knn=KNeighborsClassifier(n_neighbors=19)
knn.fit(x,y)


# In[36]:


new_data=[[3,4,5,6],[7,8,9,10],[1,3.4,5.6,7.8],[3,4,5,2],[5,4,2,2],[3, 2, 4, 0.2], [  4.7, 3, 1.3, 0.2 ]]
new_predict=knn.predict(new_data)
new_predict


# In[37]:


accuracy = accuracy_score(y_test, y_pred)*100
print('Accuracy of our model is equal ' + str(round(accuracy, 2)) + ' %.')

