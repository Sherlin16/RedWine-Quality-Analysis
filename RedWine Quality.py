#!/usr/bin/env python
# coding: utf-8

# # IMPORTING LIBRARIES

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import plotly.express as px
import plotly.subplots as sp
import plotly.graph_objects as go

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

#MODELS
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC


from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix


get_ipython().run_line_magic('matplotlib', 'inline')


# # LOADING DATASET

# In[2]:


df = pd.read_csv('redwine_quality.csv')


# # UNDERSTANDING THE DATA

# In[3]:


df.shape


# In[4]:


df.head(5)


# In[5]:


df.tail(5)


# In[6]:


df.info()


# In[7]:


df.columns


# In[8]:


df.isnull().sum()


# **The rows where reduced to 1359 because of 240 duplicate rows**

# In[9]:


df.describe().T


# * The maximum of quality is 8 and the minimum is 3
# * The alcohol content ranges from 8.4 to 14.9 and the average is 10.4
# * The huge difference between max value of total sulphur dioxide and its average

# # EDA

# In[10]:


plt.figure(figsize = (10, 4))
z = df.quality.value_counts()
sns.barplot(x = z.index, y = z.values, order = z.index, palette = 'winter')
plt.title('Countplot - Quality')
for index, value in enumerate(z.values):
    plt.text(index, value, value, ha = 'center', va = 'bottom', fontweight = 'black')


# * The most rated quality is 5 & 6
# * The least rated 8 & 3

# In[11]:


value = df['quality'].value_counts().values
index = df['quality'].value_counts().index
explode = [0.1, 0, 0, 0, 0, 0]
plt.pie(value[:], labels = index[:], autopct = '%1.1f%%', explode = explode)
plt.title('Pie chart - quality')


# * The tastiest RedWine is 1.1% (i.e Least - **8**) out of the five top quality
# * 5 & 6 are the medium quality which is the top 1st and 2nd 
# 

# In[12]:


sns.kdeplot(data = df, x = 'pH', fill = True)
plt.title('KDE - pH')
plt.show()


# * 3.30 pH is the highest density
# * 2.74 is minimun and 4.01 is the maximum

# In[13]:


fig = sp.make_subplots(rows = 1, cols = 2, subplot_titles = ('fixed acidity', 'volatile acidity'))
fig.add_trace(go.Box(y = df['fixed acidity']), row = 1, col = 1)
fig.add_trace(go.Box(y = df['volatile acidity']), row = 1, col = 2)


# In[14]:


plt.figure(figsize = (6, 4))
sns.stripplot(x = 'alcohol', data = df, color = 'green')
plt.title('Strip plot - alcohol')


# The alochol ranges from 9 to 12

# In[15]:


plt.figure(figsize = (4, 5))
sns.boxplot(y = 'total sulfur dioxide', data = df, color = 'blue')
plt.title('Box plot - total sulfur dioxide')


# * From the above boxplot the minimum value lies slightly above 0 and the maximum value is like 125-130
# * More than than the test are outliers.
# * Q1 ranges from apprx. 20
# * Q2 is 46.46 from the above found mean
# * Q3 ranges from 46.46 to 60 apprx.

# In[16]:


plt.figure(figsize = (10, 5))
sns.heatmap(df.corr(),annot = True, cmap = 'viridis')

plt.title('Correlation Heatmap')


# **The correlation of quality is higher in:**
# * fixed acidity
# * citric acid
# * residual sugar
# * sulphates
# * alcohol
# 

# In[17]:


f, ax = plt.subplots(1, 2, figsize = (15,5))
sns.lineplot(x = 'quality', y = 'citric acid', data = df, ax = ax[0])
sns.lineplot(x = 'quality', y = 'volatile acidity', data = df, ax = ax[1])
pass


# * If the Volatile acidity level decreases there is an increase in the quality of Red Wine
# * Composition of citric acid go higher as we go higher in the quality of the wine

# In[18]:


plt.figure(figsize = (12, 6))
sns.barplot(x = 'quality', y = 'sulphates', data = df, color = 'purple', errorbar = None)


# If the Sulphates are more than 70%, the quality of the Redwine is good

# In[19]:


sns.set_style('whitegrid')
plt.figure(figsize = (10, 5))
sns.lineplot(x = 'quality', y = 'alcohol', data = df, color = 'blue')


# * Alcohol level determines the RedWine Quality.
# * If the alcohol level is more than 11, the Wine is tasty and good

# # FEATURE ENGINEERING

# In[20]:


df.info()


# In[21]:


bins = (2, 6.5, 8)
group_names = ['bad', 'good']
df['quality'] = pd.cut(df['quality'], bins = bins, labels = group_names)


# In[22]:


plt.figure(figsize = (15, 6))
plt.subplot(1,2,1)
quality_counts = df['quality'].value_counts()
sns.barplot(x = quality_counts.index, y = quality_counts.values, palette = ['#1d7874','#8B0000'])
plt.title('Wine Quality Value Counts', fontweight = 'black', size = 20, pad = 20)
for index, values in enumerate(quality_counts.values):
    plt.text(index, values, values, ha = 'center', fontweight = 'black', fontsize = 18)

plt.subplot(1,2,2)
plt.pie(quality_counts, labels = ['Bad', 'Good'], autopct = '%.2f%%', textprops = {'fontweight':'black', 'size':15},
        colors = ['#1d7874', '#AC1F29'], explode = [0,0.1], startangle = 90)
center_circle = plt.Circle((0, 0), 0.3, fc = 'white')
fig = plt.gcf()
fig.gca().add_artist(center_circle)
plt.title('Wine Quality Values Distrbution' ,fontweight = 'black', size = 20, pad = 10)
plt.show()


# * The BAD quality wine is more in numbers 1382
# * The GOOD quality wine is less in numbers 217 which is only 13.57% 

# In[23]:


le = LabelEncoder()
df['quality'] = le.fit_transform(df['quality'])


# In[24]:


df.drop_duplicates(inplace = True)


# In[25]:


X = df.drop(columns = ['quality'])
y = df['quality']


# # SPLITTING DATASET

# In[26]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)


# In[27]:


X.shape


# In[28]:


y.shape


# # K-NEIGHBORS CLASSIFIER

# In[29]:


model_knn = KNeighborsClassifier(n_neighbors = 5)
model_knn.fit(X_train, y_train)


# In[30]:


y_pred_knn = model_knn.predict(X_test)


# **ACCURACY OF KNN**

# In[31]:


accuracy_knn = accuracy_score(y_test, y_pred_knn)
print("Accuracy of KNeighbors Classifier:", accuracy_knn * 100)


# **CLASSIFICATION REPORT FOR KNN**

# In[32]:


print('\nCLASSIFICATION REPORT FOR KNN')
report_knn = classification_report(y_test, y_pred_knn)
print(report_knn)


# **CONFUSION MATRIX FOR KNN**

# In[33]:


con_mat_knn = confusion_matrix(y_test, y_pred_knn)

print('\nCONFUSION MATRIX FOR KNN')
plt.figure(figsize = (6, 4))
sns.heatmap(con_mat_knn, annot = True,fmt = 'd', cmap = 'YlGnBu')


# # RANDOM FOREST CLASSIFIER

# In[34]:


model_rfc = RandomForestClassifier(n_estimators = 450)
model_rfc.fit(X_train, y_train)


# In[35]:


y_pred_rfc = model_rfc.predict(X_test)


# **ACCURACY FOR RFC**

# In[36]:


accuracy_rfc = accuracy_score(y_test, y_pred_rfc)
print("Accuracy of Random Forest Classifier:", accuracy_rfc * 100)


# **CLASSIFICATION REPORT FOR RFC**

# In[37]:


print('\nCLASSIFICATION REPORT FOR RFC')
report_rfc = classification_report(y_test, y_pred_rfc)
print(report_rfc)


# **CONFUSION MATRIX FOR RFC**

# In[38]:


con_mat_rfc = confusion_matrix(y_test, y_pred_rfc)

print('\nCONFUSION MATRIX FOR RFC')
plt.figure(figsize = (6, 4))
sns.heatmap(con_mat_rfc, annot = True,fmt = 'd', cmap = 'YlGnBu')


# # DECISION TREE CLASSIFIER

# In[39]:


model_dt = DecisionTreeClassifier()
model_dt.fit(X_train, y_train)


# In[40]:


y_pred_dt = model_dt.predict(X_test)


# **ACCURACY FOR DT**

# In[41]:


accuracy_dt = accuracy_score(y_test, y_pred_dt)
print("Accuracy of Decision Tree Classifier:", accuracy_dt * 100)


# **CLASSIFICATION REPORT FOR DT**

# In[42]:


print('\nCLASSIFICATION REPORT FOR DT')
report_dt = classification_report(y_test, y_pred_dt)
print(report_dt)


# **CONFUSION MATRIX FOR DT**

# In[43]:


con_mat_dt = confusion_matrix(y_test, y_pred_knn)

print('\nCONFUSION MATRIX FOR KNN')
plt.figure(figsize = (6, 4))
sns.heatmap(con_mat_dt, annot = True,fmt = 'd', cmap = 'YlGnBu')


# # LOGISTIC REGRESSION

# In[44]:


model_lr = LogisticRegression()
model_lr.fit(X_train, y_train)


# In[45]:


y_pred_lr = model_lr.predict(X_test)


# **ACCURACY FOR LR**

# In[46]:


accuracy_lr = accuracy_score(y_test, y_pred_lr)
print("Accuracy of Logistic regression :", accuracy_lr * 100)


# **CLASSIFICATION REPORT FOR LR**

# In[47]:


print('\nCLASSIFICATION REPORT FOR LR')
report_lr = classification_report(y_test, y_pred_lr)
print(report_lr)


# **CONFUSION MATRIX FOR LR**

# In[48]:


con_mat_lr = confusion_matrix(y_test, y_pred_lr)

print('\nCONFUSION MATRIX FOR LR')
plt.figure(figsize = (6, 4))
sns.heatmap(con_mat_lr, annot = True,fmt = 'd', cmap = 'YlGnBu')


# # SUPPORT VECTOR CLASSIFIER

# In[49]:


model_svc = SVC()
model_svc.fit(X_train, y_train)


# In[50]:


y_pred_svc = model_svc.predict(X_test)


# **ACCURACY FOR SVC**

# In[51]:


accuracy_svc = accuracy_score(y_test, y_pred_svc)
print('Accuracy of SVC :', accuracy_svc * 100)


# **CLASSIFICATION REPORT FOR SVC**

# In[52]:


print('\nCLASSIFICATION REPORT FOR SVC')
report_svc = classification_report(y_test, y_pred_svc)
print(report_svc)


# **CONFUSION MATRIX FOR SVC**

# In[53]:


con_mat_svc = confusion_matrix(y_test, y_pred_svc)

print('\nCONFUSION MATRIX FOR SVC')
plt.figure(figsize = (6, 4))
sns.heatmap(con_mat_svc, annot = True,fmt = 'd', cmap = 'YlGnBu')


# In[ ]:




