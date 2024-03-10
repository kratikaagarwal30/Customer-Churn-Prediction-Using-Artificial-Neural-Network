#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


df = pd.read_csv("churn_prediction.csv")


# In[3]:


df.head()


# UNDERSTANDING THE DATA

# In[4]:


df.sample(5)


# In[5]:


df.shape


# In[6]:


df.isnull().sum()


# In[7]:


df.duplicated().sum()


# In[8]:


from skimpy import skim
skim(df)


# In[9]:


df.info()


# In[10]:


num_cols = df.select_dtypes(include = ["float64", "int64"]).columns
num_data = df[num_cols]


# In[11]:


num_data.corr()["Exited"]


# UNIVARIATE AND BIVARIATE ANALYSIS

# In[12]:


#CATEGORICAL DATA
df["Geography"].value_counts().plot(kind = 'pie', autopct = "%.2f")


# In[13]:


df["Gender"].value_counts().plot(kind = "pie", autopct = "%.2f")


# In[14]:


#For numerical columns
plt.figure(figsize = (20,10))
plot_no = 1
for col in num_data:
    plt.subplot(5,5,plot_no)
    sns.boxplot(x = df[col])
    plot_no+=1
plt.tight_layout()


# In[15]:


#Distribution plot of numerical columns
cols_to_plot = num_data.columns
num_cols = len(num_data.columns)
num_rows = num_cols //5 +1 if num_cols %5 != 0 else num_cols//5
fig,axes = plt.subplots(nrows = num_rows, ncols = 5, figsize = (20,4*num_rows))
axes = axes.flatten()
for i , col in enumerate (cols_to_plot):
    ax = axes[i]
    sns.histplot(df[col], ax = ax, kde = True)
    ax.set_title(col)
#to remove extra plots
for j in range(i+1, len(axes)):
    fig.delaxes(axes[j])
plt.tight_layout()
plt.show()



# MULTIVARIATE ANALYSIS

# In[16]:


sns.pairplot(df)


# In[17]:


df.columns


# In[18]:


df = pd.get_dummies(df,columns = ["Geography", "Gender"], drop_first = True)
df.head()


# TRAINING THE MODEL

# In[19]:


x = df.drop(columns = ["Exited"])
y = df["Exited"]


# In[20]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train,y_test = train_test_split(x, y, test_size = 0.2, random_state = 42)


# In[21]:


X_train.shape


# In[22]:


X_test.shape


# In[23]:


y_train.shape


# In[24]:


y_test.shape


# In[25]:


#scaling imput data
from sklearn.preprocessing import StandardScaler
std = StandardScaler()
X_train_scaled = std.fit_transform(X_train)
X_test_scaled = std.fit_transform(X_test)


# In[26]:


X_test_scaled


# In[27]:


X_train_scaled


# Building model using deep learning

# In[28]:


import tensorflow
from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense


# In[29]:


model = Sequential()
model.add(Dense(13, activation = "sigmoid", input_dim = 13))
model.add(Dense(1, activation = "sigmoid"))


# In[30]:


model.summary()


# In[31]:


model.compile(loss = "binary_crossentropy", optimizer = "Adam", metrics = ["accuracy"])


# In[32]:


X_train = X_train.astype(np.float32)
y_train = y_train.astype(np.float32)


# In[33]:


#checking history of trained model
hist = model.fit(X_train, y_train, epochs = 10)


# In[34]:


hist.history


# In[35]:


import matplotlib.pyplot as plt
plt.plot(hist.history['loss'])
plt.show()


# In[36]:


model.layers[0].get_weights()


# In[37]:


model.predict(X_test_scaled)


# In[38]:


y_pred = np.where(model.predict(X_test_scaled)>0.5, 1,0)
y_pred


# In[39]:


from sklearn.metrics import accuracy_score
accuracy_score(y_test,y_pred)


# NOW TRAINING THE MODEL WITH THE HELP OF OTHER ACTIVATION FUNCTION

# In[40]:


model1 = Sequential()
#input layer
model1.add(Dense(13, activation = "relu", input_dim = 13))
#hidden layer
model1.add(Dense(13, activation = "relu"))
#output layer
model1.add(Dense(1, activation = "sigmoid"))


# In[41]:


model1.summary()


# In[42]:


model1.compile(loss = "binary_crossentropy", optimizer = "Adam",metrics = ["accuracy"])


# In[43]:


hist1 = model1.fit(X_train_scaled, y_train, epochs = 40, validation_split = 0.2)


# In[44]:


hist1.history


# In[46]:


import matplotlib.pyplot as plt
plt.plot(hist1.history['loss'])
plt.plot(hist1.history["val_loss"])
plt.show()


# In[48]:


model1.layers[0].get_weights()


# In[49]:


model1.predict(X_test_scaled)


# In[50]:


y_pred = np.where(model1.predict(X_test_scaled)>0.5,1,0)
y_pred


# In[51]:


from sklearn.metrics import accuracy_score
accuracy_score(y_test,y_pred)


# In[52]:


#comparing the losses of both activation function
import matplotlib.pyplot as plt
plt.plot(hist.history['loss'])
plt.plot(hist1.history["loss"])
plt.show()


# In[ ]:




