#!/usr/bin/env python
# coding: utf-8

# # Fetching Dataset

# In[1]:


from sklearn.datasets import fetch_openml


# In[2]:


mnist = fetch_openml('mnist_784')


# In[3]:


mnist


# In[4]:


x,  y = mnist['data'], mnist['target']


# In[5]:


x


# In[6]:


y


# In[7]:


x[0]


# In[8]:


x.shape


# In[9]:


y.shape


# In[10]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[11]:


import matplotlib 
import matplotlib.pyplot as plt


# In[12]:


some_digit = x[36003]
some_digit_image = some_digit.reshape(28,28)    #Lets reshape it to plot it


# In[13]:


plt.imshow(some_digit_image, cmap=matplotlib.cm.binary, interpolation='nearest')
plt.axis('off')


# In[14]:


y[36003]


# In[15]:


x_train, x_test = x[:60000], x[60000:]


# In[16]:


y_train, y_test = y[:60000], y[60000:]


# In[17]:


import numpy as np
shuffle_index = np.random.permutation(60000)
x_train, y_train = x_train[shuffle_index], y_train[shuffle_index]


# # Creating a 2 detector

# In[18]:


y_train = y_train.astype(np.int8)
y_test = y_test.astype(np.int8)
y_train_2 = (y_train==2)
y_test_2 = (y_test==2)


# In[19]:


y_train


# In[20]:


y_test_2


# In[21]:


from sklearn.linear_model import LogisticRegression


# In[22]:


clf = LogisticRegression(tol = 0.1)           #clf =classifier


# In[23]:


clf.fit(x_train, y_train_2)


# In[24]:


clf.predict([some_digit])


# In[25]:


from sklearn.model_selection import cross_val_score
a = cross_val_score(clf, x_train, y_train_2, cv=3, scoring='accuracy')


# In[26]:


a.mean()


# In[ ]:




