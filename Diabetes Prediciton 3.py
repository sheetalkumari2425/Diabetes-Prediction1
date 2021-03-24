#!/usr/bin/env python
# coding: utf-8

# # Importing library, reading dataset and getting dataset information

# In[ ]:


import pandas as pd


# In[ ]:


diabetes = pd.read_csv('pima-data.csv')


# In[ ]:


diabetes.head(5)


# In[ ]:


diabetes.info()


# In[ ]:


diabetes.shape


# In[ ]:


diabetes['insulin'].value_counts()


# In[ ]:


diabetes.describe()


# #  Converting the labels from boolean to binary

# In[ ]:


data_map = {True:1, False:0}


# In[ ]:


diabetes['diabetes'] = diabetes['diabetes'].map(data_map)


# In[ ]:


diabetes.head(7)


# In[ ]:


true_count = len(diabetes.loc[diabetes['diabetes'] == True])
false_count = len(diabetes.loc[diabetes['diabetes'] == False])


# In[ ]:


(true_count, false_count)


# # Plotting Histogram

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt


# In[ ]:


diabetes.hist(bins=50, figsize=(20, 15))


# # Train-Test Split

# We need to keep some data seperately for training and testing

# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


features = ['num_preg', 'glucose_conc', 'diastolic_bp', 'thickness', 'insulin', 'bmi', 'diab_pred', 'age', 'skin']
labels = ['diabetes']


# In[ ]:


x = diabetes[features].values
y = diabetes[labels].values


# In[ ]:


import numpy as np
shuffle_index = np.random.permutation(500)


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(x[shuffle_index], y[shuffle_index], test_size=0.20, random_state=50)


# # Applying Classifiers

# # 1. Logistic Regression

# In[ ]:


from sklearn.linear_model import LogisticRegression 


# In[ ]:


clf = LogisticRegression(tol=0.1, solver='lbfgs')


# In[ ]:


clf.fit(X_train, y_train)


# In[ ]:


#clf.predict(X_train[:2, :])


# # Cross Validation: cross_val_score

# In[ ]:


from sklearn.model_selection import cross_val_score
a = cross_val_score(clf, X_train, y_train, cv=3, scoring="accuracy")


# In[ ]:


# 74% accuracy.
a.mean()


# # cross_val_predict

# In[ ]:


from sklearn.model_selection import cross_val_predict
y_train_pred = cross_val_predict(clf, X_train, y_train, cv=3)


# In[ ]:


y_train_pred


# # Confusion Matrix

# In[ ]:


from sklearn.metrics import confusion_matrix


# In[ ]:


confusion_matrix(y_train, y_train_pred)


# In[ ]:


# Ideal confusion matrix, when we get accurate prediction
confusion_matrix(y_train, y_train)


# # Precision Recall Curve

# In[ ]:


from sklearn.metrics import precision_recall_curve


# In[ ]:


y_scores = cross_val_predict(clf, X_train, y_train.ravel(), cv=3, method="decision_function")


# In[ ]:


y_scores


# In[ ]:


precisions, recalls, thresholds = precision_recall_curve(y_train, y_scores)


# In[ ]:


precisions


# In[ ]:


recalls


# In[ ]:


thresholds


# # Plotting the Precision Recall Curve

# In[ ]:


plt.plot(thresholds, precisions[:-1], "b--", label="Precision")
plt.plot(thresholds, recalls[:-1], "g-", label="Recall")
plt.xlabel("Thresholds")
plt.legend(loc="upper left")
plt.ylim([0,1])
plt.show()


# # F1_Score

# In[ ]:


from sklearn.metrics import f1_score


# In[ ]:


f1_score(y_train, y_train_pred)


# # 2. Random Forest Classifier

# In[ ]:


from sklearn.ensemble import RandomForestClassifier


# In[ ]:


clf1 = RandomForestClassifier(random_state=10)


# In[ ]:


clf1.fit(X_train, y_train)


# In[ ]:


clf1.predict(X_train[:2, :])


# # Cross Validation: cross_val_score

# In[ ]:


from sklearn.model_selection import cross_val_score
a = cross_val_score(clf1, X_train, y_train, cv=3, scoring="accuracy")


# In[ ]:


a.mean()


# # cross_val_predict

# In[ ]:


from sklearn.model_selection import cross_val_predict
y_train_pred = cross_val_predict(clf1, X_train, y_train, cv=3)


# In[ ]:


y_train_pred

