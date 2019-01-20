#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from tpot import TPOTRegressor
from sklearn.model_selection import train_test_split


# In[2]:


airdata = pd.read_csv('airdata.csv', sep=',', dtype=np.float64)
airdata


# In[3]:


y = airdata['pm25'].values
#y = airdata.iloc[:,2]
#y.shape
#y = np.array(y)
#y = y.reshape(y.size)
y.shape
#y


# In[4]:


X = airdata.drop('pm25', axis=1).values
X.shape
#X


# In[5]:


X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=None)


# In[6]:


X_train


# In[7]:


X_test


# In[8]:


y_train


# In[9]:


y_test


# In[10]:


tpot = TPOTRegressor(scoring='neg_mean_absolute_error',
                     max_time_mins=5,
                     n_jobs=-1,
                     verbosity=2,
                     cv=5)


# In[11]:


tpot.fit(X_train, y_train)


# In[12]:


tpot.export('tpot_exported_pipeline_airdata.py')


# In[13]:


print(tpot.score(X_test, y_test))


# In[ ]:




