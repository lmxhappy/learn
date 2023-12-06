#!/usr/bin/env python
# coding: utf-8

# In[5]:


import pandas as pd
from sklearn.metrics import precision_score


# In[2]:



act = pd.Series(['A', 'A', 'B', 'B', 'C', 'A', 'C', 'A', 'C',
                'A', 'B', 'B', 'B', 'B', 'C', 'A', 'C', 'C',
                'A', 'C', 'A', 'C', 'A', 'A', 'C', 'C', 'B',
                'B','B','B'], name='Actual')
pred = pd.Series(['A', 'A', 'A', 'A', 'C', 'A', 'C', 'A', 'B', 
                 'B', 'B', 'B', 'B', 'C', 'A', 'C', 'C', 'B',
                 'A', 'B', 'A','C','C','C','C','B','B',
                 'B','B','C'],name='pred')


# In[4]:


confusion_matrix = pd.crosstab(pred, act)
confusion_matrix


# In[8]:


precision_score(act, pred, average=None)


# In[9]:



