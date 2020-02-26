#!/usr/bin/env python
# coding: utf-8

# In[3]:


import keras
from keras import regularizers


# In[6]:


reg_l2 = regularizers.l2(0.01)


# In[7]:


reg_l1 = regularizers.l1(0.02)


# In[13]:


reg_l2.get_config()
reg_l1.get_config()


# In[14]:


type(reg_l2)


# In[ ]:




