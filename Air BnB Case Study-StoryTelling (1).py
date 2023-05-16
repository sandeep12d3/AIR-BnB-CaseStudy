#!/usr/bin/env python
# coding: utf-8

# ### StoryTelling Case Study AIR BnB

# In[1]:


# Import the necessary libraries
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns


# In[2]:


# Data Converzsion and Understanding
airbnb = pd.read_csv(r'C:\Users\ABHIPSA\Desktop\Upgrad\AB_NYC_2019.csv')
airbnb.head(15)


# In[3]:


# Analyse the dataste
airbnb.shape


# ###### The dataset contains 48895 rows and 16 columns
# ###### We have to check for missing values

# In[4]:


# Missing Values
airbnb.isnull().sum()


# In[5]:


# Drop columns which are not essential
airbnb.drop(['id','name','last_review'], axis = 1, inplace = True)


# In[6]:


# check if the columns are dropped
airbnb.head(15)


# In[7]:


airbnb.reviews_per_month.isnull().sum()


# In[8]:


# Now reviews per month contains more missing values which should be replaced with 0 respectively
airbnb.fillna({'reviews_per_month':0},inplace=True)


# In[9]:


# Check Unique values of other columns
airbnb.room_type.unique()


# In[10]:


len(airbnb.room_type.unique())


# In[11]:


airbnb.neighbourhood_group.unique()


# In[12]:


len(airbnb.neighbourhood_group.unique())


# In[13]:


airbnb.host_id.value_counts().head(10)


# In[14]:


airbnb2 = airbnb.sort_values(by="calculated_host_listings_count",ascending=False)
airbnb2.head()


# In[16]:


airbnb.to_csv(r'C:\Users\ABHIPSA\Desktop\Upgrad\airbnb.csv',index=False, header=True)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




