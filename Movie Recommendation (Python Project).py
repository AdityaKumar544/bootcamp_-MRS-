#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import warnings 


# In[2]:


warnings.filterwarnings('ignore')  


# In[4]:


df=pd.read_csv('u.data',sep="\t")  


# In[6]:


df.head(10)  


# In[7]:


df.shape  


# In[8]:


columns_name=['user_id','item_id','rating','timestamp']
df=pd.read_csv('u.data',sep="\t",names=columns_name)


# In[9]:


df.head(10)  


# In[10]:


df['user_id']    


# In[11]:


df['user_id'].nunique()   


# In[12]:


df['item_id'].nunique() 


# In[13]:


movies_title=pd.read_csv('u.item',sep="\|",header=None) 


# In[14]:


movies_title.tail()   


# In[15]:


movies_title.shape     


# In[17]:


movies_titles=movies_title[[0,1]]
movies_titles.columns=["item_id","title"]            
movies_titles.head(7)


# In[18]:


df=pd.merge(df,movies_titles,on="item_id")     


# In[19]:


df


# In[23]:


grades=pd.DataFrame(df.groupby('title').mean()['rating'])     


# In[24]:


grades.head()   


# In[25]:


grades['num of ratings']=pd.DataFrame(df.groupby('title').count()['rating'])   


# In[27]:


df.head(6)


# In[28]:


moviematrix=df.pivot_table(index="user_id",columns="title",values="rating")     


# In[32]:


moviematrix.head(10)


# In[33]:


faust_user_ratings=moviematrix['Faust (1994)']  


# In[34]:


faust_user_ratings.head(10)


# In[35]:


similar_to_faust=moviematrix.corrwith(faust_user_ratings)  


# In[36]:


similar_to_faust


# In[38]:


corr_faust=pd.DataFrame(similar_to_faust,columns=['correlation']) 


# In[39]:


corr_faust.dropna(inplace=True)


# In[40]:


corr_faust


# In[41]:


corr_faust.sort_values('correlation',ascending=True).head(10)   


# In[42]:


ratings


# In[43]:


corr_faust=corr_faust.join(ratings['num of ratings'])   


# In[44]:


corr_faust


# In[45]:


corr_faust[corr_faust['num of ratings']>200].sort_values('correlation')  


# In[62]:


def predict_movies(movie_name):
    movie_user_ratings=moviematrix[movie_name]
    similar_to_movie=moviematrix.corrwith(movie_user_ratings)
    corr_movie=pd.DataFrame(similar_to_movie,columns=['correlation'])
    corr_movie.dropna(inplace=True)
    corr_movie=corr_movie.join(ratings['num of ratings'])
    
    predictions=corr_movie[corr_movie['num of ratings']>200].sort_values('correlation',ascending=False)
    
    return predictions


# In[63]:


predict_my_movie=predict_movies("Heat (1995)")   


# In[64]:


predict_my_movie.head()


# ##                                                                                                                                THANK YOU
