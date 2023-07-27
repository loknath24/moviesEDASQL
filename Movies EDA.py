#!/usr/bin/env python
# coding: utf-8

# # IMPORT THE LIBRARIES

# In[1]:



import pandas as pd
import numpy as np
import seaborn as sns

import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import matplotlib
plt.style.use('ggplot')
from matplotlib.pyplot import figure

get_ipython().run_line_magic('matplotlib', 'inline')
matplotlib.rcParams['figure.figsize'] = (12,8)


# # Read in the data

# In[4]:


df = pd.read_csv(r'C:\Users\joybose\Desktop\F1,,,,DATA\movies.csv')


# In[5]:


df


# # FIND IF WE HAVE ANY MISSING VALUES

# In[6]:


for col in df.columns:
    pct_missing = np.mean(df[col].isnull())
    print('{} - {}%'.format(col, round(pct_missing*100)))


# # Data types for our columns

# In[7]:


print(df.dtypes)


# # Change Data types of Columns

# In[8]:


df['budget'] = df['budget'].astype('int64',errors='ignore')

df['gross'] = df['gross'].astype('int64',errors='ignore')


# In[9]:


df


# # Create  correct year column

# In[10]:


df['yearcorrect']=df['released'].astype(str).str[:4]


# In[11]:


df


# In[12]:


df.sort_values(by=['gross'],inplace=False,ascending=False)


# In[13]:


pd.set_option('display.max_rows',None)


# # check any duplicates

# In[14]:


df['company'].drop_duplicates().sort_values(ascending=True)


# # correlations

# # Scatter plot with budget vs gross

# In[20]:


plt.scatter(x=df['budget'], y=df['gross'])
plt.title('Budget vs Gross Earnings')
plt.xlabel('Gross Earnings')
plt.ylabel('Budget')
plt.show


# # Plot the budget vs gross using seaborn

# In[23]:


sns.regplot(x='budget',y='gross',data=df,scatter_kws={"color":"red"},line_kws={"color":"black"})


# In[26]:


df.corr(method='kendall') #pearson
df.corr(method='pearson')
df.corr(method='spearman')


# In[27]:


#high correlation between budget and gross


# In[29]:


correlation_matrix = df.corr(method='pearson')

sns.heatmap(correlation_matrix,annot=True)

plt.title('Correlation matrix for numeric features')

plt.xlabel('Movie Features')

plt.ylabel('Budget for Film')

plt.show


# In[31]:


df_numerized = df

for col_name in df_numerized.columns:
    if(df_numerized[col_name].dtype == 'object'):
        df_numerized[col_name] = df_numerized[col_name].astype('category')
        df_numerized[col_name]=df_numerized[col_name].cat.codes

df_numerized


# In[32]:


correlation_matrix = df.apply(lambda x: x.factorize()[0]).corr(method='pearson')

sns.heatmap(correlation_matrix, annot = True)

plt.title("Correlation matrix for Movies")

plt.xlabel("Movie features")

plt.ylabel("Movie features")

plt.show()


# In[33]:


correlation_mat = df_numerized.corr()

corr_pairs = correlation_mat.unstack()

corr_pairs

