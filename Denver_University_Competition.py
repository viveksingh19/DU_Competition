
# coding: utf-8

# Importing the packages

# In[1]:


from __future__ import division, print_function, unicode_literals
from statsmodels.tsa.seasonal import seasonal_decompose


# Common imports
import numpy as np
import os

# to make this notebook's output stable across runs
np.random.seed(42)

# To plot pretty figures
get_ipython().magic(u'matplotlib inline')
import matplotlib
import matplotlib.pyplot as plt
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12

# Where to save the figures
PROJECT_ROOT_DIR = "."

def save_fig(fig_id, tight_layout=True, fig_extension="png", resolution=300):
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig( fig_id,format=fig_extension, dpi=resolution)

import pandas as pd
from datetime import datetime
import  os


# In[2]:


# Matplotlib for plotting
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
get_ipython().magic(u'matplotlib inline')

plt.style.use('fivethirtyeight')
matplotlib.rcParams['axes.labelsize'] = 30
matplotlib.rcParams['xtick.labelsize'] = 30
matplotlib.rcParams['ytick.labelsize'] = 30
matplotlib.rcParams['text.color'] = 'k'

import seaborn; seaborn.set()


# In[3]:


DU = pd.read_csv("//mevwofwfil001ap/MESUsers$/wofmwem/Desktop/R&D Team/DATA/DU/DU_Final/DATA_PY_LEDGER_TABLE_FTE_2.csv")
DU.head(10)


# In[4]:


DU = DU.rename(columns={"DocumentDate": "date" })
DU.head(3)
#DU.dtypes


# In[5]:


DU= DU[(DU['date'] > '2013-01-01') & (DU['date'] < '2018-01-01')]
DU.head()


# In[6]:


DU['date'] =  pd.to_datetime(DU['date'])
DU.dtypes

#DU['date'] = datetime.strptime(DU['date'], '%d-%b-%Y')
#print DU.strftime('%d-%m-%Y')


# In[8]:


#df.index = pd.DatetimeIndex(df['date'], inplace=True)
#df.head()
#df.drop('date', axis=1, inplace=True)
DU.head()


# In[9]:


DU.index = pd.DatetimeIndex(DU['date'], inplace=True)
DU.head()


# In[10]:


M=DU.resample('M').mean()
M.head()
M=M.fillna(method='ffill')


# In[11]:


#M.shape
M.head()


# In[12]:


DU.head()


# In[13]:


DU.shape


# In[14]:


DU['FBCProductTypeCode']=DU['FBCProductTypeCode'].astype('category')
DU.dtypes


# In[15]:


pd_code=DU['FBCProductTypeCode'].unique()
#df.name.unique()

pd_code


# In[85]:


A=DU[DU.FBCProductTypeCode == pd_code[5]]
A.head()

#movies[movies.duration >= 200]
#A=A.fillna(method='ffill')


# In[16]:


df1 = DU[DU.FBCProductTypeCode == '1000']
#df1=df1.resample('M').mean()

df1.drop(['EntryType','Quantity','date'], axis=1, inplace=True)
df1['INCOMING_QTY_t'] = np.nan
df1['INCOMING_QTY_s'] = np.nan
df1.head()


# In[17]:


df1.head()


# In[18]:


df2= DU[DU.FBCProductTypeCode == pd_code[3]]
df2=df2.resample('M').mean()
df2=df2.fillna(method='ffill')
df2['FBCProductTypeCode']=pd_code[3]
decomposition = seasonal_decompose(df2['INCOMING_QTY'], model='additive',freq = 12)
trend = decomposition.trend
seasonal = decomposition.seasonal
seasonal=seasonal.to_frame(name = None)
trend= trend.to_frame(name = None)
df2= df2.join(trend, lsuffix='', rsuffix='_t')
df2= df2.join(seasonal, lsuffix='', rsuffix='_s')
df2.head(20)


# In[19]:


df3= df2[df2.FBCProductTypeCode == 1000]
df3.head()


# In[194]:


for i, j in zip(range(len(pd_code)), pd_code): 
    df2= DU[DU.FBCProductTypeCode == pd_code[i]]
    df2=df2.resample('M').mean()
    df2=df2.fillna(method='bfill')
    df2['FBCProductTypeCode']=pd_code[i]
    decomposition = seasonal_decompose(df2['INCOMING_QTY'], model='additive',freq = 12)
    trend = decomposition.trend
    seasonal = decomposition.seasonal
    seasonal=seasonal.to_frame(name = None)
    trend= trend.to_frame(name = None)
    df2= df2.join(trend, lsuffix='', rsuffix='_t')
    df2= df2.join(seasonal, lsuffix='', rsuffix='_s')
    df3= df3.append(df2)


# In[21]:


df3.to_csv('DUtimeseries.csv')

#os.getcwd()
#os.chdir("//mevwofwfil001ap/MESUsers$/wofmwem/Desktop/R&D Team/DATA/DU/DU_Final")


# In[20]:


import pandas as pd
trend = trend.to_frame(name = None)
trend.head()


# In[22]:


df1['INCOMING_QTY_t'] = np.nan
df1['INCOMING_QTY_s'] = np.nan
df1.head()


# In[23]:


df2= DU[DU.FBCProductTypeCode == pd_code[3]]
df2=df2.resample('M').mean()
df2=df2.fillna(method='ffill')
df2['FBCProductTypeCode']=pd_code[3]
decomposition = seasonal_decompose(df2['INCOMING_QTY'], model='additive',freq = 12)
trend = decomposition.trend
seasonal = decomposition.seasonal
seasonal=seasonal.to_frame(name = None)
trend= trend.to_frame(name = None)
df2= df2.join(trend, lsuffix='', rsuffix='_t')
df2= df2.join(seasonal, lsuffix='', rsuffix='_s')
df2.head(20)


# In[24]:


df1.shape


# In[25]:


df1.head()


# In[109]:


df1.shape


# In[18]:


DU.head()


# In[19]:


from statsmodels.tsa.seasonal import seasonal_decompose


for i in DU['FBCProductTypeCode'].unique():
    df=M[M['FBCProductTypeCode']==i]
    df= M.loc[:, 'INCOMING_QTY']

    decomposition = seasonal_decompose(df, model='additive',freq = 12)

    trend = decomposition.trend
    seasonal = decomposition.seasonal
    residual = decomposition.resid
    print(i)
    plt.subplot(411)
    plt.plot(df, label='Original')
    plt.legend(loc='best')
    plt.subplot(412)
    plt.plot(trend, label='Trend')
    plt.legend(loc='best')
    plt.subplot(413)
    plt.plot(seasonal,label='Seasonality')
    plt.legend(loc='best')
    plt.subplot(414)
    plt.plot(residual, label='Residuals')
    plt.legend(loc='best')

    plt.tight_layout()
    plt.show()
    plt.close()


# In[20]:


from statsmodels.tsa.seasonal import seasonal_decompose


for i in DU['FBCProductTypeCode'].unique():
    df=M[M['FBCProductTypeCode']==i]
    df= M.loc[:, 'INCOMING_QTY']

    decomposition = seasonal_decompose(df, model='additive',freq = 12)

    trend = decomposition.trend
    seasonal = decomposition.seasonal
    residual = decomposition.resid
    print(i)
    plt.plot(seasonal,label='Seasonality')
    plt.legend(loc='best')

    plt.tight_layout()
    plt.show()
    plt.close()


# In[21]:


from statsmodels.tsa.seasonal import seasonal_decompose


for i in DU['FBCProductTypeCode'].unique():
    df=M[M['FBCProductTypeCode']==i]
    df= M.loc[:, 'INCOMING_QTY']

    decomposition = seasonal_decompose(df, model='additive',freq = 12)

    trend = decomposition.trend
    seasonal = decomposition.seasonal
    residual = decomposition.resid
    print(i)
    plt.plot(trend, label='Trend')
    plt.legend(loc='best')

    plt.tight_layout()
    plt.show()
    plt.close()


# In[28]:


#pd.isnull(trend).sum()
#pd.isnull(seasonal).sum()
seasonal.head()


# In[43]:


seasonal=pd.DataFrame(seasonal)
trend=pd.DataFrame(trend)


#merged = pd.merge(seasonal,trend,how=left,  on=['date'])
#merged.head()
#seasonal.join(trend., how='left', lsuffix='_s', rsuffix='_t', sort=True) 

x= seasonal.join(trend, lsuffix='_s', rsuffix='_t')
x.head(20)


# In[44]:


M.head()

