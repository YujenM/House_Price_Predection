#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
matplotlib.rcParams["figure.figsize"] = (20,10) #runtime parameters


# In[2]:


# pd.set_option('display.max_rows', None)



# ##### Load the csv file into a variable

# In[3]:


df1 = pd.read_csv("Bengaluru_House_Data.csv")
df1.head()


# In[4]:


df1.shape #13k rows so this is a decently sized dataset. 


# I do not want to complicate the data and I am going to drop the columns which I believe will not contribute much to the analysis of data. This is also a very important part of data cleaning 

# In[5]:


df2 = df1.drop(['area_type', 'society', 'availability', 'balcony'], axis = 'columns')
df2


# Data cleaning missing values

# In[6]:


df2.isnull().sum()


# we already have 13k rows, we are fine to drop the na values

# In[7]:


df3 = df2.dropna()
df3.isnull().sum()


# In[8]:


df3.shape


# ## Feature Engineering
# 

# In[9]:


df3['size'].unique()


# In[10]:


df3['bhk'] = df3['size'].apply(lambda x : int(x.split(' ')[0]))


# In[11]:


df3.head()


# In[12]:


df3['bhk'].unique()


# In[13]:


df3[df3.bhk > 20]


# In[14]:


df3['total_sqft'].unique()


# Range is not very useful to me. I need to calculate the total_sqft into float.

# In[15]:


def is_float(x):
    try: 
        float(x)
    except: 
        return False
    return True


# In[16]:


df3[~df3['total_sqft'].apply(is_float)].head(20)


# In[17]:


def convert_sqft_to_num(x):
    tokens = x.split('-')
    if len(tokens) == 2:
        return (float(tokens[0])+float(tokens[1]))/2
    try:
        return float(x)
    except:
        return None   


# In[18]:


convert_sqft_to_num('2100 - 2850')


# In[19]:


df4 = df3.copy()
df4['total_sqft'] = df4['total_sqft'].apply(convert_sqft_to_num)
df4.head()


# In[20]:


df4.loc[30]


# In[21]:


df5 = df4.copy()
df5["price_per_sqft"] = df5["price"] * 100000 / df5["total_sqft"]
df5.head()


# Now let us analyze one of the most important aspects of house price prediction. Location

# In[22]:


# lets get unique location
len(df5.location.unique())


# You will be aware that a lot of these 1300 columns will only contain one or two data points, a good way would be to categorize this into a single column, called the Other column.

# In[23]:


df5.location = df5.location.apply(lambda x: x.strip())
location_stats = df5.groupby('location')['location'].agg('count').sort_values(ascending = False)
location_stats


# In[24]:


location_stats.values.sum()


# In[25]:


len(location_stats[location_stats<=10])


# In[26]:


location_stats_less_than_10 = location_stats[location_stats<=10]
location_stats_less_than_10


# In[27]:


len(df5.location.unique())


# In[28]:


df5.location = df5.location.apply(lambda x: 'other' if x in location_stats_less_than_10 else x)
len(df5.location.unique())


# In[29]:


df5.head(10)


# Outlier Detection and Analysis

# As a data scientist when you have a conversation with your business manager (who has expertise in real estate), he will tell you that normally square ft per bedroom is 300 (i.e. 2 bhk apartment is minimum 600 sqft. If you have for example 400 sqft apartment with 2 bhk than that seems suspicious and can be removed as an outlier. We will remove such outliers by keeping our minimum thresold per bhk to be 300 sqft
# 
# 

# In[30]:


df6 = df5[~(df5.total_sqft/df5.bhk<300)] 
df6.shape


# ##### Outlier Removal Using Standard Deviation and Mean

# In[31]:


df6.price_per_sqft.describe()


# Here we find that min price per sqft is 267 rs/sqft whereas max is 12000000, this shows a wide variation in property prices. We should remove outliers per location using mean and one standard deviation
# 
# 

# In[32]:


def remove_pps_outliers(df):
    df_out = pd.DataFrame()
    for key, subdf in df.groupby('location'):
        m = np.mean(subdf.price_per_sqft)
        st = np.std(subdf.price_per_sqft)
        reduced_df = subdf[(subdf.price_per_sqft>(m-st)) & (subdf.price_per_sqft<=(m+st))]
        df_out = pd.concat([df_out,reduced_df],ignore_index=True)
    return df_out
df7 = remove_pps_outliers(df6)
df7.shape


# Let's check if for a given location how does the 2 BHK and 3 BHK property prices look like
# 
# 

# In[49]:


def plot_scatter_chart(df,location):
    bhk2 = df[(df.location==location) & (df.bhk==2)]
    bhk3 = df[(df.location==location) & (df.bhk==3)]
    matplotlib.rcParams['figure.figsize'] = (10,6)
    plt.scatter(bhk2.total_sqft,bhk2.price,color='blue',label='2 BHK', s=50)
    plt.scatter(bhk3.total_sqft,bhk3.price,marker='+', color='green',label='3 BHK', s=50)
    plt.xlabel("Total Square Feet Area")
    plt.ylabel("Price (Lakh Indian Rupees)")
    plt.title(location)
    plt.legend()
    
plot_scatter_chart(df7,"Rajaji Nagar")


# We should also remove properties where for same location, the price of (for example) 3 bedroom apartment is less than 2 bedroom apartment (with same square ft area). What we will do is for a given location, we will build a dictionary of stats per bhk, i.e.
# 
# {
#     '1' : {
#         'mean': 4000,
#         'std: 2000,
#         'count': 34
#     },
#     
#     '2' : {
#         'mean': 4300,
#         'std: 2300,
#         'count': 22
#     },    
# }
# 

# In[34]:


def remove_bhk_outliers(df):
    exclude_indices = np.array([])
    for location, location_df in df.groupby('location'):
        bhk_stats = {}
        for bhk, bhk_df in location_df.groupby('bhk'):
            bhk_stats[bhk] = {
                'mean': np.mean(bhk_df.price_per_sqft),
                'std': np.std(bhk_df.price_per_sqft),
                'count': bhk_df.shape[0]
            }
        for bhk, bhk_df in location_df.groupby('bhk'):
            stats = bhk_stats.get(bhk-1)
            if stats and stats['count']>5:
                exclude_indices = np.append(exclude_indices, bhk_df[bhk_df.price_per_sqft<(stats['mean'])].index.values)
    return df.drop(exclude_indices,axis='index')
df8 = remove_bhk_outliers(df7)
# df8 = df7.copy()


# In[35]:


import matplotlib
matplotlib.rcParams["figure.figsize"] = (20,10)
plt.hist(df8.price_per_sqft,rwidth=0.8)
plt.xlabel("Price Per Square Feet")
plt.ylabel("Count")


# In[36]:


df8.bath.unique()


# In[37]:


plt.hist(df8.bath,rwidth=0.8)
plt.xlabel("Number of bathrooms")
plt.ylabel("Count")


# In[38]:


df8[df8.bath>10]


# In[39]:


df8[df8.bath>df8.bhk+2]


# Again the business manager has a conversation with you (i.e. a data scientist) that if you have 4 bedroom home and even if you have bathroom in all 4 rooms plus one guest bathroom, you will have total bath = total bed + 1 max. Anything above that is an outlier or a data error and can be removed
# 
# 

# In[40]:


df9 = df8[df8.bath<df8.bhk+2]
df9.shape


# In[41]:


df10 = df9.drop(['size','price_per_sqft'],axis='columns')
df10.head(3)


# Use One Hot Encoding For Location
# 

# In[42]:


dummies = pd.get_dummies(df10.location)
dummies.head(3)


# In[43]:


df11 = pd.concat([df10,dummies.drop('other',axis='columns')],axis='columns')
df11.head()


# In[44]:


df12 = df11.drop('location',axis='columns')
df12.head(2)


# In[45]:


X = df12.drop(['price'],axis='columns')
X.head(3)


# In[46]:


y = df12.price
y.head(3)


# In[47]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=10)


# In[48]:


from sklearn.linear_model import LinearRegression
lr_clf = LinearRegression()
lr_clf.fit(X_train,y_train)
lr_clf.score(X_test,y_test)


# In[ ]:




