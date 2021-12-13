#!/usr/bin/env python
# coding: utf-8

# In[3]:


#Imports
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[4]:


df = pd.read_csv('Mall_Customers.csv')
df.head()


# In[5]:


df.info


# In[6]:


df.rename(index=str, columns={'Annual Income (k$)': 'Income',
                              'Spending Score (1-100)': 'Score'}, inplace=True)
df.head()


# In[7]:


dataset_select = df.replace({ 'Genre': {'Male':0 , 'Female':1}})
dataset_select


# In[8]:


dataset_select.drop(["CustomerID"], axis=1,inplace=True)


# In[52]:


corr = dataset_select.corr()
plt.figure(figsize=(20,20))
sns.heatmap(corr, cbar=True, square= True, annot=True, annot_kws={'size':20}, cmap='YlGnBu')


# In[ ]:


#From this we can see that age and Spending score are the most correlated, so these are the factors I will use to cluster customers


# In[10]:


dataset_select


# In[ ]:





# In[11]:


sns.displot(data = dataset_select, x = "Income",hue = 'Genre')


# In[12]:


plt.rcParams['figure.figsize'] = (18, 8)
plt.subplot(1,2,1)
sns.distplot(df['Income'])
plt.title('Distribution of Annual Income', fontsize = 20)
plt.xlabel('Range of Annual Income')
plt.ylabel('Count')

plt.subplot(1,2,2)
sns.distplot(df['Age'], color = 'red')
plt.title('Distribution of Age', fontsize = 20)
plt.xlabel('Range of Age')
plt.ylabel('Count')
plt.show()


# In[13]:


sns.countplot(df['Age'], palette = "Set3")
plt.title('Distribution of Age', fontsize =10)
plt.show()


# In[14]:


sns.pairplot(df)
plt.title('Pairplot for the Data', fontsize = 20)
plt.show()


# In[15]:


X = df.drop(['CustomerID', 'Genre'], axis = 1)


# In[28]:


from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


# In[17]:


kmeans_kwargs = {
     "init": "random",
     "n_init": 10,
     "max_iter": 300,
     "random_state": 42,
 }
 
 # A list holds the SSE values for each k
sse = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, **kmeans_kwargs)
    kmeans.fit(X)
    sse.append(kmeans.inertia_)


# In[66]:


#Choosing appropriate number of clusters


# In[67]:


#Elbow Method:


# In[18]:


plt.style.use("fivethirtyeight")
plt.plot(range(1, 11), sse)
plt.xticks(range(1, 11))
plt.xlabel("Number of Clusters")
plt.ylabel("SSE")
plt.show()


# In[72]:


#Possible elbow points at 3 and 5


# In[20]:


from kneed import KneeLocator
kl = KneeLocator(
     range(1, 11), sse, curve="convex", direction="decreasing"
 )


# In[22]:


kl.elbow


# In[23]:


#Silhouette score method:


# In[29]:


silhouette_coefficients = []
for k in range(2, 11):
    kmeans = KMeans(n_clusters=k, **kmeans_kwargs)
    kmeans.fit(X)
    score = silhouette_score(X, kmeans.labels_)
    silhouette_coefficients.append(score)


# In[30]:


plt.style.use("fivethirtyeight")
plt.plot(range(2, 11), silhouette_coefficients)
plt.xticks(range(2, 11))
plt.xlabel("Number of Clusters")
plt.ylabel("Silhouette Coefficient")
plt.show()


# In[31]:


#Evaluate visually which appears better: 4 or 6 clusters


# In[33]:


km4 = KMeans(n_clusters=4).fit(X)

X['Labels'] = km4.labels_
plt.figure(figsize=(12, 8))
sns.scatterplot(X['Income'], X['Score'], hue=X['Labels'], 
                palette=sns.color_palette('hls', 4))
plt.title('KMeans with 4 Clusters')
plt.show()


# In[34]:


km6 = KMeans(n_clusters=6).fit(X)

X['Labels'] = km6.labels_
plt.figure(figsize=(12, 8))
sns.scatterplot(X['Income'], X['Score'], hue=X['Labels'], 
                palette=sns.color_palette('hls', 6))
plt.title('KMeans with 4 Clusters')
plt.show()


# In[35]:


#Neither seems all that good so let's try out 5 clusters:


# In[37]:


km5 = KMeans(n_clusters=5).fit(X)

X['Labels'] = km5.labels_
plt.figure(figsize=(12, 8))
sns.scatterplot(X['Income'], X['Score'], hue=X['Labels'], 
                palette=sns.color_palette('hls', 5))
plt.title('KMeans with 5 Clusters')
plt.show()


# In[38]:


#It appears that 5 clusters is the best method


# In[39]:


#Now, let's try DBSCAN:


# In[40]:


from sklearn.cluster import DBSCAN


# In[48]:


db = DBSCAN(eps=11, min_samples = 6).fit(X)


# In[51]:


X['Labels'] = db.labels_
plt.figure(figsize=(12, 8))
sns.scatterplot(X['Income'], X['Score'], hue=X['Labels'], 
                palette=sns.color_palette('hls', np.unique(db.labels_).shape[0]))
plt.title('DBSCAN with epsilon 11, min samples 6')
plt.show()


# In[53]:


#From this we can see that DBSCAN might not be the best function for this situation, possibly due to the fact the data is not that dense


# In[58]:


km = KMeans(n_clusters=5, random_state=1)
new = df._get_numeric_data().dropna(axis=1)
km.fit(new)
predict=km.predict(new)


# In[60]:


df['Cluster'] = pd.Series(predict, index=df.index)


# In[64]:


avg_df = df.groupby(['Cluster'], as_index=False).mean()
avg_df


# In[68]:


plt.subplot(1,3,1)
sns.barplot(x='Cluster',y='Age',data=avg_df)

plt.subplot(1,3,2)
sns.barplot(x='Cluster',y='Spending Score (1-100)',data=avg_df)

plt.subplot(1,3,3)
sns.barplot(x='Cluster',y='Annual Income (k$)',data=avg_df)


# In[72]:


df2 = pd.DataFrame(df.groupby(['Cluster','Genre'])['Genre'].count())
df2


# Type 0:
# Middle age, medium income, medium spending score, more female
# 
# Type1:
# Young, high income, high spending, roughly equal male and female
# 
# Type2:
# Young, low income, high spending, more female
# 
# Type3:
# Middle age, high income, low spending, roughly equal male and female
# 
# Type4:
# Middle age, low income, low spending, more female
# 
