import pandas as pd
import numpy as np 
import matplotlib.pyplot as py
from scipy.spatial.distance import cdist 

from	sklearn.cluster	import	KMeans


crime = pd.read_csv("C:/Users/nidhchoudhary/Desktop/Assignment/Clustering/crime_data.csv")

def norm_func(i):
	x= i-i.min()/i.max() - i.min()
	return(x)

df_norm = norm_func(crime.iloc[:,1:])

#print(df_norm.head())

k = list(range(2,25))
TWSS = []
for i in k:
    kmeans = KMeans(n_clusters = i).fit(df_norm)
    WSS=[]
    for j in range(i):
        WSS.append(sum(cdist(df_norm.iloc[kmeans.labels_==j,:],kmeans.cluster_centers_[j].reshape(1,df_norm.shape[1]),"euclidean")))
    TWSS.append(sum(WSS))

print(TWSS)

print(len(TWSS))




py.plot(k,TWSS,'ro-');py.xlabel("Nos of Clusters");py.ylabel("Within Sum Distance");py.xticks(k)
py.show()

##Selecting 5 clusters from the scree Plot as it has optimum number

model = KMeans(n_clusters = 5).fit(df_norm)
clus = pd.Series(model.labels_)


crime["Cluster_Nos"] = clus
#print(crime.head())

crime = crime.iloc[:,[5,0,1,2,3,4]]

#print(crime.head())

print(crime.iloc[:,1:5].groupby(crime.Cluster_Nos).mean())

#crime.to_csv("crime.csv")

