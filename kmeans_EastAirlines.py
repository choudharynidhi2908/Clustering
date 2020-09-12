import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster  import KMeans
from scipy.spatial.distance import cdist

import numpy as np 


airlines = pd.read_csv("C:/Users/nidhchoudhary/Desktop/Assignment/Clustering/EastWestAirlines.csv")

def norm(i):
	x = i - i.min()/i.max() - i.min()
	return(x)

airlines = norm(airlines.iloc[:,1:])

print(airlines.head())

k = list(range(5,100))
TWSS=[]

for i in k:
    kmeans = KMeans(n_clusters=i).fit(airlines)
    WSS=[]
    for j in range(i):
        WSS.append(sum(cdist(airlines.iloc[kmeans.labels_==j,:],kmeans.cluster_centers_[j].reshape(1,airlines.shape[1]),"euclidean")))
    TWSS.append(sum(WSS))
 ###ELBOW CURV
   #plt.plot(k,TWSS,'ro-');
   #plt.xlabel('K values');
   #plt.ylabel('Total Within Sum');
   #plt.xticks(k)
   #plt.show()


# # # Scree plot 
plt.plot(k,TWSS, 'ro-');plt.xlabel('K values');plt.ylabel('Total Within Sum');plt.xticks(k);plt.show()


##selecting 13 clusters from plot
model = KMeans(n_clusters=13).fit(airlines)
clust_labels = pd.Series(model.labels_)

airlines["Clust_Nos"] = clust_labels

print(airlines.head())

print(airlines.iloc[:,[-1]])


mean_values = print(airlines.iloc[:,1:-2].groupby(airlines.Clust_Nos).mean())





