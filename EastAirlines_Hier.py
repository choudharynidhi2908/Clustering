import pandas as pd 
import numpy as np 
import matplotlib.pylab as plt

from scipy.cluster.hierarchy import linkage

import scipy.cluster.hierarchy as sch


airlines = pd.read_csv("C:/Users/nidhchoudhary/Desktop/Assignment/Clustering/EastWestAirlines.csv")

print(airlines.head())

def norm(i):
	x = i - i.min()/i.max()- i.min()
	return(x)


airlines_norm = norm(airlines.iloc[:,1:])

#linkage.help()

from sklearn.cluster import AgglomerativeClustering as ag

z = linkage(airlines_norm,method = "complete", metric = "euclidean")
plt.figure(figsize = (15,5))
sch.dendrogram(z,leaf_rotation =0.,leaf_font_size = 5.)
#plt.show()
h_complete = ag(n_clusters = 2,linkage ='complete',affinity = 'euclidean').fit(airlines_norm)
h_complete_labels = pd.Series(h_complete.labels_)
airlines_norm["Clust_Nos"] = h_complete_labels
print(airlines_norm.head())