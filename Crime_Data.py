import pandas as pd 
import numpy as np 
import matplotlib.pylab as plt

crime = pd.read_csv("C:/Users/nidhchoudhary/Desktop/Assignment/Clustering/crime_data.csv")
#crime = crime.rename([0] = "seq" axis='columns') 
  


print(crime.head())

#input()

def norm_func(i):
	x = (i - i.min()) / (i.max() - i.min())
	return(x)

df_norm = norm_func(crime.iloc[:,1:])
print("After Normalization")
print(df_norm.head())


#input()
from scipy.cluster.hierarchy import linkage
import  scipy.cluster.hierarchy as sch

z = linkage(df_norm,method = "complete", metric = "euclidean")
plt.figure(figsize=(15 ,5));
plt.title('Hierarchial Clustering')
plt.xlabel('Index')
plt.ylabel('Distance')
sch.dendrogram(z,leaf_rotation = 0., leaf_font_size =5.)
plt.show()

from sklearn.cluster import AgglomerativeClustering as ag 
h_complete = ag(n_clusters=2,linkage = 'complete',affinity = "euclidean").fit(df_norm)
h_complete_labels = pd.Series(h_complete.labels_)
crime['clust_levels'] = h_complete_labels
crime = crime.iloc[:,[5,0,1,2,3,4]]

print(crime.head())

#crime.to_csv("CrimeDataAssignment.csv",encoding= "utf-8")
