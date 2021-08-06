# -*- coding: utf-8 -*-
"""
Created on Fri Apr 23 08:10:23 2021

@author: Amartya
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as sch
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import classification_report
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale

wine=pd.read_csv("F:\Softwares\Data Science Assignments\Python-Assignment\PCA//wine.csv")
wine.Type.value_counts()
#Excluding first column "Type" which basically clustered or labelled the data
data=wine.iloc[:,1:14]
dsc=scale(data) #Calculating the z-score

pca=PCA(n_components=13)
pc_val=pca.fit_transform(dsc)

# The amount of variance that each PCA explains is
var=pca.explained_variance_ratio_
var

# Cumulative variance
var1=np.cumsum(np.round(var,decimals=4)*100)
var1

# Variance plot for PCA components obtained 
plt.plot(var1,color="red")
plt.xlabel("Principal Components")
plt.ylabel("Accumulated Data Percentage")


#Formation of new data table with first three PCA
new=pd.DataFrame(pc_val)
new=new.iloc[:,0:3]

#Hierarchial Clustering
dendogram=sch.dendrogram(sch.linkage(new,method='ward',metric='euclidean'))
h_clus=AgglomerativeClustering(n_clusters=3,affinity='euclidean',linkage='ward').fit(new)
help(AgglomerativeClustering())
c=pd.Series(h_clus.labels_)
c.value_counts()
wine["H_Cluster"]=c
wine["H_Cluster"]=wine["H_Cluster"].map({0:1,1:2,2:3})
wine.H_Cluster.value_counts()


#KMeans Clustering
twss=[]

for i in range(1,11):
    kmeans=KMeans(n_clusters=i,random_state=0)
    kmeans.fit(new)
    twss.append(kmeans.inertia_)

plt.plot(range(1,11),twss)
plt.title("Scree Plot")
plt.xlabel("No. of CLusters")
plt.ylabel("TWSS")

kmeans=KMeans(n_clusters=3,random_state=0)
pred=kmeans.fit_predict(new)
y=pd.Series(pred)
wine["K_Cluster"]=y
wine["K_Cluster"]=wine["K_Cluster"].map({0:1,1:2,2:3})
wine.K_Cluster.value_counts()


print(classification_report(wine.H_Cluster, wine.K_Cluster))
print(classification_report(wine.H_Cluster, wine.Type))
print(classification_report(wine.K_Cluster,wine.Type))
