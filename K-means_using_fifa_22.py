import pandas as pd
import numpy as np


players=pd.read_csv("players_22.csv")

features=["overall","potential","wage_eur","value_eur","age"]

players=players.dropna(subset=features)

data=players[features].copy()

data=((data-data.min())/(data.max()-data.min()))*9+1

def random_centroids(data,k):
    centroids=[]
    for i in range(k):
        centroid=data.apply(lambda x: float(x.sample()))
        centroids.append(centroid)
    return pd.concat(centroids,axis=1)
    

centroids=random_centroids(data,5)
def get_labels(data,centroids):
    distances=centroids.apply(lambda x: np.sqrt(((data-x)**2).sum(axis=1)))
    return distances.idxmin(axis=1)
labels=get_labels(data,centroids)

    



def new_centroids(data,labels,k):
    return data.groupby(labels).apply(lambda x: np.exp(np.log(x).mean())).T

from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from IPython.display import clear_output

def plot_clusters(data,labels,centroids,iteration):
    pca=PCA(n_components=2)
    data_2d=pca.fit_transform(data)
    centroids_2d=pca.transform(centroids.T)
    clear_output(wait=True)
    plt.title(f'Iteration num: {iteration}')
    plt.scatter(x=data_2d[:,0],y=data_2d[:,1],c=labels)
    plt.scatter(x=centroids_2d[:,0],y=centroids_2d[:,1])
    plt.show()

max_iterations=100
k=3

centroids=random_centroids(data,k)
old_centroids=pd.DataFrame()
iteration=1

while iteration<max_iterations and not centroids.equals(old_centroids):
    old_centroids=centroids
    
    labels=get_labels(data,centroids)
    centroids=new_centroids(data,labels,k)
    plot_clusters(data,labels,centroids,iteration)
    iteration+=1

print(centroids)
#later compared using sklearn
players[labels==2][["short_name"]+features]

#Checking with sklearn to see if we got correct values
from sklearn.cluster import KMeans

kmeans=KMeans(3)
kmeans.fit(data)

centroids=kmeans.cluster_centers_

pd.DataFrame(centroids,columns=features).T
