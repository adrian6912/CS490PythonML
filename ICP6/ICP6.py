import pandas as pd
from sklearn import preprocessing
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
from sklearn import metrics
import matplotlib.pyplot as plt
import numpy as np

# Import data
data = pd.read_csv('CC.csv')
# Convert categorical to integer
data = data.apply(LabelEncoder().fit_transform)
x = data
# x = data.drop("TENURE", axis=1)

# Preprocessing data
scaler = preprocessing.StandardScaler()
scaler.fit(x)
X_scaled_array = scaler.transform(x)
X_scaled = pd.DataFrame(X_scaled_array, columns=x.columns)

# Kmeans cluster
k_value = 3
km = KMeans(n_clusters=k_value)
km.fit(X_scaled)

# Silhouette Score
y_cluster_kmeans = km.predict(X_scaled)
score = metrics.silhouette_score(X_scaled, y_cluster_kmeans)
print(score)

# elbow method to know the number of clusters
wcss = []
for k in range(1, 11):
    km = KMeans(n_clusters=k)
    km.fit(X_scaled)
    wcss.append(km.inertia_)

plt.plot(range(1, 11), wcss)
plt.title('the elbow method')
plt.xlabel('Number of Clusters')
plt.ylabel('Wcss')
plt.show()




# PCA
x_scaler = scaler.transform(x)
pca = PCA(3)
x_pca = pca.fit_transform(x_scaler)
finaldf = df2 = pd.DataFrame(data=x_pca)
# finaldf = pd.concat([df2, data.TENURE], axis=1)
print(finaldf)

# Apply PCA
wcss = []
for k in range(1, 11):
    km = KMeans(n_clusters=k)
    km.fit(finaldf)
    wcss.append(km.inertia_)

plt.plot(range(1, 11), wcss)
plt.title('the elbow method')
plt.xlabel('Number of Clusters')
plt.ylabel('Wcss')
plt.show()

# Preprocessing data
scaler = preprocessing.StandardScaler()
scaler.fit(finaldf)
X_scaled_array = scaler.transform(finaldf)
X_scaled = pd.DataFrame(X_scaled_array, columns=finaldf.columns)

# Kmeans cluster
k_value = 4
km = KMeans(n_clusters=k_value)
km.fit(X_scaled)

# predict the cluster for each data point
y_cluster_kmeans = km.predict(X_scaled)
score = metrics.silhouette_score(X_scaled, y_cluster_kmeans)
print(score)
