import kagglehub
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

path = kagglehub.dataset_download("START-UMD/gtd")
df = pd.read_csv(f"{path}/globalterrorismdb_0718dist.csv", encoding='ISO-8859-1', low_memory=False)

df_filtered = df[(df['iyear'] == 2017) & (df['region'] == 6)].copy()

df_filtered = df_filtered.dropna(subset=['latitude', 'longitude'])

X = df_filtered[['latitude', 'longitude']].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

dbscan = DBSCAN(eps=0.2, min_samples=10)
clusters = dbscan.fit_predict(X_scaled)

df_filtered['Cluster'] = clusters

plt.figure(figsize=(12, 8))

noise = df_filtered[df_filtered['Cluster'] == -1]
clustered = df_filtered[df_filtered['Cluster'] != -1]

plt.scatter(noise['longitude'], noise['latitude'], c='gray', s=5, label='Noise', alpha=0.3)
plt.scatter(clustered['longitude'], clustered['latitude'], c=clustered['Cluster'], cmap='tab20', s=20, label='Clusters')

plt.title('DBSCAN Spatial Clustering of Terrorist Events (South Asia 2017)')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.legend()
plt.show()

n_clusters = len(set(clusters)) - (1 if -1 in clusters else 0)
n_noise = list(clusters).count(-1)
print(f"Estimated number of clusters: {n_clusters}")
print(f"Estimated number of noise points: {n_noise}")