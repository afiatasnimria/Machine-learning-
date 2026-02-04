import kagglehub
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import hdbscan 
from sklearn.preprocessing import StandardScaler

path = kagglehub.dataset_download("START-UMD/gtd")
df = pd.read_csv(f"{path}/globalterrorismdb_0718dist.csv", encoding='ISO-8859-1', low_memory=False)

df_filtered = df[(df['iyear'] >= 2015) & (df['region'] == 10)].copy()
df_filtered = df_filtered.dropna(subset=['latitude', 'longitude'])

X = df_filtered[['latitude', 'longitude']].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

clusterer = hdbscan.HDBSCAN(min_cluster_size=15, min_samples=5, gen_min_span_tree=True)
cluster_labels = clusterer.fit_predict(X_scaled)

df_filtered['Cluster'] = cluster_labels

plt.figure(figsize=(12, 8))

noise = df_filtered[df_filtered['Cluster'] == -1]
clustered = df_filtered[df_filtered['Cluster'] != -1]

plt.scatter(noise['longitude'], noise['latitude'], c='lightgray', s=2, label='Isolated Events', alpha=0.5)
plt.scatter(clustered['longitude'], clustered['latitude'], c=clustered['Cluster'], cmap='Set1', s=15, alpha=0.7)

plt.title('HDBSCAN: Identifying Multi-Density Conflict Hotspots (MENA 2015+)')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.show()

print(f"Number of clusters found: {cluster_labels.max() + 1}")