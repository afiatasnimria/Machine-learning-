import kagglehub
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

path = kagglehub.dataset_download("jessicali9530/animal-crossing-new-horizons-nookplaza-dataset")
df = pd.read_csv(f"{path}/villagers.csv")

df['Birthday_Day'] = pd.to_datetime(df['Birthday'], format='%d-%b', errors='coerce').dt.dayofyear
df['Birthday_Day'] = df['Birthday_Day'].fillna(0)

df['Name_Length'] = df['Name'].apply(len)
X = df[['Birthday_Day', 'Name_Length']]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

best_k = 2
best_score = -1
scores = []

for k in range(2, 11):
    model = KMeans(n_clusters=k, init='k-means++', n_init=10, random_state=42)
    labels = model.fit_predict(X_scaled)
    score = silhouette_score(X_scaled, labels)
    scores.append(score)

    if score > best_score:
        best_score = score
        best_k = k

print(f"The mathematically optimal number of clusters (K) is: {best_k}")

final_model = KMeans(n_clusters=best_k, init='k-means++', random_state=42)
df['Cluster'] = final_model.fit_predict(X_scaled)

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(range(2, 11), scores, marker='s', color='green')
plt.title('Silhouette Scores (Higher is Better)')

plt.subplot(1, 2, 2)
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=df['Cluster'], cmap='plasma')
plt.title(f'Modified K-Means (K={best_k})')
plt.show()