!pip install scikit-fuzzy

import kagglehub
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import skfuzzy as fuzzy
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

path = kagglehub.dataset_download("muratkokludataset/pistachio-image-dataset")
data_dir = os.path.join(path, "Pistachio_Image_Dataset", "Pistachio_Image_Dataset")

images = []
categories = ['Kirmizi_Pistachio', 'Siirt_Pistachio']
for category in categories:
    folder = os.path.join(data_dir, category)
    for f in os.listdir(folder)[:200]:
        img = cv2.imread(os.path.join(folder, f), cv2.IMREAD_GRAYSCALE)
        if img is not None:
            img = cv2.resize(img, (50, 50))
            images.append(img.flatten())

X = np.array(images)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

data_to_cluster = X_pca.T

cntr, u, u0, d, jm, p, fpc = fuzzy.cluster.cmeans(
    data_to_cluster, c=2, m=2, error=0.005, maxiter=1000, init=None
)

cluster_membership = np.argmax(u, axis=0)

plt.figure(figsize=(10, 6))

for j in range(2):
    plt.scatter(X_pca[cluster_membership == j, 0],
                X_pca[cluster_membership == j, 1],
                alpha=0.5, label=f'Cluster {j}')

plt.scatter(cntr[:, 0], cntr[:, 1], marker='X', s=200, c='red', label='Centroids')

plt.title(f'Fuzzy C-Means on Pistachios (FPC Score: {fpc:.2f})')
plt.xlabel('PCA Feature 1')
plt.ylabel('PCA Feature 2')
plt.legend()
plt.show()

print("Membership degrees for first 5 samples (Cluster 0, Cluster 1):")
print(u[:, :5].T)