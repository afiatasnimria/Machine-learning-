import kagglehub
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.preprocessing import StandardScaler

path = kagglehub.dataset_download("muratkokludataset/pistachio-image-dataset")
data_dir = os.path.join(path, "Pistachio_Image_Dataset", "Pistachio_Image_Dataset")

images = []
labels = []
categories = ['Kirmizi_Pistachio', 'Siirt_Pistachio']
sample_size = 75

for category in categories:
    folder = os.path.join(data_dir, category)
    filenames = os.listdir(folder)[:sample_size]

    for f in filenames:
        img_path = os.path.join(folder, f)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            img = cv2.resize(img, (64, 64))
            images.append(img.flatten())
            labels.append(category)

X = np.array(images)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

Z = linkage(X_scaled, method='ward')

plt.figure(figsize=(15, 8))
plt.title('Hierarchical Clustering Dendrogram - Pistachio Varieties')
plt.xlabel('Pistachio Samples')
plt.ylabel('Distance (Ward)')

dendrogram(
    Z,
    labels=labels,
    leaf_rotation=90.,
    leaf_font_size=8.,
    color_threshold=150
)

plt.axhline(y=150, color='r', linestyle='--')
plt.tight_layout()
plt.show()