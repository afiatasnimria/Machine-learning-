import kagglehub
import os
import cv2
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

path = kagglehub.dataset_download("iarunava/cell-images-for-detecting-malaria")
base_dir = os.path.join(path, "cell_images")

def extract_morphological_features(img_path):
    img = cv2.imread(img_path)
    if img is None: return None
    img = cv2.resize(img, (64, 64))

    mean, std = cv2.meanStdDev(img)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    edge_density = np.sum(edges) / 255

    return np.hstack([mean.flatten(), std.flatten(), edge_density])

data, labels = [], []
categories = ['Parasitized', 'Uninfected']

for category in categories:
    folder = os.path.join(base_dir, category)
    for filename in os.listdir(folder)[:1000]:
        feats = extract_morphological_features(os.path.join(folder, filename))
        if feats is not None:
            data.append(feats)
            labels.append(category)

X = np.array(data)
y = np.array(labels)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rf_classifier = RandomForestClassifier(n_estimators=100, criterion='gini', random_state=42)
rf_classifier.fit(X_train, y_train)

y_pred = rf_classifier.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.2%}")
print("\nClassification Report:\n", classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=categories, yticklabels=categories, cmap='Blues')
plt.title('Confusion Matrix: Malaria Detection')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()