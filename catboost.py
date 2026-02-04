!pip install catboost
import kagglehub
import os
import cv2
import numpy as np
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

path = kagglehub.dataset_download("paultimothymooney/breast-histopathology-images")

def extract_features(img_path):
    img = cv2.imread(img_path)
    if img is None: return None
    img = cv2.resize(img, (50, 50))

    mean, std = cv2.meanStdDev(img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    edge_sum = np.sum(edges) / 255

    return np.hstack([mean.flatten(), std.flatten(), edge_sum])

data, labels = [], []
max_samples = 3000
count = 0

for root, dirs, files in os.walk(path):
    for file in files:
        if file.endswith('.png') and count < max_samples:
            label = int(os.path.basename(root))
            img_path = os.path.join(root, file)
            feats = extract_features(img_path)
            if feats is not None:
                data.append(feats)
                labels.append(label)
                count += 1

X = np.array(data)
y = np.array(labels)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = CatBoostClassifier(
    iterations=500,
    learning_rate=0.05,
    depth=6,
    loss_function='Logloss',
    verbose=100, 
    random_seed=42
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print(f"CatBoost Accuracy: {accuracy_score(y_test, y_pred):.2%}")
print("\nClassification Report:\n", classification_report(y_test, y_pred))