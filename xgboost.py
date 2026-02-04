import kagglehub
import os
import cv2
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

path = kagglehub.dataset_download("paultimothymooney/breast-histopathology-images")

def extract_histology_features(img_path):
    img = cv2.imread(img_path)
    if img is None: return None
    img = cv2.resize(img, (50, 50))

    mean, std = cv2.meanStdDev(img)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()

    hist_features = []
    for i in range(3):
        hist = cv2.calcHist([img], [i], None, [8], [0, 256]).flatten()
        hist_features.extend(hist)

    return np.hstack([mean.flatten(), std.flatten(), laplacian_var, hist_features])

data, labels = [], []
count = 0
max_samples = 2000

for root, dirs, files in os.walk(path):
    for file in files:
        if file.endswith('.png') and count < max_samples:
            label = int(os.path.basename(root))
            img_path = os.path.join(root, file)
            feats = extract_histology_features(img_path)
            if feats is not None:
                data.append(feats)
                labels.append(label)
                count += 1

X = np.array(data)
y = np.array(labels)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = xgb.XGBClassifier(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    objective='binary:logistic',
    tree_method='hist', 
    random_state=42
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.2%}")
print("\nClassification Report:\n", classification_report(y_test, y_pred))