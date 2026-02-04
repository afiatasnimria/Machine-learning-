import kagglehub
import os
import cv2
import numpy as np
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

path = kagglehub.dataset_download("paultimothymooney/breast-histopathology-images")

def extract_features(img_path):
    img = cv2.imread(img_path)
    if img is None: return None
    img = cv2.resize(img, (50, 50))

    mean, std = cv2.meanStdDev(img)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5).var()
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5).var()

    return np.hstack([mean.flatten(), std.flatten(), sobelx, sobely])

data, labels = [], []
max_samples = 2500
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

base_estimator = DecisionTreeClassifier(max_depth=1)
ada_model = AdaBoostClassifier(
    estimator=base_estimator,
    n_estimators=150,
    learning_rate=1.0,
    random_state=42
)

ada_model.fit(X_train, y_train)

y_pred = ada_model.predict(X_test)
print(f"AdaBoost Accuracy: {accuracy_score(y_test, y_pred):.2%}")
print("\nClassification Report:\n", classification_report(y_test, y_pred))