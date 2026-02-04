import kagglehub
import os
import cv2
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

path = kagglehub.dataset_download("iarunava/cell-images-for-detecting-malaria")
base_dir = os.path.join(path, "cell_images")

def extract_features(img_path):
    img = cv2.imread(img_path)
    if img is None:
        return None
    img = cv2.resize(img, (64, 64))

    mean_rgb = cv2.mean(img)[:3]

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    std_dev = np.std(gray)

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hue_hist = cv2.calcHist([hsv], [0], None, [8], [0, 180]).flatten()

    return np.hstack([mean_rgb, std_dev, hue_hist])

data = []
targets = []
categories = {'Parasitized': 1.0, 'Uninfected': 0.0}

for category, score in categories.items():
    folder = os.path.join(base_dir, category)
    for filename in os.listdir(folder)[:500]:
        features = extract_features(os.path.join(folder, filename))
        if features is not None:
            data.append(features)
            targets.append(score)

X = np.array(data)
y = np.array(targets)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rf_regressor = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
rf_regressor.fit(X_train, y_train)

predictions = rf_regressor.predict(X_test)
mse = mean_squared_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print(f"Mean Squared Error: {mse:.4f}")
print(f"R-squared Score: {r2:.4f}")

print(f"Predicted 'Infection Score' for sample 0: {predictions[0]:.2f} (Actual: {y_test[0]})")