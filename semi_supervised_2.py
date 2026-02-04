import pandas as pd
import numpy as np
import os
import kagglehub
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

path = kagglehub.dataset_download("borismarjanovic/price-volume-data-for-all-us-stocks-etfs")
file_path = os.path.join(path, "Data", "Stocks", "aapl.us.txt")
df = pd.read_csv(file_path)

df['Return'] = df['Close'].pct_change()
df['MA_5'] = df['Close'].rolling(window=5).mean()
df['MA_20'] = df['Close'].rolling(window=20).mean()
df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int) # 1 if price goes UP tomorrow

df = df.dropna()
X = df[['Return', 'MA_5', 'MA_20']].values
y = df['Target'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_labeled, X_unlabeled, y_labeled, _ = train_test_split(X_train, y_train, train_size=0.1, random_state=42)

scaler = StandardScaler()
X_labeled = scaler.fit_transform(X_labeled)
X_unlabeled = scaler.transform(X_unlabeled)
X_test = scaler.transform(X_test)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_labeled, y_labeled)

probs = model.predict_proba(X_unlabeled)

high_confidence_mask = np.max(probs, axis=1) > 0.8
X_pseudo = X_unlabeled[high_confidence_mask]
y_pseudo = model.predict(X_unlabeled)[high_confidence_mask]

X_combined = np.vstack([X_labeled, X_pseudo])
y_combined = np.concatenate([y_labeled, y_pseudo])

model.fit(X_combined, y_combined)

y_pred = model.predict(X_test)
print(f"Final Semi-Supervised Accuracy: {accuracy_score(y_test, y_pred):.2%}")