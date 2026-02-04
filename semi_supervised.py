import kagglehub
import pandas as pd
import numpy as np
from sklearn.semi_supervised import SelfTrainingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report

path = kagglehub.dataset_download("szamil/who-suicide-statistics")
df = pd.read_csv(f"{path}/who_suicide_statistics.csv")

df['suicides_no'] = df['suicides_no'].fillna(0)
df['population'] = df['population'].fillna(df['population'].median())

median_suicides = df['suicides_no'].median()
df['risk_level'] = (df['suicides_no'] > median_suicides).astype(int)

le = LabelEncoder()
df['country'] = le.fit_transform(df['country'])
df['year'] = le.fit_transform(df['year'])
df['sex'] = le.fit_transform(df['sex'])
df['age'] = le.fit_transform(df['age'])

X = df[['country', 'year', 'sex', 'age', 'population']]
y = df['risk_level']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rng = np.random.RandomState(42)
random_unlabeled_points = rng.rand(y_train.shape[0]) < 0.7
y_train_semi = np.copy(y_train)
y_train_semi[random_unlabeled_points] = -1

base_model = RandomForestClassifier(n_estimators=100, random_state=42)
self_training_model = SelfTrainingClassifier(base_model, threshold=0.75, max_iter=10)

self_training_model.fit(X_train, y_train_semi)

y_pred = self_training_model.predict(X_test)
print(f"Labeled samples used: {np.sum(y_train_semi != -1)}")
print(f"Unlabeled samples used: {np.sum(y_train_semi == -1)}")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.2%}")
print("\nClassification Report:\n", classification_report(y_test, y_pred))