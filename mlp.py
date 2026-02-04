import kagglehub
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, accuracy_score

path = kagglehub.dataset_download("sobhanmoosavi/us-accidents")
df = pd.read_csv(f"{path}/US_Accidents_March23.csv", nrows=100000)

features = ['Temperature(F)', 'Humidity(%)', 'Pressure(in)', 'Visibility(mi)',
            'Wind_Speed(mph)', 'Precipitation(in)']
target = 'Severity'

df_clean = df[features + [target]].dropna()

X = df_clean[features]
y = df_clean[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

mlp = MLPClassifier(
    hidden_layer_sizes=(64, 32),
    activation='relu',
    solver='adam',
    max_iter=500,
    random_state=42,
    verbose=True 
)

mlp.fit(X_train, y_train)

predictions = mlp.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, predictions):.2%}")
print("\nClassification Report:\n", classification_report(y_test, predictions))