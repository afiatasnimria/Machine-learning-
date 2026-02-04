import kagglehub
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report

!pip install pyGRNN
from pyGRNN import GRNN
path = kagglehub.dataset_download("uciml/iris")
df = pd.read_csv(f"{path}/Iris.csv")

X = df.drop(['Id', 'Species'], axis=1).values
y = df['Species'].values

le = LabelEncoder()
y = le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


grnn = GRNN(calibration="gradient_search")
grnn.fit(X_train, y_train)


y_pred_continuous = grnn.predict(X_test)
y_pred = np.round(y_pred_continuous).astype(int)


y_pred = np.clip(y_pred, 0, 2)

print(f"GRNN Iris Accuracy: {accuracy_score(y_test, y_pred):.2%}")
print("\nDetailed Report:\n", classification_report(y_test, y_pred, target_names=le.classes_))