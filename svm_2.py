import kagglehub
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score

path = kagglehub.dataset_download("gregorut/videogamesales")
df = pd.read_csv(f"{path}/vgsales.csv")

df = df.dropna(subset=['Year', 'Publisher'])

df['Is_Hit'] = (df['Global_Sales'] > 1.0).astype(int)

le = LabelEncoder()
df['Platform_Enc'] = le.fit_transform(df['Platform'])
df['Genre_Enc'] = le.fit_transform(df['Genre'])

X = df[['Platform_Enc', 'Genre_Enc', 'Year']]
y = df['Is_Hit']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

svm_model = SVC(kernel='rbf', C=1.0, gamma='scale')
svm_model.fit(X_train, y_train)

y_pred = svm_model.predict(X_test)

print(f"Accuracy: {accuracy_score(y_test, y_pred):.2%}")
print("\nClassification Report:\n", classification_report(y_test, y_pred))