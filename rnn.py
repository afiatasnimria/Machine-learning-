import kagglehub
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense, Dropout

path = kagglehub.dataset_download("sobhanmoosavi/us-accidents")
df = pd.read_csv(f"{path}/US_Accidents_March23.csv", nrows=200000)

df['Start_Time'] = pd.to_datetime(df['Start_Time'], errors='coerce')
df = df.dropna(subset=['Start_Time']).sort_values('Start_Time')

df_hourly = df.set_index('Start_Time').resample('H').size().reset_index()
df_hourly.columns = ['Time', 'Accident_Count']

scaler = MinMaxScaler(feature_range=(0, 1))
scaled_counts = scaler.fit_transform(df_hourly['Accident_Count'].values.reshape(-1, 1))

def create_sequences(data, window=24):
    X, y = [], []
    for i in range(len(data) - window):
        X.append(data[i:i+window, 0])
        y.append(data[i+window, 0])
    return np.array(X), np.array(y)

window_size = 24
X, y = create_sequences(scaled_counts, window_size)

X = np.reshape(X, (X.shape[0], X.shape[1], 1))

split = int(len(X) * 0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

model = Sequential([
    SimpleRNN(64, activation='tanh', input_shape=(window_size, 1), return_sequences=True),
    Dropout(0.2),
    SimpleRNN(32, activation='tanh'),
    Dense(1) # Predict the count for the next hour
])

model.compile(optimizer='adam', loss='mse')
model.fit(X_train, y_train, epochs=15, batch_size=32, verbose=1)

predictions = model.predict(X_test)
predicted_counts = scaler.inverse_transform(predictions)
actual_counts = scaler.inverse_transform(y_test.reshape(-1, 1))

print(f"Predicted accidents for next hour: {predicted_counts[0][0]:.2f}")
print(f"Actual accidents for that hour: {actual_counts[0][0]}")