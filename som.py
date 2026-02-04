import kagglehub
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
!pip install minisom
from minisom import MiniSom # pip install minisom
from sklearn.preprocessing import MinMaxScaler

path = kagglehub.dataset_download("sobhanmoosavi/us-accidents")
df = pd.read_csv(f"{path}/US_Accidents_March23.csv", nrows=50000)

features = ['Temperature(F)', 'Humidity(%)', 'Pressure(in)', 'Visibility(mi)', 'Wind_Speed(mph)']
df_som = df[features + ['Severity']].dropna()

X = df_som[features].values
y = df_som['Severity'].values

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

som_grid_x, som_grid_y = 20, 20
som = MiniSom(som_grid_x, som_grid_y, len(features), sigma=1.0, learning_rate=0.5)

som.random_weights_init(X_scaled)
som.train_random(X_scaled, 2000, verbose=True)

plt.figure(figsize=(10, 8))
plt.pcolor(som.distance_map().T, cmap='bone_r')
plt.colorbar()

markers = ['o', 's', 'D', 'X'] 
colors = ['C0', 'C1', 'C2', 'C3']

for i, x in enumerate(X_scaled[:500]):
    w = som.winner(x)
    plt.plot(w[0] + 0.5, w[1] + 0.5, markers[int(y[i]-1)],
             markerfacecolor='None', markeredgecolor=colors[int(y[i]-1)],
             markersize=8, markeredgewidth=2)

plt.title('SOM U-Matrix: Weather Feature Clusters & Severity Overlay')
plt.show()