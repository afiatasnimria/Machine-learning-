import kagglehub
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder

path = kagglehub.dataset_download("jessicali9530/animal-crossing-new-horizons-nookplaza-dataset")
df = pd.read_csv(f"{path}/villagers.csv")

def birthday_to_day(bday):
    try:
        return pd.to_datetime(bday, format='%d-%b').dayofyear
    except:
        return 0

df['Birthday_Day'] = df['Birthday'].apply(birthday_to_day)

le_personality = LabelEncoder()
le_species = LabelEncoder()

df['Personality_Enc'] = le_personality.fit_transform(df['Personality'])
df['Species_Enc'] = le_species.fit_transform(df['Species'])

features = ['Birthday_Day', 'Personality_Enc', 'Species_Enc']
X = df[features]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)

plt.figure(figsize=(8, 4))
plt.plot(range(1, 11), wcss, marker='o')
plt.title('Elbow Method for Optimal K')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.show()

kmeans = KMeans(n_clusters=4, init='k-means++', random_state=42)
df['Cluster'] = kmeans.fit_predict(X_scaled)

plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='Birthday_Day', y='Personality', hue='Cluster', palette='viridis', s=100)
plt.title('Villager Clusters: Birthday vs Personality')
plt.show()

print(df[['Name', 'Personality', 'Species', 'Cluster']].head(10))