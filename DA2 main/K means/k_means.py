import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans

print("21BBS0163 HARSHITH KUMAR")

file_path = 'diet_2.csv'  
data = pd.read_csv(file_path)

print(data.head())

label_encoders = {}
for column in ['Gender', 'Physical Activity Level', 'Dietary Habit', 'Strict Diet']:
    le = LabelEncoder()
    data[column] = le.fit_transform(data[column])
    label_encoders[column] = le

features = ['Age', 'Weight', 'Height', 'BMI',
            'Physical Activity Level', 'Dietary Habit', 'Strict Diet']
X = data[features]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

kmeans = KMeans(n_clusters=3, random_state=42)
data['Cluster'] = kmeans.fit_predict(X_scaled)

cluster_labels = {0: 'Healthy', 1: 'Normal', 2: 'Weak'}
data['Cluster Label'] = data['Cluster'].map(cluster_labels)

print("Cluster Centers:")
print(kmeans.cluster_centers_)

print(data[['Age', 'Weight', 'Height', 'BMI', 'Cluster Label']].head())

# Optional: Visualize the clusters
# plt.scatter(data['Age'], data['BMI'], c=data['Cluster'], cmap='viridis')
# plt.xlabel('Age')
# plt.ylabel('BMI')
# plt.title('Clusters')
# plt.colorbar(label='Cluster')
# plt.show()
