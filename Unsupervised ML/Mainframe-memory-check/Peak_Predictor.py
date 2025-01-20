import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

# Load the dataset from the specified file path
file_path = '/workspaces/ML/Unsupervised ML/Mainframe-memory-check/memory_usage_data.csv'
data = pd.read_csv(file_path)

# Data Preprocessing
data['Datetime'] = pd.to_datetime(data['Date'] + ' ' + data['Time'])
data.set_index('Datetime', inplace=True)
data.drop(['Date', 'Time'], axis=1, inplace=True)

# Handle missing values and normalize data
data.ffill(inplace=True)  # Updated to use ffill method

# Feature Extraction (example: mean and std deviation for CPU Time)
time_window = '10min'  # Updated to use '10min' instead of '1h' to ensure more samples
features = data.resample(time_window).agg({
    'CPU_Time_Sec': ['mean', 'std'],
    'Used_Real_Storage_MB': 'mean',
    'Used_Virtual_Storage_MB': 'mean',
    'Used_Auxiliary_Storage_GB': 'mean',
    'Used_Shared_Memory_Pages': 'mean',
    'Service_Units': 'mean'
})

# Flatten the multi-level columns resulting from aggregation
features.columns = ['_'.join(col).strip() for col in features.columns.values]

# Drop rows with NaN values resulting from the aggregation
features.dropna(inplace=True)

# Normalize the data
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

# Clustering
kmeans = KMeans(n_clusters=3)
clusters = kmeans.fit_predict(scaled_features)

# Add cluster labels to the dataset
features['Cluster'] = clusters

# Anomaly Detection
iso_forest = IsolationForest(contamination=0.05)
anomalies = iso_forest.fit_predict(scaled_features)
features['Anomaly'] = anomalies

# Train a prediction model (example: a simple regression model)
from sklearn.linear_model import LinearRegression
model = LinearRegression()
X = features.drop(columns=['Cluster', 'Anomaly'])
y = features['Cluster']
model.fit(X, y)

# Predict future CPU peaks (for illustration purposes, using the same dataset)
# Resample and aggregate future data similarly to the training data
future_features = data.resample(time_window).agg({
    'CPU_Time_Sec': ['mean', 'std'],
    'Used_Real_Storage_MB': 'mean',
    'Used_Virtual_Storage_MB': 'mean',
    'Used_Auxiliary_Storage_GB': 'mean',
    'Used_Shared_Memory_Pages': 'mean',
    'Service_Units': 'mean'
})

# Flatten the multi-level columns in future_features
future_features.columns = ['_'.join(col).strip() for col in future_features.columns.values]

# Ensure the future_features have the same columns as the training features
future_features = future_features[X.columns]

# Drop rows with NaN values resulting from the aggregation
future_features.dropna(inplace=True)

# Normalize the future features
future_scaled_data = scaler.transform(future_features)
future_predictions = model.predict(future_scaled_data)

# GPU Allocation Strategy (example: a simple rule-based allocation)
def allocate_gpu(cpu_peak_prediction):
    threshold = 2  # Example threshold, adjust based on your criteria
    if cpu_peak_prediction > threshold:
        print("Allocate GPU resources")
    else:
        print("Deallocate GPU resources")

# Deployment and Monitoring (pseudo code)
# for demonstration purposes, iterating over the dataset
for index, row in future_features.iterrows():
    new_data = row.values.reshape(1, -1)
    new_data_scaled = scaler.transform(new_data)
    prediction = model.predict(new_data_scaled)
    allocate_gpu(prediction[0])

# Example output to check the result
print("Clusters:\n", features['Cluster'].value_counts())
print("Anomalies:\n", features['Anomaly'].value_counts())