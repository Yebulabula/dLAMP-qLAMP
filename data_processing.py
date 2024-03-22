import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt

# Load your dataset
curves = pd.read_csv('data/20230404/processed_data/inlier_df_ac.csv')


# Split the dataset into data and target
data = curves.iloc[:, -40:].to_numpy()
data = data / np.max(data, axis = 1)[:, None]
target = curves['Target'].to_numpy()

X_train, X_test, y_train, y_test = [], [], [], []
for i, d in enumerate(data):
    panel_id = int(curves['Channel'][i].split('panel')[1])
    if panel_id % 3 == 0:
        X_test.append(data[i])
        y_test.append(target[i])
    else:
        X_train.append(data[i])
        # plt.plot(data[i])
        # plt.show()
        y_train.append(target[i])
        
X_train, X_test, y_train, y_test = np.array(X_train), np.array(X_test), np.array(y_train), np.array(y_test)

# Split the dataset into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.3, random_state=42)

# It's often a good practice to scale the data for neural network models
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize the MLPClassifier
mlp = MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, activation='relu', solver='adam', random_state=1)

# Train the model
mlp.fit(X_train_scaled, y_train)

# Predict the test set results
y_pred = mlp.predict(X_test_scaled)

# Print classification report and confusion matrix
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
