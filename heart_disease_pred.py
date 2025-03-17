import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

data_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
# ? -> NaN
df = pd.read_csv(data_url, header=None, na_values='?')

df.columns = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal", "target"]

df.dropna(inplace=True)

df["sex"] = df["sex"].astype(int)
df["cp"] = df["cp"].astype(int)
df["fbs"] = df["fbs"].astype(int)
df["restecg"] = df["restecg"].astype(int)
df["exang"] = df["exang"].astype(int)
df["slope"] = df["slope"].astype(int)
df["ca"] = df["ca"].astype(int)
df["thal"] = df["thal"].astype(int)

# 1,2,3,4 -> 1 (effectively converting the target into a binary classification problem)
df["target"] = df["target"].apply(lambda x: 1 if x > 0 else 0)


X = df.drop(columns=['target']).values  # X contains all input features except the target.
y = df['target'].values

# 80% to train, 20% to test
#random_state value chosen based on chat_gpt suggestions
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalizing feature values (mu = 0, sigma^2 = 1) as helps the neural network train faster
scaler = StandardScaler() # normalize the feature values
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


model = keras.Sequential([
    keras.layers.Dense(16, activation='relu', input_shape=(X_train.shape[1],)),
    keras.layers.Dense(8, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')  # Binary classification
])

#Adaptive Moment Estimation
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


model.fit(X_train, y_train, epochs=20, batch_size=16, validation_data=(X_test, y_test))

loss, accuracy = model.evaluate(X_test, y_test)
print(f'Test Accuracy: {accuracy:.4f}')
