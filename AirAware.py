import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.metrics import mean_squared_error, r2_score
from math import sqrt
import zipfile
import urllib.request
import os

# Step 1: Download and extract
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00360/AirQualityUCI.zip"
zip_path = "AirQualityUCI.zip"
csv_file = "AirQualityUCI.csv"

if not os.path.exists(csv_file):
    urllib.request.urlretrieve(url, zip_path)
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall()

# Step 2: Load and clean
df = pd.read_csv(csv_file, sep=';', decimal=',')
df = df.drop(columns=['Date', 'Time'], errors='ignore')
df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
df.replace(-200, np.nan, inplace=True)
df = df.ffill().bfill()

# Step 3: Normalize and select target (CO(GT))
scaler = MinMaxScaler()
scaled = scaler.fit_transform(df)
target_index = df.columns.get_loc('CO(GT)')

def create_sequences(data, steps=30):
    X, y = [], []
    for i in range(len(data) - steps):
        X.append(data[i:i + steps])
        y.append(data[i + steps][target_index])
    return np.array(X), np.array(y)

X, y = create_sequences(scaled, 30)
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# Step 4: Build LSTM model
model = Sequential([
    LSTM(100, activation='tanh', return_sequences=True, input_shape=(X.shape[1], X.shape[2])),
    Dropout(0.3),
    LSTM(50, activation='tanh'),
    Dropout(0.3),
    Dense(1)
])
model.compile(optimizer='adam', loss='mse')

# Step 5: Train longer for better results
model.fit(X_train, y_train, epochs=100, batch_size=32, verbose=1)

# Step 6: Predict & evaluate
y_pred = model.predict(X_test)
rmse = sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"RMSE: {rmse:.4f}")
print(f"R² Score: {r2:.4f}")
print(f"Accuracy-like Score: {r2 * 100:.2f}%")

# Step 7: Plot the graph
plt.figure(figsize=(10,5))
plt.plot(y_test, label='Actual' , color= 'red')
plt.plot(y_pred, label='Predicted', color='blue')
plt.title("Air Quality Forecasting (CO Level)")
plt.legend()
plt.show()
