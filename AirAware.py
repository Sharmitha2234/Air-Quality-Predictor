import os
import zipfile
import urllib.request
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from math import sqrt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00360/AirQualityUCI.zip"
ZIP_NAME = "AirQualityUCI.zip"
CSV_NAME = "AirQualityUCI.csv"

if not os.path.isfile(CSV_NAME):
    urllib.request.urlretrieve(URL, ZIP_NAME)
    with zipfile.ZipFile(ZIP_NAME, "r") as z:
        z.extractall()

df = pd.read_csv(CSV_NAME, sep=";", decimal=",")
df = df.loc[:, ~df.columns.str.contains("^Unnamed")]
for c in ["Date", "Time"]:
    if c in df.columns:
        df.drop(columns=c, inplace=True)
df.replace(-200, np.nan, inplace=True)
df.interpolate(method="linear", limit_direction="both", inplace=True)
df.fillna(method="ffill", inplace=True)
df.fillna(method="bfill", inplace=True)

scaler = MinMaxScaler()
scaled = scaler.fit_transform(df.values.astype(float))
target_name = "CO(GT)"
if target_name not in df.columns:
    raise ValueError("Target column missing")

target_idx = df.columns.get_loc(target_name)

def make_windows(data, lookback=30):
    X, y = [], []
    n = data.shape[0]
    for i in range(n - lookback):
        X.append(data[i : i + lookback])
        y.append(data[i + lookback, target_idx])
    return np.array(X), np.array(y)

LOOKBACK = 30
X, y = make_windows(scaled, LOOKBACK)
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

model = Sequential()
model.add(LSTM(128, input_shape=(X.shape[1], X.shape[2]), return_sequences=True))
model.add(Dropout(0.25))
model.add(LSTM(64))
model.add(Dropout(0.25))
model.add(Dense(1))
model.compile(optimizer="adam", loss="mse")

es = EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True)
history = model.fit(X_train, y_train, epochs=60, batch_size=32, validation_split=0.1, callbacks=[es], verbose=1)

y_pred_scaled = model.predict(X_test)

def unscale(preds, idx, scaler_obj, n_features):
    arr = np.zeros((len(preds), n_features))
    arr[:, idx] = preds.flatten()
    return scaler_obj.inverse_transform(arr)[:, idx]

y_pred = unscale(y_pred_scaled, target_idx, scaler, scaled.shape)
y_true = unscale(y_test.reshape(-1, 1), target_idx, scaler, scaled.shape)

rmse = sqrt(mean_squared_error(y_true, y_pred))
mae = mean_absolute_error(y_true, y_pred)
mask = y_true != 0
mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100 if mask.any() else np.nan
r2 = r2_score(y_true, y_pred)

print(f"RMSE: {rmse:.4f}")
print(f"MAE: {mae:.4f}")
print(f"MAPE: {mape:.2f}%")
print(f"R2: {r2:.4f}")

plt.figure(figsize=(12, 5))
plt.plot(y_true, label="Actual CO(GT)")
plt.plot(y_pred, label="Predicted CO(GT)")
plt.legend()
plt.xlabel("Samples")
plt.ylabel("CO (mg/m^3)")
plt.title("CO(GT) - Actual vs Predicted")
plt.grid(True)
plt.tight_layout()
plt.show()
