import pandas as pd
import datetime
import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import scrolledtext
from tensorflow import keras
from keras.models import Sequential
from keras.optimizers import Adam
from keras import layers, regularizers
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load data
df = pd.read_csv('usd_to_inr_new.csv')
df = df[['Date', 'Value']]

# Convert string dates to datetime and handle missing dates
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)
df = df.reindex(pd.date_range(start=df.index.min(), end=df.index.max(), freq='D'), method='ffill')

# Compute technical indicators
df['SMA_7'] = df['Value'].rolling(window=7).mean()
df['EMA_7'] = df['Value'].ewm(span=7, adjust=False).mean()

def compute_RSI(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

df['RSI_14'] = compute_RSI(df['Value'])

# Fill NaN values from technical indicators
df.fillna(method='bfill', inplace=True)

# Convert data to time-windowed format
def df_to_windowed_df(dataframe, n):
    X, Y = [], []
    for i in range(len(dataframe) - n):
        X.append(dataframe.iloc[i:i+n].values)
        Y.append(dataframe.iloc[i+n]['Value'])
    return np.array(X), np.array(Y)

n = 3  # Window size
X, y = df_to_windowed_df(df[['Value', 'SMA_7', 'EMA_7', 'RSI_14']], n)

# Split data
q_80, q_95 = int(len(y) * .95), int(len(y) * .99)
X_train, y_train = X[:q_80], y[:q_80]
X_val, y_val = X[q_80:q_95], y[q_80:q_95]
X_test, y_test = X[q_95:], y[q_95:]

# Reshape X for LSTM input
X_train = X_train.reshape((-1, n, 4))
X_val = X_val.reshape((-1, n, 4))
X_test = X_test.reshape((-1, n, 4))

# Learning rate finder function
def find_learning_rate(model, X_train, y_train, min_lr=1e-5, max_lr=1e-2, epochs=5, batch_size=32):
    learning_rates = np.geomspace(min_lr, max_lr, num=epochs)
    losses = []
    for lr in learning_rates:
        model.compile(optimizer=Adam(learning_rate=lr), loss='mse')
        history = model.fit(X_train, y_train, epochs=1, batch_size=batch_size, verbose=0)
        losses.append(history.history['loss'][0])
    return learning_rates, losses

# Build model
model = Sequential([
    layers.Input((n, 4)),
    layers.LSTM(256, return_sequences=True, kernel_regularizer=regularizers.l2(0.01)),
    layers.BatchNormalization(),
    layers.LSTM(256, kernel_regularizer=regularizers.l2(0.01)),
    layers.BatchNormalization(),
    layers.Dense(256, activation='linear', kernel_regularizer=regularizers.l2(0.01)),
    layers.Dense(1)
])

# Run learning rate finder
learning_rates, losses = find_learning_rate(model, X_train, y_train)
optimal_lr = learning_rates[np.argmin(losses)]  # Select best learning rate

# Plot learning rate vs. loss curve
plt.plot(learning_rates, losses)
plt.xlabel('Learning Rate')
plt.ylabel('Loss')
plt.xscale('log')
plt.show()

# Compile model with optimal learning rate
model.compile(loss='mse', optimizer=Adam(learning_rate=optimal_lr), metrics=['mean_absolute_error'])

# Callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=25, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=0.00001)

# Train model
history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=200, callbacks=[early_stopping, reduce_lr])

# Plot training & validation loss
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Make predictions
train_predictions = model.predict(X_train).flatten()
val_predictions = model.predict(X_val).flatten()
test_predictions = model.predict(X_test).flatten()

# Evaluate model
mae = mean_absolute_error(y_test, test_predictions)
mse = mean_squared_error(y_test, test_predictions)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, test_predictions)

# Directional accuracy
directional_accuracy = np.mean((np.sign(test_predictions[1:] - test_predictions[:-1]) == 
                               np.sign(y_test[1:] - y_test[:-1])) * 100)

# Display results
print(f"Mean Absolute Error: {mae}")
print(f"Mean Squared Error: {mse}")
print(f"Root Mean Squared Error: {rmse}")
print(f"R-squared: {r2}")
print(f"Directional Accuracy: {directional_accuracy:.2f}%")

# Plot predictions
plt.plot(y_train, label='Train Observations')
plt.plot(train_predictions, label='Train Predictions')
plt.plot(range(len(y_train), len(y_train) + len(y_val)), y_val, label='Val Observations')
plt.plot(range(len(y_train), len(y_train) + len(y_val)), val_predictions, label='Val Predictions')
plt.plot(range(len(y_train) + len(y_val), len(y_train) + len(y_val) + len(y_test)), y_test, label='Test Observations')
plt.plot(range(len(y_train) + len(y_val), len(y_train) + len(y_val) + len(y_test)), test_predictions, label='Test Predictions')
plt.legend()
plt.show()

# Store predictions in a DataFrame
results_df = pd.DataFrame({'Actual': y_test, 'Predicted': test_predictions})

# Display results using Tkinter
root = tk.Tk()
root.title("Predictions Results")
text_area = scrolledtext.ScrolledText(root, width=60, height=20)
text_area.pack(padx=10, pady=10)
text_area.insert(tk.END, results_df.to_string(index=False))
root.mainloop()

model.save("usd_inr_lstm_model.h5")