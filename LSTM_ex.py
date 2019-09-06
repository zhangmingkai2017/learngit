import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 数据准备
FlowData = pd.read_csv("flow_data.csv")

# standardize numerical values
quant_features = ['rainfall','flow']
flow_data = FlowData.copy()

for each in quant_features:
    s = FlowData[each].copy()
    mean, std = s.mean(),  s.std()
    s = (s - mean)/std
    flow_data[each] = s


features, targets = flow_data[['flow', 'rainfall']], flow_data['flow']

# window_size
window_size = 6

def get_data(feature_arr, target_arr, window_size):
    for n in range(0, feature_arr.shape[0]-window_size):
        x = feature_arr[n:n+window_size]
        y = target_arr[n+window_size]
        yield x, y


#For LSTM [batch_size, Length, Features]
feature_time = []
target_time = []

for x, y in get_data(features.values, targets.values, window_size):
    feature_time.append(x)
    target_time.append(y)

feature_time = np.array(feature_time)
target_time = np.array(target_time)

# train test split
X_train, y_train = feature_time[:-100], target_time[:-100].reshape(-1, 1)
X_test, y_test = feature_time[-100:], target_time[-100:].reshape(-1, 1)

# keras
model = tf.keras.models.Sequential([
    tf.keras.layers.LSTM(64, return_sequences=True, input_shape=(
        X_train.shape[1], X_train.shape[2])),
    tf.keras.layers.LSTM(32),
    tf.keras.layers.Dense(16),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(1)
])
model.compile(loss='mean_squared_error', optimizer='adam')

res = model.fit(X_train, y_train, epochs=10, batch_size=128, verbose=2)

# evaluate
model.evaluate(X_test, y_test)
predictions = model.predict(X_test)

# 去标准化
YPred = std * predictions + mean
YObs = std * y_test + mean
# plot
fig, ax = plt.subplots(figsize=(16, 8))

ax.plot(YPred, label='Prediction')
ax.plot(YObs, label='Data')
ax.legend()
