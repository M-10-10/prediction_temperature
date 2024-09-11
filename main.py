import sklearn
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd
import keras
import matplotlib.pyplot as plt
import tensorflow as tf

tf.random.set_seed(99)

# Dataset loading
dataFrame11 = pd.read_csv('D:/jena_climate_2009_2016.csv')
dataFrame1=dataFrame11[125120:126128]
pr=['Date Time','T (degC)']
print(dataFrame1[pr])

features_considered = ['T (degC)']
dataFrame=dataFrame1[features_considered]
print(dataFrame)

fig, ax = plt.subplots(figsize=(10, 6))
#  highlight the  forecast


# Plot the actual values
plt.plot(dataFrame[['T (degC)']], label=['Non-norm'])
plt.show()
# Applying feature scaling
scaler = MinMaxScaler(feature_range=(0, 1))
df_scaled = scaler.fit_transform(dataFrame.to_numpy())
df_scaled = pd.DataFrame(df_scaled, columns=list(dataFrame.columns))
target_scaler = MinMaxScaler(feature_range=(0, 1))
df_scaled[['T (degC)']] = target_scaler.fit_transform(dataFrame[['T (degC)']].to_numpy())
df_scaled = df_scaled.astype(float)
plt.plot(df_scaled,  'm-')

# Single step dataset preparation

def univariate_data(dataset, start_index, end_index, history_size, target_size):
  data = []
  labels = []

  start_index = start_index + history_size
  if end_index is None:
    end_index = len(dataset) - target_size

  for i in range(start_index, end_index):
    indices = range(i-history_size, i)
    # Reshape data from (history_size,) to (history_size, 1)
    data.append(np.reshape(dataset[indices], (history_size, 1)))
    labels.append(dataset[i+target_size])
  return np.array(data), np.array(labels)

SPLIT = 0.85

TRAIN_SPLIT = int(len(df_scaled)*SPLIT)
uni_data = df_scaled.values
print("uni_data[:TRAIN_SPLIT]", len(uni_data[:TRAIN_SPLIT]))
uni_train_mean = uni_data[:TRAIN_SPLIT].mean()
uni_train_std = uni_data[:TRAIN_SPLIT].std()
uni_data = (uni_data-uni_train_mean)/uni_train_std

plt.plot(uni_data, 'b-')
plt.show()


shp=len(uni_data[:,0])
window_size = shp // 70
denoised_data: np.ndarray = (pd.Series(uni_data[:,0]).rolling(window=window_size).mean().iloc[window_size - 1 :].values)
fig, ax = plt.subplots(figsize=(10, 6))
plt.plot(uni_data, label=['Original'])
plt.plot(denoised_data, label=['Filtr'])
plt.show()

l=len(denoised_data)
denoised_data=np.reshape(denoised_data,(l,1))


univariate_past_history = 20
univariate_future_target = 0

x_train_uni, y_train_uni = univariate_data(uni_data, 0, TRAIN_SPLIT,univariate_past_history,univariate_future_target)
x_val_uni, y_val_uni = univariate_data(uni_data, TRAIN_SPLIT, None,univariate_past_history,univariate_future_target)
print('**********************************')
print(x_train_uni.shape)
print ('Single window of past history')
print (x_train_uni[0])
print ('\n Target temperature to predict')
print (y_train_uni[0])

multivariate_lstm = keras.Sequential()
multivariate_lstm.add(keras.layers.LSTM(400, input_shape=(x_train_uni.shape[1], x_train_uni.shape[2])))#200
multivariate_lstm.add(keras.layers.Dropout(0.2))
multivariate_lstm.add(keras.layers.Dense(2, activation='linear'))
multivariate_lstm.compile(loss = 'MeanSquaredError', metrics=['MAE'], optimizer='Adam')
multivariate_lstm.summary()


history = multivariate_lstm.fit(x_train_uni, y_train_uni, epochs=20)


# Reload the data with the date index

dataFrame = dataFrame1['T (degC)']
dataFrame.index = dataFrame1['Date Time']
print(dataFrame)

# Forecast Plot with Dates on X-axis
predicted_values = multivariate_lstm.predict(x_val_uni)
predicted_values = predicted_values.transpose()
y_val_uni = y_val_uni.transpose()

d = {
    'Predicted_Temp': predicted_values[0],
    'Actual_Temp': y_val_uni[0],
}


d = pd.DataFrame(d)

d.index = dataFrame.index[-len(y_val_uni[0]):]  # Assigning the correct date index

fig, ax = plt.subplots(figsize=(10, 6))
#  highlight the  forecast
highlight_start = int(len(d) * 0.9)
highlight_end = len(d) - 1  # Adjusted to stay within bounds

# Plot the actual values
plt.plot(d[['Actual_Temp']][:highlight_start], label=['Actual_Temp'])


# Plot predicted values with a dashed line
plt.plot(d[['Predicted_Temp']], label=['Predicted_Temp'], linestyle='--')

# Highlight the forecasted portion with a different color
plt.axvspan(d.index[highlight_start], d.index[highlight_end], facecolor='lightgreen', alpha=0.5, label='Forecast')

plt.title('Multivariate Time-Series forecasting using LSTM')
plt.xlabel('Dates')
plt.ylabel('Values')
ax.legend()
plt.show()

# Model Evaluation
def eval(model):
    return {
        'MSE': sklearn.metrics.mean_squared_error(d[f'Actual_{model.split("_")[1]}'].to_numpy(), d[model].to_numpy()),
        'MAE': sklearn.metrics.mean_absolute_error(d[f'Actual_{model.split("_")[1]}'].to_numpy(), d[model].to_numpy()),
        'R2': sklearn.metrics.r2_score(d[f'Actual_{model.split("_")[1]}'].to_numpy(), d[model].to_numpy())
    }


result = dict()

for item in ['Predicted_Temp']:
    result[item] = eval(item)

print(result)