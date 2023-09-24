import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('TkAgg')

# 1. 데이터 생성
x = np.linspace(0, 100, 1000)
y1 = np.sin(x)  # sin
y2 = np.cos(x)  # cos
y3 = np.tan(x)  # tan
y3 = np.clip(y3, -10, 10)  # tan 값이 무한대로 갈 수 있으므로 클리핑

input_window = 50
output_window = 5

inputs = []
outputs = []

# 입력과 출력 데이터 추출해서 리스트에 추가
for i in range(len(y1) - input_window - output_window):
    inputs.append(np.array([y1[i:i + input_window], y2[i:i + input_window]]).T)
    outputs.append(y3[i + input_window:i + input_window + output_window])

inputs = np.array(inputs)
outputs = np.array(outputs)

# Input Layer
input_layer = keras.layers.Input(shape=(input_window, 2))

# Convolutional Layer
conv_layer = keras.layers.Conv1D(filters=64,
                                 kernel_size=5,
                                 strides=1,
                                 padding="causal",
                                 activation="relu")(input_layer)

# Multi-Head Attention Layer
multi_head_attention_layer = keras.layers.MultiHeadAttention(num_heads=4, key_dim=64)(conv_layer, conv_layer, conv_layer)

# Flatten Layer
flatten_layer = keras.layers.Flatten()(multi_head_attention_layer)

# Output layer (Dense)
output_dense_layer = keras.layers.Dense(output_window)(flatten_layer)

# 모델 구성
model = keras.Model(input_layer, output_dense_layer)

# 모델 컴파일
model.compile(loss="mse", optimizer="adam")

# 모델 학습
model.fit(inputs, outputs, epochs=10)

# 예측값
sample_input = np.array([y1[-input_window - 5:-5], y2[-input_window - 5:-5]]).T.reshape(1, input_window, 2)
predicted_output = model.predict(sample_input)
actual_output = y3[-output_window:]

print(predicted_output)

plt.figure(figsize=(12, 6))
plt.plot(np.arange(input_window), sample_input[0, :, 0], 'b-', label='Input sin sequence')
plt.plot(np.arange(input_window), sample_input[0, :, 1], 'g-', label='Input cos sequence')
plt.plot(np.arange(input_window, input_window + output_window), predicted_output.flatten(), 'r-', label='Predicted tan sequence')
plt.plot(np.arange(input_window, input_window + output_window), actual_output, 'm--', label='Actual tan sequence')
plt.legend()
plt.show()

