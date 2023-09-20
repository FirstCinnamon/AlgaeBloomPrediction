import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')


# 1. 데이터 생성
x = np.linspace(0, 100, 1000) # 0부터 100까지 균일하게 나누는 배열 생성
y = np.sin(x) # x의 사인 값을 계산

input_window = 50 # 입력 윈도우 크기 설정
output_window = 5 # 출력 윈도우 크기 설정

inputs = [] # 입력 데이터 저장 리스트 초기화
outputs = [] # 출력 데이터 저장 리스트 초기화

# 입력과 출력 데이터 추출해서 리스트에 추가
for i in range(len(y) - input_window - output_window):
    inputs.append(y[i:i+input_window])
    outputs.append(y[i+input_window:i+input_window+output_window])

# inputs와 outputs 배열을 numpy 배열로 변환 후 적당한 형태로 재구성
inputs = np.array(inputs).reshape(-1, input_window, 1)
outputs = np.array(outputs).reshape(-1, output_window)

# Input Layer: 모델의 입력 정의
input_layer = keras.layers.Input(shape=(input_window, 1))

# Convolutional Layer
conv_layer = keras.layers.Conv1D(filters=64,
                                 kernel_size=5,
                                 strides=1,
                                 padding="causal",
                                 activation="relu")(input_layer)

# Multi-Head Attention Layer: 멀티 헤드 어텐션 연산
multi_head_attention_layer= keras.layers.MultiHeadAttention(num_heads=4,
                                                           key_dim=64)(conv_layer,
                                                                        conv_layer,
                                                                        conv_layer)

# Flatten Layer: 다차원 텐서를 일차원으로 flatten
flatten_layer= keras.layers.Flatten()(multi_head_attention_layer)

# Output layer (Dense): fully-connected 레이어 정의 및 출력 윈도우 크기만큼 뉴런 설정
output_dense_layer=keras.layers.Dense(output_window)(flatten_layer)

# 위에서 정의한 레이어들로 모델 구성
model = keras.Model(input_layer,output_dense_layer )

# 모델 컴파일. 손실 함수는 MSE, 최적화 알고리즘은 Adam 사용
model.compile(loss="mse", optimizer="adam")

# 모델 학습. 에포크 수는 10으로 설정.
model.fit(inputs, outputs, epochs=10)

# 예측값 얻기 위해 학습된 모델에 새로운 입력 데이터 넣음.
sample_input=y[-input_window-5:-5].reshape(1,input_window ,1)
predicted_output=model.predict(sample_input )

print(predicted_output) # 예측값 출력

plt.figure(figsize=(12,6))
plt.plot(np.arange(input_window), sample_input.flatten(), 'b-', label='Input sequence')
plt.plot(np.arange(input_window,input_window + output_window), predicted_output.flatten(), 'r-', label='Predicted sequence')
plt.plot(np.arange(input_window,input_window + output_window), y[-output_window:], 'g-', label='Actual sequence')
plt.legend()
plt.show()
