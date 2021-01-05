# convolution
# 특정한 패턴 특징이 어디서 나타나는지 확인하는 도구
# 필터 하나는 이미지 하나를 만든다 가로 핉터, 세로 필터
# 표를 활용한 Flatten 보다 더 빠르게 정확도가 높음

# 1 필터셋은 3차원 형태의 가중치(W) 모음
# 2 필터셋 하나는 앞선 레이어의 결과인 특징맵 전체를 본다
# 3 필터셋 개수만큼 특징맵을 만들게 된다
# kernel_size = 5 : 각 필터는 (5,5) 사이즈
# 필터셋 하나의 모양
# (5,5, 칼라는 3 흑백은 1) (5,5, 칼라는 3 흑백은 1)
# 필터셋 전체의 모양
# (3, 5, 5, 3) (6, 5, 5, 앞선 채널의 수?)
# (28,28,1) 흑백그림 -> (5,5,1) *3 필터셋 -> (24,24,3) 특징맵 -> (5,5,3) *6 필터셋 -> (20,20,6) 의 특징
# 28 -> 28 - (size-1) = 24맵
# 컴퓨터는 특징맵을 만들어 내는 convolution filter 를 학습

import tensorflow as tf
import pandas as pd

###########################
# 데이터를 준비하고
(독립, 종속), _ = tf.keras.datasets.mnist.load_data()
독립 = 독립.reshape(60000, 28, 28, 1) # 3차원으로만 인풋을 받아들임
종속 = pd.get_dummies(종속)
print(독립.shape, 종속.shape)

###########################
# 모델을 만들고
X = tf.keras.layers.Input(shape=[28, 28, 1]) # 3차원으로만 인풋을 받아들임

# 기존과 다른 점
# 1. 필터 셋을 몇개를 사용할 것인가 2. 필터셋의 사이즈를 얼마로 할 것인가
H = tf.keras.layers.Conv2D(3, kernel_size=5, activation='swish')(X) # 3개의 필터셋 -> 3개의 특징맵: 6채널의 특징맵, 사이즈 5
H = tf.keras.layers.Conv2D(6, kernel_size=5, activation='swish')(H) # 6개의 필터셋 -> 6개의 특징맵: 6채널의 특징맵, 사이즈 5
# 6채널의 특징맵을 flatten layer에 의해 픽셀단위로 한 줄로 펼친 후 학습
H = tf.keras.layers.Flatten()(H)
H = tf.keras.layers.Dense(84, activation='swish')(H) # 84개의 히든레이어
Y = tf.keras.layers.Dense(10, activation='softmax')(H) # 10개의 종속변수
model = tf.keras.models.Model(X, Y)
model.compile(loss='categorical_crossentropy', metrics='accuracy')

###########################
# 모델을 학습하고
model.fit(독립, 종속, epochs=10)
# 표를 활용한 Flatten 보다 더 정확도가 높음 parameter 가 늘어남에 따라 느려지긴 함 

###########################
# 모델을 이용합니다.
pred = model.predict(독립[0:5])
pd.DataFrame(pred).round(2)

# 정답 확인
종속[0:5]

# 모델 확인
model.summary()
