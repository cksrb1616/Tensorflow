import tensorflow as tf
import pandas as pd

# X = tf.keras.layers.Input(shape=[28, 28])
# H = tf.keras.layers.Flatten()(H)
# H = tf.keras.layers.Dense(84, activation='swish')(H) # 84개의 히든레이어
# Y = tf.keras.layers.Dense(10, activation='softmax')(H) # 10개의 종속변수
# model = tf.keras.models.Model(X, Y)
# model.compile(loss='categorical_crossentropy', metrics='accuracy')
# model.summary()

# 84 * (784+1) = 65940
# 784 개의 가중치와 1개의 bias. 65940의 파라미터 서머리를 통해 볼 수 있는 숫자들
# 10 * (84+1)

# X = tf.keras.layers.Input(shape=[28, 28, 1]) # 3차원으로만 인풋을 받아들임
# H = tf.keras.layers.Conv2D(3, kernel_size=5, activation='swish')(X) # 3개의 필터셋 -> 3개의 특징맵: 6채널의 특징맵, 사이즈 5
# H = tf.keras.layers.Conv2D(6, kernel_size=5, activation='swish')(H) # 6개의 필터셋 -> 6개의 특징맵: 6채널의 특징맵, 사이즈 5
# H = tf.keras.layers.Flatten()(H)
# H = tf.keras.layers.Dense(84, activation='swish')(H) # 84개의 히든레이어
# Y = tf.keras.layers.Dense(10, activation='softmax')(H) # 10개의 종속변수
# model = tf.keras.models.Model(X, Y)
# model.compile(loss='categorical_crossentropy', metrics='accuracy')
# model.summary()

# (none, 20, 20, 6) : (20*20) 이미지가 6장 : 한 장에 400개의 숫자가 있고 그것이 6장 = 2400 개의 flatten data
# 84 * (2400+1) = 201684

##############################################################################################
# 데이터를 준비하고
(독립, 종속), _ = tf.keras.datasets.mnist.load_data()
독립 = 독립.reshape(60000, 28, 28, 1)
종속 = pd.get_dummies(종속)
print(독립.shape, 종속.shape)

X = tf.keras.layers.Input(shape=[28, 28, 1])
H = tf.keras.layers.Conv2D(3, kernel_size=5, activation='swish')(X)
H = tf.keras.layers.MaxPool2D()(H) ######
H = tf.keras.layers.Conv2D(6, kernel_size=5, activation='swish')(H)
H = tf.keras.layers.MaxPool2D()(H) ######
H = tf.keras.layers.Flatten()(H)
H = tf.keras.layers.Dense(84, activation='swish')(H)
Y = tf.keras.layers.Dense(10, activation='softmax')(H)
model = tf.keras.models.Model(X, Y)
model.compile(loss='categorical_crossentropy', metrics='accuracy')
model.summary()

# 24,24 -> 12, 12
# 8, 8 -> 4, 4
# maxpool이후 특징맵의 수가 반이 된다.
# maxpool 이용시 그냥 flatten 이용시보다 가중치 수가 적어질 수 도 있다. (패러미터(가중치) 또한 줄어드는 것) 20만개에서 8천개로
# 6*6의 특징맵 있다면, 2*2씩 9개의 구획으로 쪼개, 각 2*2에서 가장 큰 수만을 3*3 (반쪽 크기)의 표에 남겨둔다
# max 값만을 옮겨오기에 MaxPooling
# 특징맵에서 값이 크다 = 필터로 찾으려는 특징이 많이 나타난 부분

# 모델을 학습하고
model.fit(독립, 종속, epochs=10)

###########################
# 모델을 이용합니다.
pred = model.predict(독립[0:5])
pd.DataFrame(pred).round(2)

# 정답 확인
종속[0:5]