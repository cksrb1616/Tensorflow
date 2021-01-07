# 라이브러리 로딩
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

###########################
# 이미지 읽어서 데이터 준비하기
paths = glob.glob('./notMNIST_small/*/*.png') # 정확해야 함. #개 고양이면 notMNIST_small 대신 catsandogs 등으로 적고 하면 됨
# 다양한 확장자를 위해서는 공부 필요
paths = np.random.permutation(paths) # 경로들을 가져와서 랜덤하게 셔플
ind = np.array([plt.imread(paths[i]) for i in range(len(paths))]) # 이미지를 읽어들임 하나씩 (28,28) 형
dep = np.array([paths[i].split('/')[-2] for i in range(len(paths))])태 # 정답을 가져와서 그 값을 채움.
print(ind.shape, dep.shape)
# (18724, 28, 28), (18274,)
# 흑백이미지
# 칼라였다면 (18274, 28, 28, 3)
# 이미지의 크기는 제각각일 것이기 때문에 이미지 도구로 resize 하면 사용가능해짐

# dep[0:10]
# plt.imshow(ind[0], cmap='gray')
# ind인 이미지와 dep인 결과 알파벳 값이 맞는지 확인위한 코드

ind = ind.reshape(18724, 28, 28, 1) # 만들었던 모델에 적용시키기 위해 4차원 형태로 변형시켜 이미지 한창은 3차원으로 변형시킨다.
dep = pd.get_dummies(dep)  # OneHot Encoding
print(ind.shape, dep.shape)

##########################
# 모델을 완성합니다.
X = tf.keras.layers.Input(shape=[28, 28, 1])

H = tf.keras.layers.Conv2D(6, kernel_size=5, padding='same', activation='swish')(X)
H = tf.keras.layers.MaxPool2D()(H)

H = tf.keras.layers.Conv2D(16, kernel_size=5, activation='swish')(H)
H = tf.keras.layers.MaxPool2D()(H)

H = tf.keras.layers.Flatten()(H)
H = tf.keras.layers.Dense(120, activation='swish')(H)
H = tf.keras.layers.Dense(84, activation='swish')(H)
Y = tf.keras.layers.Dense(10, activation='softmax')(H)

model = tf.keras.models.Model(X, Y)
model.compile(loss='categorical_crossentropy', metrics='accuracy')

###########################
# 모델을 학습
model.fit(ind, dep, epochs=10)

###########################
# 모델을 이용합니다.
pred = model.predict(ind[0:5])
pd.DataFrame(pred).round(2)

# 정답 확인
dep[0:5]

# 모델 확인
model.summary()
