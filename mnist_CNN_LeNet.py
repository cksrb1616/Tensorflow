import tensorflow as tf
import pandas as pd

(ind, dep), _ = tf.keras.datasets.cifar10.load_data()
ind = ind.reshape(60000, 28, 28, 1)
dep = pd.get_dummies(dep)
print(ind.shape, dep.shape)

X = tf.keras.layers.Input(shape=[28, 28, 1])

H = tf.keras.layers.Conv2D(6, kernel_size=5,padding='same', activation='swish')(X)
# padding ='same' convolution의 결과인 특징맵의 사이즈가 입력이미지와 동일한 크기로 출력
# 입력값이 28*28이니 그대로 특징맵을 출력하기 위해
# mnist 는 28*28 사용하였고 LeNet은 32*32 사용하였음
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
# 모델을 학습하고
model.fit(ind, dep, epochs=10)

###########################
# 모델을 이용합니다. 
pred = model.predict(ind[0:5])
pd.DataFrame(pred).round(2)

# 정답 확인
dep[0:5]

# 모델 확인
model.summary()