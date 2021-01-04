import tensorflow as tf
import pandas as pd

###########################
# # with reshape
# # 데이터를 준비하고
# (ind, dep), _ = tf.keras.datasets.mnist.load_data()
# ind = ind.reshape(60000, 784)
# dep = pd.get_dummies(dep)
# print(ind.shape, dep.shape)
# 
# # 모델을 만들고
# X = tf.keras.layers.Input(shape=[784])
# H = tf.keras.layers.Dense(84, activation='swish')(X) # 0~9 까지 구분하기위한  84개의 특징을 정해줘 기계야! 
# Y = tf.keras.layers.Dense(10, activation='softmax')(H) # https://m.blog.naver.com/PostView.nhn?blogId=wideeyed&logNo=221017173808&proxyReferer=https:%2F%2Fwww.google.com%2F
# model = tf.keras.models.Model(X, Y)
# model.compile(loss='categorical_crossentropy', metrics='accuracy')
# 
# # 모델을 학습하고
# model.fit(ind, dep, epochs=10)
#
# # 모델을 이용합니다.
# pred = model.predict(ind[0:5])
# print(pd.DataFrame(pred).round(2))
# print(dep[0:5])

###########################
# with flatten
# 데이터를 준비하고
(ind, dep), _ = tf.keras.datasets.mnist.load_data()
dep = pd.get_dummies(dep)
print(ind.shape, dep.shape)

# 모델을 만들고
X = tf.keras.layers.Input(shape=[28,28]) # 28*28 = 784 so, shape=[28,28]
H = tf.keras.layers.Flatten()(X) # Flatten에서 28,28을 784로 알아서 풀어줌
H = tf.keras.layers.Dense(84, activation='swish')(H) # reshape 때는 X
Y = tf.keras.layers.Dense(10, activation='softmax')(H)
model = tf.keras.models.Model(X, Y)
model.compile(loss='categorical_crossentropy', metrics='accuracy')

# 모델을 학습하고
model.fit(ind, dep, epochs=10)

# 모델을 이용합니다. 
pred = model.predict(ind[0:5])
print(pd.DataFrame(pred).round(2)) # 예쁘게 보여주는 방식
print(dep[0:5]) # 정답확인
