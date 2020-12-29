import tensorflow as tf
import pandas as pd

dir = 'https://raw.githubusercontent.com/blackdew/tensorflow1/master/csv/lemonade.csv'
lemon = pd.read_csv(dir)
#lemon.head()

ind = lemon[['온도']]
dep = lemon[['판매량']]
print(ind.shape, dep.shape)

# 모델을 만듭니다.
X = tf.keras.layers.Input(shape=[1]) # 1 means number of independent variable (column)
Y = tf.keras.layers.Dense(1)(X) # 1 means number of dependent variable (column)
model = tf.keras.models.Model(X, Y)
model.compile(loss='mse') # model 이 학습할 방법을 정해줌

# 모델을 학습시킵니다. (Process of Fit)
#model.fit(ind, dep, epochs=1000, verbose=0) # epochs : how many times would it repeat the test
# verbose = 0 : epochs 값 화면 출력을 하지 않음
model.fit(ind, dep, epochs=1000) # 별도의 테스크로 실행시 Epoch 횟 수별 결과물이 출력 되는데 이때 loss: (예측 - 결과)^2:회귀 선과의 거리의 제곱

# 모델을 이용합니다.
print(model.predict(ind))
print(model.predict([[15]]))