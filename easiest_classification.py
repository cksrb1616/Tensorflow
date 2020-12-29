# 라이브러리 사용
import tensorflow as tf
import pandas as pd

# 1.과거의 데이터를 준비합니다.
dir = 'https://raw.githubusercontent.com/blackdew/tensorflow1/master/csv/iris.csv'
iris = pd.read_csv(dir)

# 원핫인코딩
# dependent variable 이 범주형이기 때문에 이를 서로다른 컬럼을 만들어 해당할 때 1 아닐 때 0을 기입한 columns로 변환할 필요가 있음.
# 이를 원 핫 인코딩 이라고 함.
encoding = pd.get_dummies(iris) # 모든 범주형 데이터를 골라 원핫인코딩 된 결과를 만들어 줌.


ind = encoding[['꽃잎길이', '꽃잎폭', '꽃받침길이', '꽃받침폭']]
dep = encoding[['품종_setosa', '품종_versicolor', '품종_virginica']]
print(ind.shape, dep.shape)

# 2. 모델의 구조를 만듭니다
X = tf.keras.layers.Input(shape=[4]) # 4 independent variables
Y = tf.keras.layers.Dense(3, activation='softmax')(X) # 3 dependent variables.
model = tf.keras.models.Model(X, Y)
model.compile(loss='categorical_crossentropy',
              metrics='accuracy')
# Activation='softmax'
#   : sigmoid 와 softmax : 각각의 분류에 속할 확률로 예측하기 위해 사용된 방법
#   : default 로 identity (y=x) 라는 값이 회귀 모델에는 존재 했던 것
# loss = 'categorical_crossentropy'
# these two make model to classification
# metrics ='accuracy' loss 보다 사람이 보기 편한 지표

# 3.데이터로 모델을 학습(FIT)합니다.
model.fit(ind, dep, epochs=10000)

# 모델을 이용합니다.
# 맨 처음 데이터 5개
print(model.predict(ind[:5]))
print(dep[:5])

# 맨 마지막 데이터 5개
print(model.predict(ind[-5:]))
print(dep[-5:])

# weights & bias 출력
print(model.get_weights())

# 세로로 쭉 보고 맨 마지막에 어레이 값을 constant 로 더해주면 첫번째 카테고리에 속할 확률