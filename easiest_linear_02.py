import tensorflow as tf
import pandas as pd

dir = 'https://raw.githubusercontent.com/blackdew/tensorflow1/master/csv/boston.csv'
boston = pd.read_csv(dir)
print(boston.columns)
boston.head()

# 독립변수, 종속변수 분리
# 종속 변수가 2개가 되면 식이 두개가 필요하면 Perceptron 두개가 병렬로 연결된 모델이라 할 수 있음. 이 때는 shape =[12]
ind = boston[['crim', 'zn', 'indus', 'chas', 'nox', 'rm', 'age', 'dis', 'rad', 'tax',
            'ptratio', 'b', 'lstat']]
dep = boston[['medv']]
print(ind.shape, dep.shape)

# 2. 모델의 구조를 만듭니다
X = tf.keras.layers.Input(shape=[13])
Y = tf.keras.layers.Dense(1)(X)
model = tf.keras.models.Model(X, Y)
model.compile(loss='mse')

# 3.데이터로 모델을 학습(FIT)합니다.
model.fit(ind, dep, epochs=1000, verbose=0)
model.fit(ind, dep, epochs=10)

# 4. 모델을 이용합니다
print(model.predict(ind[5:10]))
# 종속변수 확인
print(dep[5:10])
# 모델의 수식 확인
print(model.get_weights()) # weight 가 coefficients