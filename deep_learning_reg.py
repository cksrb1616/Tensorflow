import tensorflow as tf
import pandas as pd

# 1.과거의 데이터를 준비합니다.
dir = 'https://raw.githubusercontent.com/blackdew/tensorflow1/master/csv/boston.csv'
boston = pd.read_csv(dir)

# 종속변수, 독립변수
ind = boston[['crim', 'zn', 'indus', 'chas', 'nox',
            'rm', 'age', 'dis', 'rad', 'tax',
            'ptratio', 'b', 'lstat']]
dep = boston[['medv']]
print(ind.shape, dep.shape)

# Deep Learning
# 쉽게 말하자면 함수관계에 있는 x와 y가 있지만 x로부터 y를 예측할 수 있는 모델이 없을 때 대안으로 쓸 수 있는 방법이다.
# 쉽게 이해할 수 있는 개념인 회귀분석의 상위호환격 방법이라고 생각하면 된다.

# 2. 모델의 구조를 만듭니다
X = tf.keras.layers.Input(shape=[13])
H = tf.keras.layers.Dense(10, activation='swish')(X) # activation ='swish'
Y = tf.keras.layers.Dense(1)(H) # 히든을 넣어야 함
model = tf.keras.models.Model(X, Y)
model.compile(loss='mse')

# 모델 구조 확인
#model.summary()

# 3.데이터로 모델을 학습(FIT)합니다.
model.fit(ind, dep, epochs=1000)
# 13개의 x 로 10개의 h를 만들면 bias(constant) 때문에 14 * 10 = 140개 파라미터
# 10개의 h 로 1개의 y를 구하면 11개의 파라미터

# 4. 모델을 이용합니다
print(model.predict(ind[:5]))
print(dep[:5])

