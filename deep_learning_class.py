# 1.과거의 데이터를 준비합니다.
dir = 'https://raw.githubusercontent.com/blackdew/tensorflow1/master/csv/iris.csv'
iris = pd.read_csv(dir)

# 원핫인코딩
iris = pd.get_dummies(iris)

# dep변수, ind변수
ind = iris[['꽃잎길이', '꽃잎폭', '꽃받침길이', '꽃받침폭']]
dep = iris[['품종_setosa', '품종_versicolor', '품종_virginica']]
print(ind.shape, dep.shape)
# 2. 모델의 구조를 만듭니다
X = tf.keras.layers.Input(shape=[4])
H = tf.keras.layers.Dense(8, activation="swish")(X)
H = tf.keras.layers.Dense(8, activation="swish")(H)
H = tf.keras.layers.Dense(8, activation="swish")(H) # 히든 레이어가 3개인 모델 8대신 몇몇 히든은 숫자가 달라도 됨
Y = tf.keras.layers.Dense(3, activation='softmax')(H)
model = tf.keras.models.Model(X, Y)
model.compile(loss='categorical_crossentropy',
              metrics='accuracy')
# 모델 구조 확인
#model.summary()
# 3.데이터로 모델을 학습(FIT)합니다.
model.fit(ind, dep, epochs=100)
# 4. 모델을 이용합니다
print(model.predict(ind[0:5]))
print(dep[0:5])

model.fit(ind, dep, epochs=100)