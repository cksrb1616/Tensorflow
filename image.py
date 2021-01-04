import tensorflow as tf

(mnist_x, mnist_y), _ = tf.keras.datasets.mnist.load_data() # _ : 받는 값을 사용하지 않겠다는 의미의 variable
print(mnist_x.shape,mnist_y.shape)

(cifar_x,cifar_y), _ = tf.keras.datasets.cifar10.load_data()
print(cifar_x.shape,cifar_y.shape)

import matplotlib.pyplot as plt
plt.imshow(mnist_x[0],cmap='gray') # 흑백그림일경우 흑백출력
# plt.imshow(mnist_x[0])
print(mnist_y[0]) # 그림의 숫자

plt.imshow(cifar_x[0],cmap='gray')
print(mnist_y[0])

x1 = np.array([1,2,3,4,5])
print(x1.shape)
print(mnist_y[0:5])
print(mnist_y[0:5].shape)

x2 = np.array([[1,2,3,4,5],[1,2,3,4,5],[1,2,3,4,5]])
print(x2.shape)

x3 = np.array([1],[2],[3],[4],[5])
print(x3.shape)
print(cifar_y[0:5])
print(cifar_y[0:5].shape)
