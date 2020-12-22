import tensorflow as tf

a = tf.constant(1) # in a format of tensorflow put 1 to a
b = tf.constant(2) # tensor 2 is put to b
c = tf.add(a,b)
# 세션은 하나의 흐름을 가능하게 하는 것

# https://eclipse360.tistory.com/40
# https://www.tensorflow.org/guide/effective_tf2?hl=ko
# Session & Placeholder
# No more need for two functions check the website

# Session                                       Place holder
# tensorflow 1                                  tensorflow 1
# sess = tf.Session()                           a = tf.placeholder(tf.float32)
# print(sess.run([a,b]))                        b = tf.placeholder(tf.float32)
#                                               adder_node = a + b
# tensorflow 2                                  print(sess.run(adder_node, feed_dict={a:3,b:4.5}))
# tf.print(a,b)
#                                               tensorflow 2
#                                               @tf.function
#                                               def adder(a,b):
#                                               return a + b
#                                               A = tf.constant(1)
#                                               B = tf.constant(2)
#                                               C = tf.constant([1,2])
#                                               D = tf.constant([3,4])
#                                               E = tf.constant([1,2,3],[4,5,6])
#                                               F = tf.constant([2,3,4],[5,6,7])
#                                               print(adder(A,B))
#                                               print(adder(C,D))
# W = tf.Variable(tf.ones(shape=(2,2)), name="W")
# b = tf.Variable(tf.zeros(shape=(2)), name="b")
#
# @tf.function
# def forward(x):
#   return W * x + b
# out_a = forward([1,0])
# print(out_a)

tf.print(a,b)
tf.print(c)

a = tf.Variable(5)
b = tf.Variable(3)
c = tf.multiply(a,b)
#init = tf.global_variables_initializer() # variables will be in 초기화 to be used in tensorflow # No need in tensorflow 2
# 텐서플로 1.x
# outputs = session.run(f(placeholder), feed_dict={placeholder: input})
# # 텐서플로 2.0
# outputs = f(input)

tf.print(c)