# Placeholder : variable which contains studying data. Input "X".
# tf.placeholder(dtype, shape, name)
import tensorflow as tf
input = [1,2,3,4,5]

# x = tf.placeholder(dtype=tf.float32)
# y = x + 5
# sess = tf.Session()
# sess.run(y, feed_dict{x:input})

b = tf.constant(5)
@tf.function
def fun(x):
    return x + b
output = fun(input)
print(output)

# W = tf.Variable(tf.ones(shape=(2,2)), name="W")
# b = tf.Variable(tf.zeros(shape=(2)), name="b")
#
# @tf.function
# def forward(x):
#   return W * x + b
# out_a = forward([1,0])
# print(out_a)

# a = tf.placeholder(dtype=tf.float32)
# b = tf.placeholder(dtpye=tf.float32)
# y = (a+b)/2
# sess = tf.Session()
# sess.run(y,feed_dict={a:mathScore,b:engScore})

mathScore = [85,99,84,97,92]
engScore = [59,80,84,68,77]
c = tf.constant(2)
@tf.function
def forward(a, b):
  return (a+b)/c
out_a = forward(mathScore,engScore)
print(out_a)

