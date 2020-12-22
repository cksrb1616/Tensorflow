#import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

x_data = [1,2,3,4,5,6,7]
y_data = [25,55,75,110,128,155,180]

w = tf.Variable(tf.random.uniform([1],-100,100)) # from -100 to 100 random number will be w
b = tf.Variable(tf.random.uniform([1],-100,100))
X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

H = w*X + b
cost = tf.reduce_mean(tf.square(H-Y)) # reduce_mean : finding average
a = tf.Variable(0.01)
optimizer = tf.train.GradientDescentOptimizer(a) # GradientDescent algorithm 경사하강
train = optimizer.minimize(cost)
init = tf.global_variables_initializer() # reset the variable

sess = tf.Session()
sess.run(init) # 초기화
for i in range(5001):
    sess.run(train, feed_dict={X:x_data,Y:y_data})
    if i%500 == 0: # Once in 500 times, print
        print(i,sess.run(cost,feed_dict={X:x_data,Y:y_data}), sess.run(w), sess.run(b))

print(sess.run(H, feed_dict={X:[8]}))

