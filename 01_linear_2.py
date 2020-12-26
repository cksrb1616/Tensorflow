import tensorflow as tf

x_data = [1,2,3,4,5,6,7]
y_data = [25,55,75,110,128,155,180]

w = tf.Variable(tf.random.uniform([1],-100,100)) # from -100 to 100 random number will be w
b = tf.Variable(tf.random.uniform([1],-100,100))

@tf.function
def linear(x):
    return w*x + b

###########################################################
H = linear(x_data)

cost = tf.reduce_mean(tf.square(H-Y)) # reduce_mean : finding average
a = tf.Variable(0.01)
optimizer = tf.keras.optimizers.SGD(a) # GradientDescent algorithm 경사하강
train = optimizer.minimize(cost)

for i in range(5001):
    sess.run(train, feed_dict={X:x_data,Y:y_data})
    if i%500 == 0: # Once in 500 times, print
        print(i,sess.run(cost,feed_dict={X:x_data,Y:y_data}), sess.run(w), sess.run(b))

print(sess.run(H, feed_dict={X:[8]}))