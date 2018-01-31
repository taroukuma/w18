import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt

mnist=input_data.read_data_sets("MNIST_data/",one_hot=True)

#print(mnist.train.images.shape) 
#plt.imshow(mnist.train.images[100,:].reshape(28,28),cmap='gist_gray');plt.show()

#total number of layers
num_layers=3
#no. of units in each layer
units=[784,1000,1000,10]

#placeholders
x=tf.placeholder(tf.float32,shape=[None,784])
y_true=tf.placeholder(tf.float32,shape=[None,10])
#variables
W1=tf.Variable(tf.random_normal([784,units[1]]))
b1=tf.Variable(tf.random_normal([units[1]]))
W2=tf.Variable(tf.random_normal([units[1],units[2]]))
b2=tf.Variable(tf.random_normal([units[2]]))
W3=tf.Variable(tf.random_normal([units[2],units[3]]))
b3=tf.Variable(tf.random_normal([units[3]]))

#graph operations
y1=tf.add(tf.matmul(x,W1),b1)
y2=tf.add(tf.matmul(y1,W2),b2)
y=tf.matmul(y2,W3)+b3
#loss function
cross_entropy=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_true,logits=y))
#optimizer
optimizer=tf.train.GradientDescentOptimizer(learning_rate=0.001)
train=optimizer.minimize(cross_entropy)
#session
init=tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)

    for step in range(900):
        batch_x,batch_y=mnist.train.next_batch(100)
        sess.run(train,feed_dict={x:batch_x,y_true:batch_y})
    #evaluation
    correct_prediction=tf.equal(tf.argmax(y,1),tf.argmax(y_true,1)) #THIS IS A NEW NODE
    acc=tf.reduce_mean(tf.cast(correct_prediction,tf.float32)) #SO IS THIS
    print(sess.run(acc,feed_dict={x:mnist.test.images,y_true:mnist.test.labels}))
    print('\n')
    print(sess.run(acc,feed_dict={x:mnist.train.images,y_true:mnist.train.labels}))
