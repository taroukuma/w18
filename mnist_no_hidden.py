import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt

mnist=input_data.read_data_sets("MNIST_data/",one_hot=True)

#print(mnist.train.images.shape) 
#plt.imshow(mnist.train.images[100,:].reshape(28,28),cmap='gist_gray');plt.show()

#placeholders
x=tf.placeholder(tf.float32,shape=[None,784])
y_true=tf.placeholder(tf.float32,shape=[None,10])
#variables
W1=tf.Variable(tf.zeros([784,10]))
b1=tf.Variable(tf.zeros([10]))
#graph operations
y=tf.matmul(x,W1)+b1
#loss function
cross_entropy=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_true,logits=y))
#optimizer
optimizer=tf.train.GradientDescentOptimizer(learning_rate=0.5)
train=optimizer.minimize(cross_entropy)
#session
init=tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)

    for step in range(1000):
        batch_x,batch_y=mnist.train.next_batch(100)
        sess.run(train,feed_dict={x:batch_x,y_true:batch_y})
    #evaluation
    correct_prediction=tf.equal(tf.argmax(y,1),tf.argmax(y_true,1)) #THIS IS A NEW NODE
    acc=tf.reduce_mean(tf.cast(correct_prediction,tf.float32)) #SO IS THIS
    print(sess.run(acc,feed_dict={x:mnist.test.images,y_true:mnist.test.labels}))
    
