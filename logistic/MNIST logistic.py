import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import time


MNIST = input_data.read_data_sets('/tmp/tensorflow/mnist/input_data', one_hot = True)

learning_rate = 0.01
batch_size = 128
epochs = 25

X = tf.placeholder(tf.float32, [batch_size, 784], name = "image")#this creates a matrix with batch_size columns, with 784 (28 X 28 pixels)
Y = tf.placeholder(tf.float32, [batch_size,10], name = "label")#this creates the labels dataset. Thisi s a one-hot vector

w = tf.Variable(tf.random_normal(shape = [784,10], stddev = 0.01), name = "weights")
b = tf.Variable(tf.zeros([10]), name = "bias")

logits = tf.matmul(X,w) + b # multiplies X by w, making a 1 X 10 vector, and add bias to them

entropy = tf.nn.softmax_cross_entropy_with_logits(logits = logits,labels = Y)
loss = tf.reduce_mean(entropy)#reduces the examples down to one value

optimizer = tf.train.GradientDescentOptimizer(learning_rate = learning_rate).minimize(loss)#this is the optimize function

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    number_batches = int(MNIST.train.num_examples/batch_size)
    for i in range(epochs):
        for j in range(number_batches):
            batch_xs, batch_ys = MNIST.train.next_batch(128)
            sess.run([optimizer,loss], feed_dict={X: batch_xs, Y:batch_ys})
            print("running")

    writer = tf.summary.FileWriter('tmp/MNISTlogistic', sess.graph)
    writer.close()  # you need to close the writer in order to write down the data

