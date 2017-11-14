from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import numpy as np
import edward as ed


mnist = input_data.read_data_sets('data',one_hot = True)


z = Normal(mu = tf.zeros([N,d],sigma = tf.ones([N,d])))
h = slim.fully_connected(z,256,activation_fn = tf.nn.relu)
x = bernoulli(logits = slim.fully_connected(h,28*28))



