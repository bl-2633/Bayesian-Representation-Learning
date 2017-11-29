import tensorflow as tf


def encoder(x, W_encode, b_encode):
    h = tf.nn.relu(tf.matmul(x,W_encode) + b_encode)
    return h

def decoder(x, W_decode, b_decode):
    h = tf.nn.sigmoid(tf.matmul(x, W_decode) + b_decode)
    return h


def mlp(x, W_1, W_2, b_1, b_2):
    h = tf.nn.relu(tf.matmul(x, W_1) + b_1)
    h = tf.nn.softmax(tf.matmul(h, W_2) + b_2)
    return h

def NN_classifier(x, W_0, W_1, b_0, b_1):
    h = tf.nn.relu(tf.matmul(x,W_0) + b_0)
    h = tf.matmul(h,W_1) + b_1
    h = tf.nn.softmax(h)
    return h
