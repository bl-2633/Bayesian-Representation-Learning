#!/usr/bin/env python

"""VAE with edward"""
# Author:Ben Lai bl2633@columbia.edu

from observations import mnist
import tensorflow as tf
from utils import generator
from edward.util import Progbar
from scipy.misc import imsave
from tensorflow.contrib import slim
from edward.models import Bernoulli, Normal
import edward as ed
import os



d = 2 # latent dimension
N = 100 # batch size

z = Normal(loc = tf.zeros([N,d]),scale = tf.ones([N,d]))
h = slim.fully_connected(z,256,activation_fn = tf.nn.relu)
x = Bernoulli(logits = slim.fully_connected(h,28*28), dtype = tf.float32)

qx = tf.placeholder(tf.float32,[N,28*28])
qh = slim.fully_connected(qx, 256, activation_fn = tf.nn.relu)
qz = Normal(loc = slim.fully_connected(qh, d),
           scale = slim.fully_connected(qh, d, activation_fn = tf.nn.softplus))


(x_train, _), (x_test, _) = mnist('./data')
x_train_generator = generator(x_train, N)

n_epoch = 100
n_iter_per_epoch = x_train.shape[0] // N


inference = ed.KLqp({z: qz}, data={x: qx})
optimizer = tf.train.RMSPropOptimizer(0.01, epsilon=1.0)
inference.initialize(optimizer=optimizer)
tf.global_variables_initializer().run()

for epoch in range(1, n_epoch + 1):
  print("Epoch: {0}".format(epoch))
  avg_loss = 0.0

  pbar = Progbar(n_iter_per_epoch)
  for t in range(1, n_iter_per_epoch + 1):
    pbar.update(t)
    x_batch = next(x_train_generator)
    info_dict = inference.update(feed_dict={qx: x_batch})
    avg_loss += info_dict['loss']

  # Print a lower bound to the average marginal likelihood for an
  # image.
  avg_loss = avg_loss / n_iter_per_epoch
  avg_loss = avg_loss / N
  print("-log p(x) <= {:0.3f}".format(avg_loss))

  # Prior predictive check.
  images = x.eval()
  for m in range(N):
    imsave(os.path.join('./out', '%d.png') % m, images[m].reshape(28, 28))



