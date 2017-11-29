#!/usr/bin/env python

"""Batch training helper"""
# Author:Ben Lai bl2633@columbia.edu

import numpy as np
import pickle
import matplotlib.pyplot as plt

def generator(data, batch_size):

    starts = 0
    while True:
        stop = starts + batch_size
        diff = stop - data.shape[0]
        if diff <= 0:
            batch = data[starts:stop]
            starts += batch_size
        else:
            batch = np.concatenate((data[starts:], data[:diff]))
            starts = diff
        # batch = batch.astype(np.float32) / 255.0
        # batch = np.random.binomial(1, batch)
        yield batch


def generator_xy(arrays, batch_size):
    starts = [0] * len(arrays)  # pointers to where we are in iteration
    while True:
        batches = []
        for i, array in enumerate(arrays):
            start = starts[i]
            stop = start + batch_size
            diff = stop - array.shape[0]
            if diff <= 0:
                batch = array[start:stop]
                starts[i] += batch_size
            else:
                batch = np.concatenate((array[start:], array[:diff]))
                starts[i] = diff
            batches.append(batch)
        yield batches



def load_data(path):
    f = open(path,'rb')
    return pickle.load(f)



def accuracy(y_true, y_pred):
    correct = 0
    for i in range(len(y_true)):
        if y_true[i] == y_pred[i]:
            correct += 1
        else:
            pass
    return correct/len(y_true)* 1.


def visulize(n,x_real, x_pred = None, show_pred = False):
    # visulize the data set
    if not show_pred:
        plt.figure(figsize=(n, 4))
        for i in range(n):
            # display original
            ax = plt.subplot(2, n, i + 1)
            plt.imshow(x_real[i].reshape(28, 28))
            plt.gray()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

        plt.show()
    else:
        plt.figure(figsize=(2 * n, 4))
        for i in range(n):
            ax = plt.subplot(2, n, i + 1)
            plt.imshow(x_real[i].reshape(28, 28))
            plt.gray()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

            # display reconstruction
            ax = plt.subplot(2, n, i + 1 + n)
            plt.imshow(x_pred[i].reshape(28, 28))
            plt.gray()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
        plt.show()
