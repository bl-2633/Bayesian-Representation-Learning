#!/usr/bin/env python

"""Batch training helper"""
# Author:Ben Lai bl2633@columbia.edu

import numpy as np


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
        #batch = batch.astype(np.float32) / 255.0
        #batch = np.random.binomial(1, batch)
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

