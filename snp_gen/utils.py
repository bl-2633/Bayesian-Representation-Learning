#!/usr/bin/env python

"""Batch training helper"""
# Author:Ben Lai bl2633@columbia.edu

import numpy as np


def generator(data, batch_size):

    """Generate batches, one with respect to each array's first axis."""
    starts = 0  # pointers to where we are in iteration
    while True:
        stop = starts + batch_size
        diff = stop - data.shape[0]
        if diff <= 0:
            batch = data[starts:stop]
            starts += batch_size
        else:
            batch = np.concatenate((data[starts:], data[:diff]))
            starts = diff
        batch = batch.astype(np.float32) / 255.0  # normalize pixel intensities
        batch = np.random.binomial(1, batch)  # binarize images
        yield batch
