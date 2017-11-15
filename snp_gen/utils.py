#!/usr/bin/env python

"""Batch training helper"""
# Author:Ben Lai bl2633@columbia.edu

import numpy as np


def generator(data, batch_size):

    """Generate batches, one with respect to each array's first axis."""
    starts = [0] * len(data)  # pointers to where we are in iteration
    while True:
        batches = []
        for i, array in enumerate(data):
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
