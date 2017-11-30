import numpy as np
import tensorflow as tf
import edward as ed
from edward.models import Dirichlet
from edward.models import Categorical
from edward.models import Bernoulli
from edward.models import Multinomial

def generate_PSD(N, M, K):
    alpha = np.ones(K) / K
    eta = np.ones(2) / 2
    theta = np.random.dirichlet(alpha, size=M)
    beta = np.random.dirichlet(eta, size=(K, N))
    DATA = []
    for m in range(M):
        Z_tmp = np.random.multinomial(1, theta[m], size=(2, N))
        Z = np.zeros(shape=(2, N), dtype='int32')
        X = np.zeros(N, dtype='int32')
        for _ in range(2):
            for n in range(N):
                Z[_][n] = np.argmax(Z_tmp[_][n])
                p = beta[Z[_][n]][n]
                X[n] += 1 if np.random.random() <= p[0] else 0
        DATA.append(X)
    DATA = np.array(DATA)
    Z = np.array(Z)
    return DATA, Z, theta, beta

N = 10
M = 10
K = 3
DATA, Z, theta, beta = generate_PSD(N, M, K)
theta = Dirichlet(tf.zeros([M, K]) + 0.1)
beta = Dirichlet(tf.zeros([K, N, 2]) + 0.05)
Z = [[[0] * N] * 2] * M
W_intermediate = [[[0] * N] * 2] * M
W_ob = [[0] * M] * N
for m in range(M):
    for n in range(N):
        Z[m][0][n] = Categorical(theta[m, :])
        W_intermediate[m][0][n] = Bernoulli(beta[Z[m][0][n], :])
        Z[m][1][n] = Categorical(theta[m, :])
        W_intermediate[m][1][n] = Bernoulli(beta[Z[m][1][n], :])
        W_ob[m][n] = tf.reduce_sum(tf.multiply((W_intermediate[m][0][n] + W_intermediate[m][1][n]), tf.constant([0,1], dtype='int32')))
qtheta = Dirichlet(tf.zeros([M, K]) + 0.1)
qbeta = Dirichlet(tf.zeros([K, N, 2]) + 0.05)
data_dict = {}
for m in range(M):
    for n in range(N):
        data_dict[W_ob[m][n]] = DATA[m][n]
inference = ed.KLpq({qtheta: theta, qbeta :  beta}, data=data_dict)
