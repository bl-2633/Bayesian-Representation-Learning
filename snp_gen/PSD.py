import numpy as np
import edward as ed
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns


from edward.models import Normal, Dirichlet, Categorical, Multinomial


def build_toy_data(N):
    K = 3
    M = 10
    N = 1000
    alpha = np.absolute(np.random.normal(0.8,0.01,K))
    beta = np.absolute(np.random.normal(0.1,0.01,M))
    thetas = np.random.dirichlet(alpha,N)
    phis = np.random.dirichlet(beta,K)
    X = []
    for theta in thetas:
        topics = np.random.multinomial(1,theta)
        print(topics)


    print(theta)
    return phis



D = 4
N = [11502, 213, 1523, 1351]
K = 10
V = 1000

theta = Dirichlet(alpha=tf.zeros([D, K]) + 0.1)
phi = Dirichlet(alpha=tf.zeros([K, V]) + 0.05)
z = [[0] * np.max(N)] * D
w = [[0] * np.max(N)] * D
for d in range(D):
    for n in range(N[d]):
        z[d][n] = Categorical(pi=theta[d, :])
        w[d][n] = Categorical(pi=phi[z[d][n], :])






phi = build_toy_data(1)
