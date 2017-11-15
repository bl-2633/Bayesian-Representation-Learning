import numpy as np
import edward as ed
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from edward.models import Normal


def build_toy_data(N):
    mu = np.random.normal(-10, 1, size=N)
    X = []
    for i in mu:
        X.append(np.random.normal(i, 1))

    return X


N = 5000
X_real = build_toy_data(N)
sns.distplot(X_real)
plt.show()

mu = Normal(loc=-10., scale=1.)
y = Normal(loc=mu, scale=tf.ones(N))

qmu = Normal(loc=tf.Variable(0.), scale=1.)
inference = ed.KLpq({mu: qmu}, data={y: X_real})
inference.run(n_samples=5, n_iter=2500)

mu_sample = mu.sample(1000).eval()
qmu_sample = qmu.sample(1000).eval()

sns.distplot(mu_sample, hist=False, color='r')
sns.distplot(qmu_sample, hist=False, color='b')
plt.show()
