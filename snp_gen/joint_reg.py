#!/usr/bin/env python

"""simple regression for genetics/imaging joint modeling"""
# Author:Ben Lai bl2633@columbia.edu

from snp_img_gen import *
from sklearn.linear_model import LogisticRegression, LinearRegression
from matplotlib import pyplot as plt
import seaborn as sns



sampler = snp_img_gen(100000,1000,50)

beta_1 = sampler.beta_1

beta_1_m1 = beta_1[:,0]

img = sampler.img
img_m1 = img[:,0]

snp = sampler.snp

reg1 = LinearRegression().fit(snp,img_m1)
coef1 = reg1.coef_

beta_2 = sampler.beta_2
pheno = sampler.phe
reg2 = LogisticRegression().fit(img,pheno)
coef2 = reg2.coef_

sns.distplot(coef1)
sns.distplot(beta_1_m1)
plt.show()

sns.distplot(coef2,hist = False)
sns.distplot(beta_2,hist = False)
plt.show()

