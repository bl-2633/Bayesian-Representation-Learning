#!/usr/bin/env python

"""Sampler for genetics/imaging joint modeling"""
# Author:Ben Lai bl2633@columbia.edu

import numpy as np
from scipy import stats

np.random.seed(100)

class snp_img_gen:


    def __init__(self, num_snp, num_sample, num_img_feat):
        #size of the sample
        self.num_snp = num_snp
        self.num_sample = num_sample
        self.num_img_feat = num_img_feat
        self.p = np.random.uniform(0.05,0.95,size = self.num_snp)
        self.sigma = 0.1
        #initialize the samples
        self.snp = np.zeros([self.num_sample, self.num_snp])
        self.img = np.zeros([self.num_sample, self.num_img_feat])
        self.phe = np.zeros(self.num_sample)
        #sample snps
        for i in range(self.num_sample):
            snp_i = np.random.binomial(2,self.p,self.num_snp)
            self.snp[i] = snp_i

        #sample beta_1
        self.beta_1 = np.random.normal(0,.01,(self.num_snp,num_img_feat))

        #generate img matrix
        self.img = np.dot(self.snp,self.beta_1) + np.random.normal(0,self.sigma,(self.num_sample,self.num_img_feat))

        #sample beta_2
        self.beta_2 = np.random.normal(0.1,0.01,self.num_img_feat)

        #generate phenotype
        score = np.dot(np.dot(self.snp,self.beta_1),self.beta_2)
        top_score_index = np.flip(np.argsort(score),0)[:self.num_sample/100]
        for i in top_score_index:
            self.phe[i] = 1



    def get_snp(self):

        return self.snp

    def get_beta_1(self):

        return self.beta_1

    def get_img(self):

        return self.img

    def get_beta_2(self):

        return self.beta_2

    def get_phenotype(self):

        return self.phe

'''
sampler = snp_img_gen(100,1000,50)

beta_1 = sampler.beta_1

beta_1_m1 = beta_1[:,0]

img = sampler.img
img_m1 = img[:,0]

snp = sampler.snp


print(snp.shape)
print(img_m1.shape)
'''

#slope, intercept, r_value, p_value, std_err = stats.linregress(snp,img_m1)
