#!/usr/bin/env python

"""Sampler for genetics/imaging joint modeling"""
# Author:Ben Lai bl2633@columbia.edu

import numpy as np
from scipy import stats

np.random.seed(100)

class snp_img_gen:


    def __init__(self, num_snp, num_sample):
        #size of the sample
        self.num_snp = num_snp
        self.num_sample = num_sample
        self.p = np.random.uniform(0.05,0.95,size = self.num_snp)
        self.sigma = 0.1
        #initialize the samples
        self.snp = np.zeros([self.num_sample, self.num_snp])
        self.phe = np.zeros(self.num_sample)
        #sample snps
        for i in range(self.num_sample):
            snp_i = np.random.binomial(2,self.p,self.num_snp)
            self.snp[i] = snp_i

        #sample beta
        self.beta = np.random.normal(0,.01,(self.num_snp,num_img_feat))



        #generate phenotype
        score = np.dot(self.snp,self.beta)
        top_score_index = np.flip(np.argsort(score),0)[:self.num_sample/100]
        for i in top_score_index:
            self.phe[i] = 1



    def get_snp(self):

        return self.snp

    def get_beta(self):

        return self.beta


    def get_phenotype(self):

        return self.phe

