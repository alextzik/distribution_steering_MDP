##########################################################################################
#                                      USEFUL FUNCTIONS                                  #
#                                                                                        #       
##########################################################################################

import numpy as np
import scipy as sp

"""
    Computes the KL divergence between two sample sets
    args:
        - state_1 (instance of State)
        - state_2 (instance of State)

    Adapted from https://github.com/sisl/DeepNFStateEstimation/blob/main/kl_estimator.py
"""
def two_sample_kl_estimator(state1, state2):
    x1 = state1.samples
    x2 = state2.samples

    d, n = x1.shape
    _, m = x2.shape

    k = 1  # Typically 1-NN

    # Compute all pairwise distances
    dists_x1 = np.linalg.norm(x1[:, :, np.newaxis] - x1[:, np.newaxis, :], axis=0)
    dists_x2 = np.linalg.norm(x1[:, :, np.newaxis] - x2[:, np.newaxis, :], axis=0)

    # Get the k-th nearest neighbor distances
    r_k = np.partition(dists_x1, k, axis=1)[:, k]
    s_k = np.partition(dists_x2, k-1, axis=1)[:, k-1]

    # Calculate the Kullback-Leibler divergence
    dkl = -np.sum(d / n * np.log(r_k / s_k))
    dkl += np.log(m / (n - 1.0))

    return dkl

"""
    Computes the Wasserstein distance between a set of samples and N(mu, Sigma), assuming the samples 
    follow a Gaussian distribution.
    args:
        - samples: np.array [dim, num_samples]
        - target mean: np.array
        - target covariance: np.array
"""
def wasserstein_dist(samples, mean, covariance):
    empirical_mean = np.mean(samples, axis=1)
    empirical_cov = np.cov(samples)
    
    dist = np.linalg.norm(empirical_mean - mean)**2 + \
                        np.trace(empirical_cov + covariance - 2*sp.linalg.sqrtm( sp.linalg.sqrtm(covariance)@empirical_cov@sp.linalg.sqrtm(covariance)  ) )
                
    dist = np.sqrt(dist)

    return dist