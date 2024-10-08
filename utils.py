"""
    Developed by Alexandros Tzikas
                alextzik@stanford.edu

"""

##########################################################################################
#                                      USEFUL FUNCTIONS                                  #
#                                                                                        #       
##########################################################################################

import numpy as np
import scipy as sp
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
import parameters as pars

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
def compute_wasserstein_dist(samples, mean, covariance):
    empirical_mean = np.mean(samples, axis=1)
    empirical_cov = np.cov(samples)
    
    dist = np.linalg.norm(empirical_mean - mean)**2 + \
                        np.trace(empirical_cov + covariance - 2*sp.linalg.sqrtm( sp.linalg.sqrtm(covariance)@empirical_cov@sp.linalg.sqrtm(covariance)  ) )
                
    dist = np.sqrt(dist)

    return dist

"""
    Algorithm 1 of the paper. Computes the half-space distance between the samples and the target_state distribution (given as a GMM), using the half-spaces
    in qs and bs
"""
def compute_heur_dist(samples:np.ndarray, target_state, qs:np.ndarray, bs:np.ndarray) -> float:

    vals = np.sum(qs.T@samples + bs >=0, axis = 1)/samples.shape[1]
        
    res = np.sum(np.abs(vals - np.array(target_state.prob_contents)))/pars.NUM_HALFSPACES

    return res


"""
    Plot level curves of Normal Distribution
"""
def plot_level_curves_normal(mean, covar, color):
    # Create a grid of (x, y) values
    x = np.linspace(mean[0]-10, mean[0]+10, 100)
    y = np.linspace(mean[1]-10, mean[1]+10, 100)
    X, Y = np.meshgrid(x, y)
    pos = np.dstack((X, Y))

    # Create the multivariate normal distribution
    rv = multivariate_normal(mean, covar)

    # Calculate the probability density function (pdf) values
    Z = rv.pdf(pos)

    # Plot the contour plot
    plt.contour(X, Y, Z, levels=20, cmap=color) 


"""
    Samples an orthogonal matrix of dimension dim
"""
def sample_orthogonal_mat(dim:int):
    M = np.random.randn(dim, dim)
    Q, R = np.linalg.qr(M)
    L = np.sign(np.diag(R))
    return Q*L[None,:]