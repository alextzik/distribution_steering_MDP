"""
    Developed by Alexandros Tzikas
                alextzik@stanford.edu

"""

##########################################################################################
#                                  IMPLEMENTATION OF THE                                 #
#                                   PROPOSED ALGORITHM                                   #       
##########################################################################################

##########################################################################################
# Dependencies
import math
import numpy as np
from sklearn.covariance import EmpiricalCovariance
from scipy.stats import ortho_group
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
from scipy.stats import norm

from typing import Callable

import parameters as pars
from utils import plot_level_curves_normal, compute_heur_dist, sample_orthogonal_mat, compute_heur_dist_unscented
from algorithm import gradient_algorithm

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 20

##########################################################################################
class dynamics:
    """
        This class represents the dynamics equations for the state evolution

        args:
            - dim_state [int]: the dimension of the state
            - dim_input [int]: the dimension of the input
            - dyn_func [function]: propagates a state sample, using an input, according to the dynamics
    """
    def __init__(self, dim_state:int, dim_input:int, dyn_func:Callable[[np.ndarray, np.ndarray], np.ndarray]) -> None:
        self.dim_state = dim_state
        self.dim_input = dim_input
        self.dyn_func  = dyn_func

class target_density:
    def __init__(self, weights:list, means:list, covs:list) -> None:
        """
            This class characterizes the target density as a GMM

            args:
                - weights list[float]: the weight of each component
                - means list[1d np.ndarray]: the means of the components
                - covs list[2d np.ndarray]: the covs of the components
        """

        self.weights = weights
        self.means = means
        self.covs = covs

    def compute_prob_contents(self, qs:np.ndarray, bs:np.ndarray) -> None:
        """
            Compute the prob content of the GMM in the halfspaces dictated by qs and bs

            qs [(dim, num_half-spaces) np.ndarray]: the normal vectors
            bs [(num_halfspaces, 1) np.ndarray]: the offsets
        """
        prob_contents = []
        for i in range(qs.shape[1]):
            prob_content = 0.

            q = qs[:, i].reshape(-1,1)
            b = bs[i, 0]

            for c in range(len(self.weights)):
                r_mean = q.T@self.means[c].reshape(-1,1) + b
                r_cov = q.T@self.covs[c]@q

                prob_content += self.weights[c]*(1-norm.cdf(x=0., loc=r_mean, scale=np.sqrt(r_cov)))

            prob_contents.append(prob_content.item())

        self.prob_contents = prob_contents

##########################################################################################
class State:
    def __init__(self) -> None:
        """This class contains the set of samples for a given node in the tree search.
        
            pars:
                - dim_state [int]: dimensionality of state
                - num_samples [int]: number of available samples
                - samples [(dim_state, num_samples) np.array]: the samples as columns of an array
        """
        self.dim_state = None
        self.num_samples = None
        self.samples = None

    def set(self, samples:np.ndarray) -> None:
        """
            Sets the state's particle set using a provided particle set
            args:
                - samples [(dim, num_samples) np.array]: contains the samples as columns
        """
        self.dim_state = samples.shape[0]
        self.num_samples = samples.shape[1]
        self.samples = samples

    def sample(self, mean:np.ndarray, covariance:np.ndarray, num_samples:int) -> None:
        """
        Sets the state's particle set by sampling a given Gaussian for a given
        number of samples

        args: 
                - mean [1d np.array]: the mean of the distribution
                - covariance [2d np.ndarray]: the covariance of the distribution
                - num_samples [int]: number of samples
        """
        self.num_samples = num_samples
        self.samples = np.random.multivariate_normal(mean=mean, cov=covariance, size=self.num_samples).T
        self.dim_state = self.samples.shape[0]

##########################################################################################
##########################################################################################
#                                      METHOD DEPLOYMENT                                 #
#                                                                                        #       
##########################################################################################

# Example usage
    
def dyn_func(x, u):
    dt = 0.1
    # A = np.array([[1, dt], [0, 1]])
    # B = np.array([[0, dt]]).reshape(-1,1)

    # A = np.eye(2)
    # B = 0.1*np.array([[1., 0.], [0., 1.]])

    x_next = np.array([x[0] + dt*u[0]*np.cos(x[2]),
                       x[1] + dt*u[0]*np.sin(x[2]),
                       x[2] + u[1]*dt]).reshape(-1,)

    return x_next

dyns = dynamics(3, 2, dyn_func)

num_steps = 100

# intiial state
state = State()
init_mean = np.array([-2, -2., 0.])
init_cov = np.eye(3)
init_cov[2,2]=0.
state.sample(mean = init_mean, covariance=init_cov, num_samples=3000)
baseline_state_samples = state.samples

# Target density
target_means = [np.array([3., 2.])]
target_covs = [np.array([[2, 1.5], [1.5, 2]])]
target_weights = [1.]
target_state = target_density(target_weights, target_means, target_covs)

# Distance heuristic half-spaces    
dirs = np.random.normal(size=(pars.NUM_HALFSPACES, 2))
dirs /= np.linalg.norm(dirs, axis=1, keepdims=True) + 1e-12
qs = dirs

quantiles = []
for q in qs:
    proj_samples = q@target_state.means[0].reshape(-1,1) + np.sqrt(q.T@target_state.covs[0]@q)*np.random.standard_normal(size=(1000,))
    quantiles.append(np.quantile(proj_samples, 0.9))
    quantiles.append(np.quantile(proj_samples, 0.1))


bs = np.linspace(np.min(np.array(quantiles)), np.max(np.array(quantiles)), 100)
bs = np.tile(bs, pars.NUM_HALFSPACES).reshape(-1,1)
qs = np.repeat(qs.T, 100, axis=1)


target_state.compute_prob_contents(qs, bs)
print(target_state.prob_contents)

dists_gradient = [compute_heur_dist(baseline_state_samples, target_state, qs, bs)]

# Main Loop
for t in tqdm(range(num_steps)):

    fig, ax = plt.subplots(figsize=(6, 4), dpi=150)
    plot_level_curves_normal(target_state.means[0], target_state.covs[0], "summer")
    plot_level_curves_normal(init_mean[0:2], init_cov[0:2, 0:2], "summer")
    # Heatmap of sample density instead of raw scatter
    x_vals = baseline_state_samples[0, :]
    y_vals = baseline_state_samples[1, :]
    # Define bins (adjustable)
    bins = 100
    x_min, x_max = -5, 10
    y_min, y_max = -5, 10
    x_edges = np.linspace(x_min, x_max, bins+1)
    y_edges = np.linspace(y_min, y_max, bins+1)
    hist2d, xe, ye = np.histogram2d(x_vals, y_vals, bins=[x_edges, y_edges], density=True)
    im = ax.imshow(hist2d.T, origin='lower', extent=[x_min, x_max, y_min, y_max], aspect='equal', cmap='cividis')
    # make colorbar more transparent (adjust alpha of colorbar patches)
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Density')
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    
    file_dir = os.path.dirname(os.path.realpath(__file__))
    log_dir = os.path.join("/Users/alextzik/Documents/GitHub/distribution_steering_MDP/", "results") 

    os.chdir(log_dir)
    fig.savefig(f"baseline_step_{t}.pdf", bbox_inches='tight')

    baseline_state_samples, _dists_gradient = gradient_algorithm(baseline_state_samples.copy(), 
                                                 3, 2, 
                                                 target_state, qs, bs, 
                                                 n_dyn_steps=25)
    
    dists_gradient += _dists_gradient


np.save("dists_gradient_baseline.npy", dists_gradient)

# np.save('proposed_dists.npy', dists)
plt.plot(range(num_steps), dists_gradient, color='#2ca02c', label="Distance Metric (Alg. 1) for Gradient")
# # plt.plot(range(num_steps), np.array(wass_dists)/np.array(wass_dists).max(), label="Wasserstein Distance (scaled)")
plt.ylabel("Instantaneous Cost")
plt.xlabel("Timestep")
plt.legend()
plt.show()