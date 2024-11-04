import math
import numpy as np
import torch
from utils import two_sample_kl_estimator, compute_wasserstein_dist, plot_level_curves_normal, compute_heur_dist, sample_orthogonal_mat
import parameters as pars
import matplotlib.pyplot as plt

def baseline_algorithm(samples: np.ndarray, dim_state:int, dim_input:int, target_state, qs:np.ndarray, bs:np.ndarray):
    min_dim = np.minimum(dim_input, dim_state)

    sample_V = sample_orthogonal_mat(dim=dim_state)
    sample_U = sample_orthogonal_mat(dim=dim_input)
    sample_S = np.zeros(shape=(dim_input, dim_state))
    sample_S[0:min_dim, 0:min_dim] = np.random.uniform(low=0., high=0.1, size=(min_dim,))

    K_control = torch.tensor(sample_U @ sample_S @ sample_V.T , dtype=torch.float32, requires_grad=True)
    b_control = torch.randn((dim_input, 1), requires_grad=True)

    #######################
    samples_torch = torch.tensor(samples, dtype=torch.float32, requires_grad=False)
    
    qs_torch = torch.tensor(qs, dtype=torch.float32, requires_grad=False)
    bs_torch = torch.tensor(bs, dtype=torch.float32, requires_grad=False)
    
    target_tensor = torch.tensor(target_state.prob_contents, dtype=torch.float32, requires_grad=False)

    #######################
    dt = 0.1
    A = np.array([[1, dt], [0, 1]])
    B = np.array([[0, dt]]).reshape(-1,1)
    A_dyn = torch.tensor(A, dtype=torch.float32, requires_grad=False)
    B_dyn = torch.tensor(B, dtype=torch.float32, requires_grad=False)

    step_size = 1e-4

    dists = []
    for step in range(1000):

        next_samples = A_dyn@samples_torch + B_dyn@(K_control@samples_torch+b_control)
        # Assume qs_torch, next_samples, and bs_torch are defined and require gradients
        linear_combination = qs_torch.T @ next_samples[0, :].reshape(1, -1) + bs_torch
        # Use softplus to create a differentiable approximation of the step function
        output = torch.sigmoid(100*linear_combination)

        # Now you can sum and compute gradients
        vals = torch.sum(output, dim=1)/samples.shape[1]
        res = torch.sum(torch.abs(vals -  target_tensor))/pars.NUM_HALFSPACES

        res.backward(retain_graph=False)
        dists += [res.data]

        with torch.no_grad():

            denom_K = torch.max(torch.abs(K_control.grad.data))
            denom_b = torch.max(torch.abs(b_control.grad.data))

            if denom_K == 0.0:
                denom_K = 1
            if denom_b == 0.0:
                denom_b = 1

            K_control.data += - step_size*K_control.grad.data/denom_K
            b_control.data += - step_size*b_control.grad.data/denom_b

        K_control.grad.zero_()
        b_control.grad.zero_()


    return (A_dyn@samples_torch + B_dyn@(K_control@samples_torch+b_control)).detach().numpy()