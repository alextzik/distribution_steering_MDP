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

    K_control_1 = torch.tensor(sample_U @ sample_S @ sample_V.T , dtype=torch.float32, requires_grad=True)
    b_control_1 = torch.randn((dim_input, 1), requires_grad=True)

    K_control_2 = torch.tensor(sample_U @ sample_S @ sample_V.T , dtype=torch.float32, requires_grad=True)
    b_control_2 = torch.randn((dim_input, 1), requires_grad=True)

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
    upto = int(samples.shape[1]*0.7)
    for step in range(1000):

        next_samples_b1_1 = A_dyn@samples_torch[:, :upto] + B_dyn@(K_control_1@samples_torch[:, :upto]+b_control_1)
        next_samples_b1_2 = A_dyn@next_samples_b1_1 + B_dyn@(K_control_1@next_samples_b1_1+b_control_1)

        next_samples_b2_1 = A_dyn@samples_torch[:, upto:] + B_dyn@(K_control_2@samples_torch[:, upto:]+b_control_2)
        next_samples_b2_2 = A_dyn@next_samples_b2_1 + B_dyn@(K_control_2@next_samples_b2_1+b_control_2)

        # Assume qs_torch, next_samples, and bs_torch are defined and require gradients
        linear_combination_1 = qs_torch.T @ next_samples_b1_2[0, :].reshape(1, -1) + bs_torch
        linear_combination_2 = qs_torch.T @ next_samples_b2_2[0, :].reshape(1, -1) + bs_torch

        # Use softplus to create a differentiable approximation of the step function
        output_1 = torch.sigmoid(100*linear_combination_1)
        output_2 = torch.sigmoid(100*linear_combination_2)

        # Now you can sum and compute gradients
        vals_1 = torch.sum(output_1, dim=1)/samples.shape[1]
        vals_2 = torch.sum(output_2, dim=1)/samples.shape[1]
        vals = vals_1 + vals_2

        res = torch.sum(torch.abs(vals -  target_tensor))/pars.NUM_HALFSPACES

        res.backward(retain_graph=False)
        dists += [res.data]

        with torch.no_grad():

            denom_K_1 = torch.max(torch.abs(K_control_1.grad.data))
            denom_b_1 = torch.max(torch.abs(b_control_1.grad.data))

            denom_K_2 = torch.max(torch.abs(K_control_2.grad.data))
            denom_b_2 = torch.max(torch.abs(b_control_2.grad.data))

            if denom_K_1 == 0.0:
                denom_K_1 = 1.
            if denom_b_1 == 0.0:
                denom_b_1 = 1.
            if denom_K_2 == 0.0:
                denom_K_2 = 1.
            if denom_b_2 == 0.0:
                denom_b_2 = 1.

            K_control_1.data += - step_size*K_control_1.grad.data/denom_K_1
            b_control_1.data += - step_size*b_control_1.grad.data/denom_b_1

            K_control_2.data += - step_size*K_control_2.grad.data/denom_K_2
            b_control_2.data += - step_size*b_control_2.grad.data/denom_b_2

        K_control_1.grad.zero_()
        b_control_1.grad.zero_()

        K_control_2.grad.zero_()
        b_control_2.grad.zero_()

    samples_1 = (A_dyn@samples_torch + B_dyn@(K_control_1@samples_torch+b_control_1)).detach().numpy()
    samples_2 = (A_dyn@samples_torch + B_dyn@(K_control_2@samples_torch+b_control_2)).detach().numpy()

    return np.hstack([samples_1[:, :upto], samples_2[:, upto:]])