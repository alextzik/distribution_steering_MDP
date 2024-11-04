import math
import numpy as np
import torch
from utils import two_sample_kl_estimator, compute_wasserstein_dist, plot_level_curves_normal, compute_heur_dist, sample_orthogonal_mat
import parameters as pars
import matplotlib.pyplot as plt

def baseline_algorithm(samples: np.ndarray, state_dim:int, control_dim:int, target_state, qs:np.ndarray, bs:np.ndarray):
    sample_V = sample_orthogonal_mat(dim=2)
    sample_U = sample_orthogonal_mat(dim=2)
    sample_S = np.zeros(shape=(2, 2))
    sample_S[0:2, 0:2] = np.random.uniform(low=0., high=0.5, size=(2,))

    K_control = torch.tensor(sample_U @ sample_S @ sample_V.T , dtype=torch.float32, requires_grad=True)
    b_control = torch.randn((control_dim, 1), requires_grad=True)

    #######################
    samples_torch = torch.tensor(samples, dtype=torch.float32, requires_grad=False)
    
    qs_torch = torch.tensor(qs, dtype=torch.float32, requires_grad=False)
    bs_torch = torch.tensor(bs, dtype=torch.float32, requires_grad=False)
    
    target_tensor = torch.tensor(target_state.prob_contents, dtype=torch.float32, requires_grad=False)

    #######################
    A_dyn = torch.eye(2, requires_grad=False)
    B_dyn = 0.1*torch.tensor([[1., 0.], [0., 1.]], requires_grad=False)

    step_size = 1e-4

    dists = []
    for step in range(1000):

        next_samples = A_dyn@samples_torch + B_dyn@(K_control@samples_torch+b_control)
        # Assume qs_torch, next_samples, and bs_torch are defined and require gradients
        linear_combination = qs_torch.T @ next_samples + bs_torch
        # Use softplus to create a differentiable approximation of the step function
        output = torch.sigmoid(100*linear_combination)

        # Now you can sum and compute gradients
        vals = torch.sum(output, dim=1)/samples.shape[1]
        res = torch.sum(torch.abs(vals -  target_tensor))/pars.NUM_HALFSPACES

        res.backward(retain_graph=False)
        dists += [res.data]

        with torch.no_grad():
            K_control.data += - step_size*K_control.grad.data/torch.max(torch.abs(K_control.grad.data))
            b_control.data += - step_size*b_control.grad.data//torch.max(torch.abs(b_control.grad.data))

        K_control.grad.zero_()
        b_control.grad.zero_()


    return (A_dyn@samples_torch + B_dyn@(K_control@samples_torch+b_control)).detach().numpy()



