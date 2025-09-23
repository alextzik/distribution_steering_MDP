import math
import numpy as np
import torch
from utils import compute_heur_dist, sample_orthogonal_mat
import matplotlib.pyplot as plt

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

def gradient_algorithm(samples: np.ndarray, 
                       dim_state:int, 
                       dim_input:int, 
                       target_state, 
                       qs_eval:np.ndarray, 
                       bs_eval:np.ndarray, 
                       qs_train:np.ndarray,
                        bs_train:np.ndarray,
                       n_dyn_steps:int = 1) -> np.ndarray:
    min_dim = np.minimum(dim_input, dim_state)

    sample_V = sample_orthogonal_mat(dim=dim_state)
    sample_U = sample_orthogonal_mat(dim=dim_input)
    sample_S = np.zeros(shape=(dim_input, dim_state))
    sample_S[0:min_dim, 0:min_dim] = np.random.uniform(low=0., high=0.1, size=(min_dim,))

    K_control = torch.tensor(sample_U @ sample_S @ sample_V.T , dtype=torch.float32, requires_grad=True)
    b_control = torch.randn((dim_input, 1), requires_grad=True)

    #######################
    samples_torch = torch.tensor(samples, dtype=torch.float32, requires_grad=False)
    
    qs_eval_torch = torch.tensor(qs_eval, dtype=torch.float32, requires_grad=False)
    bs_eval_torch = torch.tensor(bs_eval, dtype=torch.float32, requires_grad=False)

    qs_train_torch = torch.tensor(qs_train, dtype=torch.float32, requires_grad=False)
    bs_train_torch = torch.tensor(bs_train, dtype=torch.float32, requires_grad=False)
    
    
    target_tensor = torch.tensor(target_state.prob_contents_train, dtype=torch.float32, requires_grad=False)

    ######################
    step_size = 1e-2

    dists = []
    dt = 0.1

    for step in range(300):

        # Unroll n_dyn_steps of dynamics starting from initial samples each optimization step
        cur_samples = samples_torch.clone()
        for _dyn in range(n_dyn_steps):
            us = K_control @ cur_samples + b_control  # shape (2, N)
            v = us[0, :]
            omega = us[1, :]
            theta = cur_samples[2, :]
            x_next = cur_samples[0, :] + dt * v * torch.cos(theta)
            y_next = cur_samples[1, :] + dt * v * torch.sin(theta)
            th_next = theta + dt * omega
            cur_samples = torch.stack([x_next, y_next, th_next], dim=0)
        next_samples = cur_samples
            
        # Assume qs_torch, next_samples, and bs_torch are defined and require gradients
        linear_combination = qs_train_torch.T @ next_samples[:2, :] + bs_train_torch
        # Use softplus to create a differentiable approximation of the step function
        output = torch.sigmoid(100*linear_combination)

        # Now you can sum and compute gradients
        vals = torch.sum(output, dim=1)/samples.shape[1]
        res = torch.sum(torch.abs(vals -  target_tensor))/vals.shape[0]

        res.backward(retain_graph=False)
        dists += [res.data]

        with torch.no_grad():

            denom_K = torch.max(torch.abs(K_control.grad.data))
            denom_b = torch.max(torch.abs(b_control.grad.data))

            if denom_K == 0.0:
                denom_K = 1.
            if denom_b == 0.0:
                denom_b = 1.

            denom = np.maximum(denom_K, denom_b)

            K_control.data += - step_size*K_control.grad.data/denom
            b_control.data += - step_size*b_control.grad.data/denom

        K_control.grad.zero_()
        b_control.grad.zero_()

    dists = []
    for _dyn in range(20):
        us = K_control @ samples_torch + b_control  # shape (2, N)
        v = us[0, :]
        omega = us[1, :]
        theta = samples_torch[2, :]
        x_next = samples_torch[0, :] + dt * v * torch.cos(theta)
        y_next = samples_torch[1, :] + dt * v * torch.sin(theta)
        th_next = theta + dt * omega
        samples_torch = torch.stack([x_next, y_next, th_next], dim=0)

        dists.append(compute_heur_dist(samples_torch.detach().numpy(), target_state, qs_eval, bs_eval))

    return samples_torch.detach().numpy(), dists