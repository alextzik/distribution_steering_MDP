#!/usr/bin/env python3
"""
Test script for DescentLQR policy on batched unicycle system.
"""

import torch
import matplotlib.pyplot as plt
import numpy as np
from system import UnicycleSystem, BicycleSystem
from policy import DescentLQRPolicy, ErgodicMPCPolicy

def setup():
    """Create experiment setup with system and policy."""
    
    # Create unicycle system
    system = UnicycleSystem(
        dt=0.1,
        max_velocity=2.0,
        max_angular_velocity=2.0,
        noise_xy_std=0.0,
        noise_theta_std=0.0
    )

    # Create bicycle system
    # system = BicycleSystem(
    #     dt=0.1,
    #     wheelbase=2.5,
    #     max_velocity=2.0,
    #     max_steering_angle=0.5,
    #     max_steering_rate=1.0,
    # )
    
    # Policy parameters
    horizon = 20000
    num_restarts = 1
    num_gradient_steps = 170
    
    # Create 2-component GMM target distribution
    num_target_samples = 20000
    num_halfspaces = 400
    num_bjs = 400
    
    # Component 1: centered at (2, 2) with moderate spread
    mean1 = torch.tensor([5.0, 5.0])
    cov1 = 2*torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    
    # Component 2: centered at (-1, 1) with different orientation
    mean2 = torch.tensor([-10.0, -5.0])
    cov2 = 2*torch.tensor([[1.0, 0.0], [0.0, 1.0]])

    # Equal weights for both components
    weights = torch.tensor([0.5, 0.5])
    
    # Generate target samples from GMM
    component_assignments = torch.multinomial(weights, num_target_samples, replacement=True)
    target_density_samples = torch.zeros(2, num_target_samples)

    for i, (mean, cov) in enumerate(zip([mean1, mean2], [cov1, cov2])):
        mask = component_assignments == i
        num_component_samples = mask.sum().item()
        if num_component_samples > 0:
            component_samples = torch.distributions.MultivariateNormal(mean, cov).sample((num_component_samples,))
            target_density_samples[:, mask] = component_samples.T
    
    # Sample half-space directions from SO(2)
    angles = torch.rand(num_halfspaces) * 2 * np.pi
    qs = torch.stack([torch.cos(angles), torch.sin(angles)], dim=1)  # (num_halfspaces, 2)
    
    # for each q in qs, we will have 100 different bs. They will be linspaced from the low-th to the high-th percentile of q^T target samples
    bs = torch.zeros(num_halfspaces, num_bjs)
    for i, q in enumerate(qs):
        projections = q @ target_density_samples  # (num_target_samples,)
        p_low = torch.quantile(projections, 0.005)
        p_high = torch.quantile(projections, 0.995)
        bs[i] = -1*torch.linspace(p_low, p_high, num_bjs)
    
    # now expand qs and bs to match shape (num_halfspaces*100, 2) and (num_halfspaces*100,). Each q should be matched with the corresponding row from bs
    qs = qs.repeat_interleave(num_bjs, dim=0)  # (num_halfspaces*100, 2)
    bs = bs.flatten().unsqueeze(1)  # (num_halfspaces*100, 1)
    
    policy = ErgodicMPCPolicy(
        system=system,
        horizon=horizon,
        target_density_samples=target_density_samples,
        qs=qs,
        bs=bs,
        soft_constraint_count_penalty=0.0,
        soft_constraint_quad_penalty=0.0,
        num_restarts=num_restarts,
        num_gradient_steps=num_gradient_steps,
        optimizer_class=torch.optim.AdamW
    )
    
    return system, policy

def run_open_loop():
    """Test policy on a single initial state."""
    print("Setting up test...")
    system, policy = setup()
    
    # Initial state: start at (2, 2) with 45 degree heading
    initial_state = torch.tensor([0.0, 0.0, 0.0]) # (x, y, theta) Unicycle
    #initial_state = torch.tensor([2.0, 2.0, np.pi/4, 0.0]) # (x, y, theta, delta) Bicycle
    
    print(f"Initial state: {initial_state}")
    # print(f"Target state: {policy.target_state}")
    print(f"System state dim: {system.get_state_dim()}")
    print(f"System action dim: {system.get_action_dim()}")
    print(f"System constraint dim: {system.get_constraint_dim()}")
    
    print("\nRunning policy optimization...")
    action, info = policy.action_info(initial_state)
    
    print(f"\nOptimization complete!")
    print(f"Immediate action: {action}")
    print(f"Best cost: {info['best_cost']:.4f}")
    print(f"Best cost index: {info['best_cost_index']}")
    
    return system, policy, initial_state, action, info

def visualize_results(system, policy, initial_state, action, info):
    """Visualize the optimization results."""

    # --- New Figure: Heatmaps of trajectory occupancy vs target density samples ---
    best_trajectory = info['best_trajectory']  # (T+1, state_dim)
    traj_xy = best_trajectory[:, :2].detach().numpy()
    target_samples = policy.target_density_samples.detach().numpy()  # (2, N)

    # Define common 2D histogram bins
    all_x = np.concatenate([traj_xy[:,0], target_samples[0]])
    all_y = np.concatenate([traj_xy[:,1], target_samples[1]])
    x_min, x_max = all_x.min()-0.5, all_x.max()+0.5
    y_min, y_max = all_y.min()-0.5, all_y.max()+0.5
    bins = 60
    x_edges = np.linspace(x_min, x_max, bins+1)
    y_edges = np.linspace(y_min, y_max, bins+1)

    traj_hist, _, _ = np.histogram2d(traj_xy[:,0], traj_xy[:,1], bins=[x_edges, y_edges], density=True)
    target_hist, _, _ = np.histogram2d(target_samples[0], target_samples[1], bins=[x_edges, y_edges], density=True)

    # Avoid zeros for log color if needed later
    # Create figure
    fig_hm, axes_hm = plt.subplots(1,2, figsize=(10,4))
    im0 = axes_hm[0].imshow(traj_hist.T, origin='lower', extent=[x_min, x_max, y_min, y_max], aspect='equal', cmap='cividis')
    axes_hm[0].set_title('Trajectory Occupancy')
    axes_hm[0].set_xlabel('X')
    axes_hm[0].set_ylabel('Y')
    plt.colorbar(im0, ax=axes_hm[0], fraction=0.046, pad=0.04)

    im1 = axes_hm[1].imshow(target_hist.T, origin='lower', extent=[x_min, x_max, y_min, y_max], aspect='equal', cmap='cividis')
    axes_hm[1].set_title('Target Density Samples')
    axes_hm[1].set_xlabel('X')
    axes_hm[1].set_ylabel('Y')
    plt.colorbar(im1, ax=axes_hm[1], fraction=0.046, pad=0.04)
    plt.tight_layout()
    # plt.savefig('/Users/saturnv/sandbox/distribution_steering_MDP/ergodic/heatmaps.png', dpi=150, bbox_inches='tight')
    plt.show()

    # --- New Figure: Loss vs optimization step (already shown, but separate figure) ---
    training_losses = info['training_losses']
    fig_loss, ax_loss = plt.subplots(figsize=(6,4))
    ax_loss.plot(training_losses, linewidth=2)
    ax_loss.set_xlabel('Optimization Step')
    ax_loss.set_ylabel('Distance')
    plt.tight_layout()
    # plt.savefig('/Users/saturnv/sandbox/distribution_steering_MDP/ergodic/loss_curve.png', dpi=150, bbox_inches='tight')
    plt.show()

    

def main():
    """Main test function."""

    system, policy, initial_state, action, info = run_open_loop()
    visualize_results(system, policy, initial_state, action, info)

if __name__ == "__main__":
    main()
