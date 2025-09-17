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
        noise_xy_std=0.01,
        noise_theta_std=0.01
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
    horizon = 10000
    num_restarts = 1
    num_gradient_steps = 1000
    
    # LQR cost matrices
    state_dim = system.get_state_dim()
    action_dim = system.get_action_dim()
    
    # Q matrix: penalize deviation from target state
    Q = torch.eye(state_dim) * 10.0  # Higher weight on position errors
    Q[-1, -1] = 1.0  # Lower weight on heading error
    
    # R matrix: penalize large actions
    R = torch.eye(action_dim) * 0.1
    
    # Target state: reach origin with zero heading
    target_state = torch.zeros(state_dim)
    
    # Create 2-component GMM target distribution
    num_target_samples = 1000
    num_halfspaces = 50
    
    # Component 1: centered at (2, 2) with moderate spread
    mean1 = torch.tensor([3.0, 2.0])
    cov1 = torch.tensor([[0.5, 0.1], [0.1, 0.5]])
    
    # Component 2: centered at (-1, 1) with different orientation
    mean2 = torch.tensor([-1.0, 1.0])
    cov2 = torch.tensor([[0.3, -0.2], [-0.2, 0.8]])
    
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
    
    # Sample half-space biases from -10 to 10
    # for each q in qs, we will have 100 different bs. They will be linspaced from the 10th to the 90th percentile of q^T target samples
    bs = torch.zeros(num_halfspaces, 100)
    for i, q in enumerate(qs):
        projections = q @ target_density_samples  # (num_target_samples,)
        p10 = torch.quantile(projections, 0.1)
        p90 = torch.quantile(projections, 0.9)
        bs[i] = torch.linspace(p10, p90, 100)
    
    # now expand qs and bs to match shape (num_halfspaces*100, 2) and (num_halfspaces*100,). Each q should be matched with the corresponding row from bs
    qs = qs.repeat_interleave(100, dim=0)  # (num_halfspaces*100, 2)
    bs = bs.flatten().unsqueeze(1)  # (num_halfspaces*100, 1)

    # Create policy
    # policy = DescentLQRPolicy(
    #     system=system,
    #     horizon=horizon,
    #     Q=Q,
    #     R=R,
    #     target_state=target_state,
    #     soft_constraint_count_penalty=100.0,
    #     soft_constraint_quad_penalty=100.0,
    #     num_restarts=num_restarts,
    #     num_gradient_steps=num_gradient_steps,
    #     optimizer_class=torch.optim.AdamW,
    #     optimizer_kwargs={'lr': 0.1}
    # )
    
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
        optimizer_class=torch.optim.AdamW,
        optimizer_kwargs={'lr': 0.1}
    )
    
    return system, policy

def run_open_loop():
    """Test policy on a single initial state."""
    print("Setting up test...")
    system, policy = setup()
    
    # Initial state: start at (2, 2) with 45 degree heading
    initial_state = torch.tensor([2.0, 2.0, np.pi/4]) # (x, y, theta) Unicycle
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
    
    # # Extract best trajectory
    # best_trajectory = info['best_trajectory']  # (T+1, state_dim)
    # best_actions = info['best_actions']  # (T, action_dim)
    
    # # Convert to numpy for plotting
    # trajectory_np = best_trajectory.detach().numpy()
    # actions_np = best_actions.detach().numpy()
    
    # # Create figure with subplots (3x2 layout to accommodate training losses)
    # fig, axes = plt.subplots(3, 2, figsize=(12, 15))
    
    # # Plot 1: Trajectory in x-y plane
    # ax1 = axes[0, 0]
    # ax1.plot(trajectory_np[:, 0], trajectory_np[:, 1], 'b-o', linewidth=2, markersize=4)
    # ax1.plot(initial_state[0], initial_state[1], 'go', markersize=10, label='Start')
    # ax1.plot(policy.target_state[0], policy.target_state[1], 'ro', markersize=10, label='Target')
    # ax1.set_xlabel('X Position')
    # ax1.set_ylabel('Y Position')
    # ax1.set_title('Trajectory in X-Y Plane')
    # ax1.legend()
    # ax1.grid(True)
    # ax1.axis('equal')
    
    # # Plot 2: Actions over time
    # ax2 = axes[0, 1]
    # time_steps = np.arange(len(actions_np))
    # ax2.plot(time_steps, actions_np[:, 0], 'b-', label='Linear velocity (v)')
    # ax2.plot(time_steps, actions_np[:, 1], 'r-', label='Angular velocity (ω)')
    # ax2.set_xlabel('Time Step')
    # ax2.set_ylabel('Action Value')
    # ax2.set_title('Actions Over Time')
    # ax2.legend()
    # ax2.grid(True)
    
    # # Plot 3: State evolution
    # ax3 = axes[1, 0]
    # time_steps_full = np.arange(len(trajectory_np))
    # ax3.plot(time_steps_full, trajectory_np[:, 0], 'b-', label='X position')
    # ax3.plot(time_steps_full, trajectory_np[:, 1], 'g-', label='Y position')
    # ax3.plot(time_steps_full, trajectory_np[:, 2], 'r-', label='Heading (θ)')
    # ax3.set_xlabel('Time Step')
    # ax3.set_ylabel('State Value')
    # ax3.set_title('State Evolution')
    # ax3.legend()
    # ax3.grid(True)
    
    # # Plot 4: Cost distribution
    # ax4 = axes[1, 1]
    # costs = info['all_costs'].detach().numpy()
    # ax4.hist(costs, bins=20, alpha=0.7, edgecolor='black')
    # ax4.axvline(info['best_cost'], color='red', linestyle='--', linewidth=2, label=f'Best cost: {info["best_cost"]:.4f}')
    # ax4.set_xlabel('Cost Value')
    # ax4.set_ylabel('Frequency')
    # ax4.set_title('Cost Distribution Across Restarts')
    # ax4.legend()
    # ax4.grid(True)
    
    # # Plot 5: Training losses over optimization steps
    # ax5 = axes[2, 0]
    # training_losses = info['training_losses']
    # ax5.plot(training_losses, 'b-', linewidth=2)
    # ax5.set_xlabel('Optimization Step')
    # ax5.set_ylabel('Minimum Cost')
    # ax5.set_title('Training Losses (Minimum Cost per Step)')
    # ax5.grid(True)
    # ax5.set_yscale('log')  # Use log scale for better visualization of cost reduction
    
    # # Plot 6: Training losses (linear scale) for better view of final convergence
    # ax6 = axes[2, 1]
    # ax6.plot(training_losses, 'b-', linewidth=2)
    # ax6.set_xlabel('Optimization Step')
    # ax6.set_ylabel('Minimum Cost')
    # ax6.set_title('Training Losses (Linear Scale)')
    # ax6.grid(True)
    
    # plt.tight_layout()
    # # plt.savefig('/Users/saturnv/sandbox/distribution_steering_MDP/ergodic/test_results.png', dpi=150, bbox_inches='tight')
    # plt.show()

    # --- New Figure: Heatmaps of trajectory occupancy vs target density samples ---
    best_trajectory = info['best_trajectory']  # (T+1, state_dim)
    traj_xy = best_trajectory[:, :2].detach().numpy()
    target_samples = policy.target_density_samples.detach().numpy()  # (2, N)

    # Define common 2D histogram bins
    all_x = np.concatenate([traj_xy[:,0], target_samples[0]])
    all_y = np.concatenate([traj_xy[:,1], target_samples[1]])
    x_min, x_max = all_x.min()-0.5, all_x.max()+0.5
    y_min, y_max = all_y.min()-0.5, all_y.max()+0.5
    bins = 50
    x_edges = np.linspace(x_min, x_max, bins+1)
    y_edges = np.linspace(y_min, y_max, bins+1)

    traj_hist, _, _ = np.histogram2d(traj_xy[:,0], traj_xy[:,1], bins=[x_edges, y_edges], density=True)
    target_hist, _, _ = np.histogram2d(target_samples[0], target_samples[1], bins=[x_edges, y_edges], density=True)

    # Avoid zeros for log color if needed later
    # Create figure
    fig_hm, axes_hm = plt.subplots(1,2, figsize=(10,4))
    im0 = axes_hm[0].imshow(traj_hist.T, origin='lower', extent=[x_min, x_max, y_min, y_max], aspect='equal', cmap='viridis')
    axes_hm[0].set_title('Best Trajectory Occupancy')
    axes_hm[0].set_xlabel('X')
    axes_hm[0].set_ylabel('Y')
    plt.colorbar(im0, ax=axes_hm[0], fraction=0.046, pad=0.04)

    im1 = axes_hm[1].imshow(target_hist.T, origin='lower', extent=[x_min, x_max, y_min, y_max], aspect='equal', cmap='viridis')
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
    ax_loss.set_ylabel('Minimum Cost')
    ax_loss.set_title('Training Loss Over Time')
    ax_loss.grid(True)
    plt.tight_layout()
    # plt.savefig('/Users/saturnv/sandbox/distribution_steering_MDP/ergodic/loss_curve.png', dpi=150, bbox_inches='tight')
    plt.show()

    

def main():
    """Main test function."""

    system, policy, initial_state, action, info = run_open_loop()
    visualize_results(system, policy, initial_state, action, info)

if __name__ == "__main__":
    main()
