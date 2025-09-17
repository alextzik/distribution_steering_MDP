#!/usr/bin/env python3
"""
Test script for DescentLQR policy on batched unicycle system.
"""

import torch
import matplotlib.pyplot as plt
import numpy as np
from system import UnicycleSystem
from policy import DescentLQRPolicy

def setup():
    """Create experiment setup with system and policy."""
    
    # System parameters
    dt = 0.1
    max_velocity = 2.0
    max_angular_velocity = 2.0
    noise_xy_std = 0.01
    noise_theta_std = 0.01
    
    # Create unicycle system
    system = UnicycleSystem(
        dt=dt,
        max_velocity=max_velocity,
        max_angular_velocity=max_angular_velocity,
        noise_xy_std=noise_xy_std,
        noise_theta_std=noise_theta_std
    )
    
    # Policy parameters
    horizon = 25
    num_restarts = 50
    num_gradient_steps = 300
    
    # LQR cost matrices
    state_dim = system.get_state_dim()
    action_dim = system.get_action_dim()
    
    # Q matrix: penalize deviation from target state
    Q = torch.eye(state_dim) * 10.0  # Higher weight on position errors
    Q[2, 2] = 1.0  # Lower weight on heading error
    
    # R matrix: penalize large actions
    R = torch.eye(action_dim) * 0.1
    
    # Target state: reach origin with zero heading
    target_state = torch.tensor([0.0, 0.0, 0.0])
    
    # Create policy
    policy = DescentLQRPolicy(
        system=system,
        horizon=horizon,
        Q=Q,
        R=R,
        target_state=target_state,
        soft_constraint_count_penalty=100.0,
        soft_constraint_quad_penalty=100.0,
        num_restarts=num_restarts,
        num_gradient_steps=num_gradient_steps,
        optimizer_class=torch.optim.AdamW,
        optimizer_kwargs={'lr': 0.01}
    )
    
    return system, policy

def run_open_loop():
    """Test policy on a single initial state."""
    print("Setting up test...")
    system, policy = setup()
    
    # Initial state: start at (2, 2) with 45 degree heading
    initial_state = torch.tensor([2.0, 2.0, np.pi/4])
    
    print(f"Initial state: {initial_state}")
    print(f"Target state: {policy.target_state}")
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
    
    # Extract best trajectory
    best_trajectory = info['best_trajectory']  # (T+1, state_dim)
    best_actions = info['best_actions']  # (T, action_dim)
    
    # Convert to numpy for plotting
    trajectory_np = best_trajectory.detach().numpy()
    actions_np = best_actions.detach().numpy()
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Plot 1: Trajectory in x-y plane
    ax1 = axes[0, 0]
    ax1.plot(trajectory_np[:, 0], trajectory_np[:, 1], 'b-o', linewidth=2, markersize=4)
    ax1.plot(initial_state[0], initial_state[1], 'go', markersize=10, label='Start')
    ax1.plot(policy.target_state[0], policy.target_state[1], 'ro', markersize=10, label='Target')
    ax1.set_xlabel('X Position')
    ax1.set_ylabel('Y Position')
    ax1.set_title('Trajectory in X-Y Plane')
    ax1.legend()
    ax1.grid(True)
    ax1.axis('equal')
    
    # Plot 2: Actions over time
    ax2 = axes[0, 1]
    time_steps = np.arange(len(actions_np))
    ax2.plot(time_steps, actions_np[:, 0], 'b-', label='Linear velocity (v)')
    ax2.plot(time_steps, actions_np[:, 1], 'r-', label='Angular velocity (ω)')
    ax2.set_xlabel('Time Step')
    ax2.set_ylabel('Action Value')
    ax2.set_title('Actions Over Time')
    ax2.legend()
    ax2.grid(True)
    
    # Plot 3: State evolution
    ax3 = axes[1, 0]
    time_steps_full = np.arange(len(trajectory_np))
    ax3.plot(time_steps_full, trajectory_np[:, 0], 'b-', label='X position')
    ax3.plot(time_steps_full, trajectory_np[:, 1], 'g-', label='Y position')
    ax3.plot(time_steps_full, trajectory_np[:, 2], 'r-', label='Heading (θ)')
    ax3.set_xlabel('Time Step')
    ax3.set_ylabel('State Value')
    ax3.set_title('State Evolution')
    ax3.legend()
    ax3.grid(True)
    
    # Plot 4: Cost distribution
    ax4 = axes[1, 1]
    costs = info['all_costs'].detach().numpy()
    ax4.hist(costs, bins=20, alpha=0.7, edgecolor='black')
    ax4.axvline(info['best_cost'], color='red', linestyle='--', linewidth=2, label=f'Best cost: {info["best_cost"]:.4f}')
    ax4.set_xlabel('Cost Value')
    ax4.set_ylabel('Frequency')
    ax4.set_title('Cost Distribution Across Restarts')
    ax4.legend()
    ax4.grid(True)
    
    plt.tight_layout()
    plt.savefig('/Users/saturnv/sandbox/distribution_steering_MDP/ergodic/test_results.png', dpi=150, bbox_inches='tight')
    plt.show()

def main():
    """Main test function."""

    system, policy, initial_state, action, info = run_open_loop()
    visualize_results(system, policy, initial_state, action, info)

if __name__ == "__main__":
    main()
