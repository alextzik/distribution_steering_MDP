import abc
import torch
import torch.nn as nn
from system import BatchedSystem
from tqdm import tqdm

class Policy(abc.ABC):
    @abc.abstractmethod
    def action_info(self, state: torch.Tensor) -> tuple[torch.Tensor, any]:
        pass

    def action(self, state: torch.Tensor) -> torch.Tensor:
        action, _ = self.action_info(state)
        return action

class DescentLQRPolicy(Policy):
    def __init__(self, 
        system: BatchedSystem, 
        horizon: int, 
        Q: torch.Tensor, 
        R: torch.Tensor, 
        target_state: torch.Tensor,
        soft_constraint_count_penalty: float=1000., 
        soft_constraint_quad_penalty: float=1000.,
        num_restarts: int=100,
        num_gradient_steps: int=1000,
        optimizer_class: type = torch.optim.AdamW,
        optimizer_kwargs: dict={}):

        self.system = system
        self.horizon = horizon
        self.Q = Q
        self.R = R
        self.target_state = target_state
        self.num_restarts = num_restarts
        self.num_gradient_steps = num_gradient_steps
        self.optimizer_class = optimizer_class
        self.optimizer_kwargs = optimizer_kwargs
        self.soft_constraint_count_penalty = soft_constraint_count_penalty
        self.soft_constraint_quad_penalty = soft_constraint_quad_penalty

    def action_info(self, state: torch.Tensor) -> tuple[torch.Tensor, any]:

        # Initialize actions for trajectory
        actions = torch.stack([self.system.sample_actions(self.num_restarts) for _ in range(self.horizon)]) # (T, num_restarts, action_dim)
        actions.requires_grad_(True)
        
        # Initialize noise for trajectory
        noise = torch.rand(self.horizon, self.num_restarts, self.system.get_state_dim()) # (T, num_restarts, state_dim)
        
        # Initialize optimizer for trajectory
        optimizer = self.optimizer_class([actions], **self.optimizer_kwargs)
        for i in tqdm(range(self.num_gradient_steps)):
            # Build trajectory step by step without in-place operations
            trajectory_list = []
            constraint_violations_list = []
            
            # Initial state
            current_state = state.expand(self.num_restarts, -1)  # Broadcast state to all restarts
            trajectory_list.append(current_state)

            for t in range(self.horizon):
                # Forward pass through system dynamics
                next_states = self.system.step(current_state, actions[t, :, :], noise[t, :, :])
                trajectory_list.append(next_states)
                
                # Compute constraint violations
                constraint_violations_list.append(
                    self.system.constraint_violations(current_state, actions[t, :, :])
                )
                
                # Update current state for next iteration
                current_state = next_states
            
            # Stack into tensors
            trajectory = torch.stack(trajectory_list, dim=0)  # (T+1, num_restarts, state_dim)
            constraint_violations = torch.stack(constraint_violations_list, dim=0)  # (T, num_restarts, constraint_dim)
            
            # Compute cost
            costs = self.cost(trajectory, actions, constraint_violations)
            
            # Backward pass through costs of all trajectories
            if i < self.num_gradient_steps - 1:
                costs.mean().backward()
                optimizer.step()
                optimizer.zero_grad()

        # Get the best action and trajectory, return the immediate action and all the info
        all_actions = actions.detach()
        all_trajectories = trajectory.detach()
        costs = costs.detach()
        best_cost_index = torch.argmin(costs)
        best_actions = all_actions[:, best_cost_index, :]
        best_trajectory = all_trajectories[:, best_cost_index, :]
        immediate_action = best_actions[0]
        info = {
            "all_actions": all_actions,
            "all_trajectories": all_trajectories,
            "all_costs": costs,
            "best_cost_index": best_cost_index,
            "best_actions": best_actions,
            "best_trajectory": best_trajectory,
            "best_cost": costs[best_cost_index]
        }
        return immediate_action, info
    
    def cost(self, trajectory: torch.Tensor, actions: torch.Tensor, constraint_violations: torch.Tensor) -> torch.Tensor:
        """Compute the cost function using the LQR cost and soft constraint penalties. 
        
        Args:
            trajectory: (T+1, num_restarts, state_dim)
            actions: (T, num_restarts, action_dim)
            constraint_violations: (T, num_restarts, constraint_dim)
            
        Returns:
            costs: (num_restarts)
        """
        # LQR cost: stage state cost + stage action cost + final state cost + constraint penalties
        state_diff = trajectory - self.target_state[None, None, :]
        stage_state_costs = torch.einsum('abi,ij,abj->ab', state_diff[:-1], self.Q, state_diff[:-1])
        stage_state_cost = torch.sum(stage_state_costs, dim=0)
        stage_action_costs = torch.einsum('abi,ij,abj->ab', actions, self.R, actions)
        stage_action_cost = torch.sum(stage_action_costs, dim=0)
        
        # Final state cost currently using same Q
        final_state_cost = torch.einsum('bi,ij,bj->b', state_diff[-1], self.Q, state_diff[-1])
        
        # Constraint violations
        constraint_penalty = self.soft_constraint_count_penalty * torch.sum(constraint_violations > 0)
        constraint_penalty += self.soft_constraint_quad_penalty * torch.sum(constraint_violations**2)
        return stage_state_cost + stage_action_cost + final_state_cost + constraint_penalty