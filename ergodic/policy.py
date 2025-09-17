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

class ShootingMPCPolicy(Policy):
    """Abstract base class for policies that use PyTorch optimization to find actions.
    
    Subclasses must implement the batched_cost method that computes costs for multiple
    trajectory candidates in parallel.
    """
    
    def __init__(self, 
                 system: BatchedSystem, 
                 horizon: int, 
                 num_restarts: int=100,
                 num_gradient_steps: int=1000,
                 optimizer_class: type = torch.optim.AdamW,
                 optimizer_kwargs: dict={}):
        self.system = system
        self.horizon = horizon
        self.num_restarts = num_restarts
        self.num_gradient_steps = num_gradient_steps
        self.optimizer_class = optimizer_class
        self.optimizer_kwargs = optimizer_kwargs
    
    @abc.abstractmethod
    def batched_cost(self, trajectory: torch.Tensor, actions: torch.Tensor, constraint_violations: torch.Tensor) -> torch.Tensor:
        """Compute the cost function for multiple trajectory candidates in parallel.
        
        Args:
            trajectory: (T+1, num_restarts, state_dim)
            actions: (T, num_restarts, action_dim)
            constraint_violations: (T, num_restarts, constraint_dim)
            
        Returns:
            costs: (num_restarts)
        """
        pass
    
    def action_info(self, state: torch.Tensor) -> tuple[torch.Tensor, any]:
        # Initialize actions for trajectory
        actions = torch.stack([self.system.sample_actions(self.num_restarts) for _ in range(self.horizon)]) # (T, num_restarts, action_dim)
        actions.requires_grad_(True)
        
        # Initialize noise for trajectory
        noise = torch.rand(self.horizon, self.num_restarts, self.system.get_state_dim()) # (T, num_restarts, state_dim)
        
        # Initialize optimizer for trajectory
        optimizer = self.optimizer_class([actions], **self.optimizer_kwargs)
        training_losses = []
        for i in tqdm(range(self.num_gradient_steps)):
            # Build trajectory step by step without in-place operations
            trajectory_list = []
            constraint_violations_list = []
            
            # Initial state
            current_state = state.expand(self.num_restarts, -1)  # Broadcast state to all restarts
            trajectory_list.append(current_state)

            for t in range(self.horizon):
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
            
            # Compute cost using the abstract method
            costs = self.batched_cost(trajectory, actions, constraint_violations)
            training_losses.append(costs.detach().min().item())
            print(training_losses[-1])
            # Backward pass through costs of all trajectories
            if i < self.num_gradient_steps - 1:
                costs.sum().backward()
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
            "best_cost": costs[best_cost_index],
            "training_losses": training_losses
        }
        return immediate_action, info

class DescentLQRPolicy(ShootingMPCPolicy):
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
        
        # Call parent constructor
        super().__init__(system, horizon, num_restarts, num_gradient_steps, optimizer_class, optimizer_kwargs)
        
        # LQR-specific parameters
        self.Q = Q
        self.R = R
        self.target_state = target_state
        self.soft_constraint_count_penalty = soft_constraint_count_penalty
        self.soft_constraint_quad_penalty = soft_constraint_quad_penalty
    
    def batched_cost(self, trajectory: torch.Tensor, actions: torch.Tensor, constraint_violations: torch.Tensor) -> torch.Tensor:
        """Compute the LQR cost function with soft constraint penalties for multiple trajectory candidates.
        
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

class ErgodicMPCPolicy(ShootingMPCPolicy):
    """Ergodic MPC policy that minimizes distance to a target distribution.
    
    This policy uses a 1D distance calculation based on half-spaces to measure
    how well the trajectory covers the target distribution.
    """
    
    def __init__(self, 
        system: BatchedSystem, 
        horizon: int, 
        target_density_samples: torch.Tensor,
        qs: torch.Tensor,
        bs: torch.Tensor,
        soft_constraint_count_penalty: float=1000., 
        soft_constraint_quad_penalty: float=1000.,
        num_restarts: int=100,
        num_gradient_steps: int=1000,
        optimizer_class: type = torch.optim.AdamW,
        optimizer_kwargs: dict={}):
        """Initialize ergodic MPC policy.
        
        Args:
            system: The system dynamics
            horizon: Planning horizon
            target_density_samples: Target distribution samples (2, num_samples) - x,y coordinates
            qs: Half-space directions (num_halfspaces, 2)
            bs: Half-space biases (num_halfspaces, 1)
            soft_constraint_count_penalty: Penalty for constraint violations (count)
            soft_constraint_quad_penalty: Penalty for constraint violations (quadratic)
            num_restarts: Number of optimization restarts
            num_gradient_steps: Number of gradient descent steps
            optimizer_class: PyTorch optimizer class
            optimizer_kwargs: Optimizer keyword arguments
        """
        # Call parent constructor
        super().__init__(system, horizon, num_restarts, num_gradient_steps, optimizer_class, optimizer_kwargs)
        
        # Ergodic-specific parameters
        self.target_density_samples = target_density_samples  # (2, num_samples)
        self.qs = qs  # (num_halfspaces, 2)
        self.bs = bs  # (num_halfspaces, 1)
        self.soft_constraint_count_penalty = soft_constraint_count_penalty
        self.soft_constraint_quad_penalty = soft_constraint_quad_penalty
        
        # Compute target probability contents from samples
        self.target_prob_contents = self._compute_target_prob_contents()
    
    def _compute_target_prob_contents(self) -> torch.Tensor:
        """Compute target probability contents from target density samples.
        
        Returns:
            target_prob_contents: (num_halfspaces,) - probability content for each half-space
        """
        # Compute linear combinations for target samples
        # target_density_samples is (2, num_samples), qs is (num_halfspaces, 2)
        linear_combinations = torch.matmul(self.qs, self.target_density_samples) + self.bs  # (num_halfspaces, num_samples)
        
        # Count how many target samples are in each half-space (>= 0)
        halfspace_membership = linear_combinations > 0  # (num_halfspaces, num_samples)
        
        # Compute empirical probabilities for each half-space
        num_samples = self.target_density_samples.shape[1]
        target_prob_contents = torch.sum(halfspace_membership, dim=1) / num_samples  # (num_halfspaces,)
        
        return target_prob_contents
    
    def compute_1d_distance(self, trajectory: torch.Tensor) -> torch.Tensor:
        """Compute 1D distance between trajectory and target distribution.
        
        This implements the heuristic distance from utils.py using half-spaces.
        
        Args:
            trajectory: (T+1, num_restarts, state_dim) - trajectory states
            
        Returns:
            distances: (num_restarts) - distance for each trajectory (restart)
        """
        # Extract (T+1, R, 2)
        xy_states = trajectory[:, :, :2]  # (T+1, R, 2)
        # Rearrange to (R, T+1, 2)
        xy_by_restart = xy_states.permute(1, 0, 2)
        # qs: (H, 2), bs: (H,1)
        H = self.qs.shape[0]
        R = xy_by_restart.shape[0]
        T1 = xy_by_restart.shape[1]
        # Compute projections for all half-spaces, restarts, and time points: (H, R, T1)
        # Using einsum: 'hd,rtd->hrt'
        linear_combinations = torch.einsum('hd,rtd->hrt', self.qs, xy_by_restart) + self.bs.unsqueeze(1)  # (H,R,T1)
        # Differentiable half-space membership
        halfspace_membership = torch.sigmoid(100 * linear_combinations)  # (H,R,T1)
        # Empirical probs per restart per half-space: average over time samples
        empirical_probs = halfspace_membership.mean(dim=2)  # (H,R)
        # Target probs shape (H,) -> (H,1)
        target_probs = self.target_prob_contents.unsqueeze(1)  # (H,1)
        prob_diff = torch.abs(empirical_probs - target_probs)  # (H,R)
        distances = prob_diff.mean(dim=0)  # (R,)
        return distances
    
    def batched_cost(self, trajectory: torch.Tensor, actions: torch.Tensor, constraint_violations: torch.Tensor) -> torch.Tensor:
        """Compute the ergodic cost function for multiple trajectory candidates.
        
        Args:
            trajectory: (T+1, num_restarts, state_dim)
            actions: (T, num_restarts, action_dim)
            constraint_violations: (T, num_restarts, constraint_dim)
            
        Returns:
            costs: (num_restarts)
        """
        # Compute ergodic distance to target distribution
        ergodic_cost = self.compute_1d_distance(trajectory)
        
        # Add constraint penalties
        constraint_penalty = self.soft_constraint_count_penalty * torch.sum(constraint_violations > 0, dim=(0, 2))
        constraint_penalty += self.soft_constraint_quad_penalty * torch.sum(constraint_violations**2, dim=(0, 2))
        
        return ergodic_cost + constraint_penalty