import abc
import torch
import math

class BatchedSystem(abc.ABC):
    @abc.abstractmethod
    def step(self, state: torch.Tensor, action: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        """Advance the system state by one time step.
        
        Args:
            state: Current state vector (dim: B, state_dim)
            action: Action vector (dim: B, action_dim)
            noise: Rand noise vector in (0, 1) (dim: B, state_dim)
            
        Returns:
            Next state vector (dim: B, state_dim)

        Note:
            For the system dynamics to be differentiable, the noise that is added to the system should be transformed from (0, 1) to the appropriate range.
        """
        pass

    @abc.abstractmethod
    def get_state_dim(self) -> int:
        """Return the dimension of the state space.
        
        Returns:
            State dimension (int)
        """
        pass

    @abc.abstractmethod
    def get_action_dim(self) -> int:
        """Return the dimension of the action space.
        
        Returns:
            Action dimension (int)
        """
        pass

    @abc.abstractmethod
    def get_constraint_dim(self) -> int:
        """Return the dimension of the constraint space.
        
        Returns:
            Constraint dimension (int)
        """
        pass

    @abc.abstractmethod
    def constraint_violations(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Compute constraint violations for the state and action.
        
        Args:
            state: Input state vector (dim: B, state_dim)
            action: Input action vector (dim: B, action_dim)
            
        Returns:
            Constraint violation vector (positive if violated, zero otherwise) (dim: B, num_constraints)
        """
        pass

    def sample_actions(self, batch_size: int) -> torch.Tensor:
        """Sample an action from the action space.
        
        Returns:
            Action vector (dim: batch_size, action_dim)
        """
        raise NotImplementedError


class UnicycleSystem(BatchedSystem):
    """Unicycle system with state [x, y, θ] and action [v, ω].
    
    State variables:
        x: x-position (m)
        y: y-position (m) 
        θ: heading angle (rad)
        
    Action variables:
        v: linear velocity (m/s)
        ω: angular velocity (rad/s)
    """
    
    def __init__(self, dt: float = 0.1, max_velocity: float = 2.0, max_angular_velocity: float = 2.0, noise_xy_std: float = 0.01, noise_theta_std: float = 0.01):
        """Initialize unicycle system.
        
        Args:
            dt: Time step (s)
            max_velocity: Maximum linear velocity (m/s)
            max_angular_velocity: Maximum angular velocity (rad/s)
        """
        self.dt = dt
        self.max_velocity = max_velocity
        self.max_angular_velocity = max_angular_velocity
        self.noise_xy_std = noise_xy_std
        self.noise_theta_std = noise_theta_std
    
    def step(self, state: torch.Tensor, action: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        """Advance the unicycle state by one time step.
        
        Dynamics:
            ẋ = v * cos(θ)
            ẏ = v * sin(θ)
            θ̇ = ω
            
        Args:
            state: Current state [x, y, θ] (dim: B, 3)
            action: Action [v, ω] (dim: B, 2)
            noise: Uniform noise vector in (0, 1) (dim: B, 3)
            
        Returns:
            Next state [x, y, θ] (dim: 3)
        """
        x, y, theta = state[:, 0], state[:, 1], state[:, 2]
        v, omega = action[:, 0], action[:, 1]
        
        # Convert uniform noise (0,1) to Gaussian noise using torch.erfinv
        # This is the inverse CDF method: Φ^(-1)(u) = sqrt(2) * erfinv(2u - 1)
        gaussian_noise_standard = torch.sqrt(torch.tensor(2.0)) * torch.erfinv(2 * noise - 1)
        gaussian_noise_x = gaussian_noise_standard[:, 0] * self.noise_xy_std
        gaussian_noise_y = gaussian_noise_standard[:, 1] * self.noise_xy_std
        gaussian_noise_theta = gaussian_noise_standard[:, 2] * self.noise_theta_std
        
        # Unicycle dynamics
        x_next = x + v * torch.cos(theta) * self.dt + gaussian_noise_x
        y_next = y + v * torch.sin(theta) * self.dt + gaussian_noise_y
        theta_next = theta + omega * self.dt + gaussian_noise_theta
        
        # Normalize angle to [-π, π]
        theta_next = torch.atan2(torch.sin(theta_next), torch.cos(theta_next))
        
        return torch.stack([x_next, y_next, theta_next], dim=1)
    
    def get_state_dim(self) -> int:
        return 3
    
    def get_action_dim(self) -> int:
        return 2
    
    def get_constraint_dim(self) -> int:
        return 2
    
    def constraint_violations(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Compute constraint violations for the action.
        
        Args:
            action: Input action vector [v, ω] (dim: B, 2)
            
        Returns:
            Constraint violation vector (positive if violated, zero otherwise)
        """
        v, omega = action[:, 0], action[:, 1]
        
        # Velocity constraints
        v_violation = torch.maximum(torch.abs(v) - self.max_velocity, torch.zeros_like(v))
        omega_violation = torch.maximum(torch.abs(omega) - self.max_angular_velocity, torch.zeros_like(omega))
        
        return torch.stack([v_violation, omega_violation], dim=1)

    def sample_actions(self, batch_size: int) -> torch.Tensor:
        """Sample an action from the action space.
        
        Args:
            batch_size: Number of actions to sample
            
        Returns:
            Action vector (dim: batch_size, action_dim)
        """
        v = torch.randn(batch_size) * self.max_velocity
        omega = torch.randn(batch_size) * self.max_angular_velocity
        return torch.stack([v, omega], dim=1)


class BicycleSystem(BatchedSystem):
    """Bicycle system with state [x, y, θ, δ] and action [v, δ_dot].
    
    State variables:
        x: x-position (m)
        y: y-position (m) 
        θ: heading angle (rad)
        δ: steering angle (rad)
        
    Action variables:
        v: linear velocity (m/s)
        δ_dot: steering rate (rad/s)
        
    The bicycle model uses a wheelbase L and assumes the rear wheel follows
    the front wheel with a steering angle δ.
    """
    
    def __init__(self, 
                 dt: float = 0.1, 
                 wheelbase: float = 2.5, 
                 max_velocity: float = 2.0, 
                 max_steering_angle: float = 0.5,
                 max_steering_rate: float = 1.0,
                 noise_xy_std: float = 0.01, 
                 noise_theta_std: float = 0.01,
                 noise_delta_std: float = 0.01):
        """Initialize bicycle system.
        
        Args:
            dt: Time step (s)
            wheelbase: Distance between front and rear axles (m)
            max_velocity: Maximum linear velocity (m/s)
            max_steering_angle: Maximum steering angle (rad)
            max_steering_rate: Maximum steering rate (rad/s)
            noise_xy_std: Standard deviation of position noise (m)
            noise_theta_std: Standard deviation of heading noise (rad)
            noise_delta_std: Standard deviation of steering angle noise (rad)
        """
        self.dt = dt
        self.wheelbase = wheelbase
        self.max_velocity = max_velocity
        self.max_steering_angle = max_steering_angle
        self.max_steering_rate = max_steering_rate
        self.noise_xy_std = noise_xy_std
        self.noise_theta_std = noise_theta_std
        self.noise_delta_std = noise_delta_std
    
    def step(self, state: torch.Tensor, action: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        """Advance the bicycle state by one time step.
        
        Dynamics:
            ẋ = v * cos(θ)
            ẏ = v * sin(θ)
            θ̇ = (v / L) * tan(δ)
            δ̇ = δ_dot
            
        Args:
            state: Current state [x, y, θ, δ] (dim: B, 4)
            action: Action [v, δ_dot] (dim: B, 2)
            noise: Uniform noise vector in (0, 1) (dim: B, 4)
            
        Returns:
            Next state [x, y, θ, δ] (dim: B, 4)
        """
        x, y, theta, delta = state[:, 0], state[:, 1], state[:, 2], state[:, 3]
        v, delta_dot = action[:, 0], action[:, 1]
        
        # Convert uniform noise (0,1) to Gaussian noise using torch.erfinv
        gaussian_noise_standard = torch.sqrt(torch.tensor(2.0)) * torch.erfinv(2 * noise - 1)
        gaussian_noise_x = gaussian_noise_standard[:, 0] * self.noise_xy_std
        gaussian_noise_y = gaussian_noise_standard[:, 1] * self.noise_xy_std
        gaussian_noise_theta = gaussian_noise_standard[:, 2] * self.noise_theta_std
        gaussian_noise_delta = gaussian_noise_standard[:, 3] * self.noise_delta_std
        
        # Bicycle dynamics
        x_next = x + v * torch.cos(theta) * self.dt + gaussian_noise_x
        y_next = y + v * torch.sin(theta) * self.dt + gaussian_noise_y
        theta_next = theta + (v / self.wheelbase) * torch.tan(delta) * self.dt + gaussian_noise_theta
        delta_next = delta + delta_dot * self.dt + gaussian_noise_delta
        
        # Normalize angles to [-π, π]
        theta_next = torch.atan2(torch.sin(theta_next), torch.cos(theta_next))
        delta_next = torch.atan2(torch.sin(delta_next), torch.cos(delta_next))
        
        return torch.stack([x_next, y_next, theta_next, delta_next], dim=1)
    
    def get_state_dim(self) -> int:
        return 4
    
    def get_action_dim(self) -> int:
        return 2
    
    def get_constraint_dim(self) -> int:
        return 3
    
    def constraint_violations(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Compute constraint violations for the state and action.
        
        Args:
            state: Input state vector [x, y, θ, δ] (dim: B, 4)
            action: Input action vector [v, δ_dot] (dim: B, 2)
            
        Returns:
            Constraint violation vector (positive if violated, zero otherwise)
        """
        v, delta_dot = action[:, 0], action[:, 1]
        delta = state[:, 3]
        
        # Velocity constraint
        v_violation = torch.maximum(torch.abs(v) - self.max_velocity, torch.zeros_like(v))
        
        # Steering angle constraint
        delta_violation = torch.maximum(torch.abs(delta) - self.max_steering_angle, torch.zeros_like(delta))
        
        # Steering rate constraint
        delta_dot_violation = torch.maximum(torch.abs(delta_dot) - self.max_steering_rate, torch.zeros_like(delta_dot))
        
        return torch.stack([v_violation, delta_violation, delta_dot_violation], dim=1)

    def sample_actions(self, batch_size: int) -> torch.Tensor:
        """Sample an action from the action space.
        
        Args:
            batch_size: Number of actions to sample
            
        Returns:
            Action vector (dim: batch_size, action_dim)
        """
        v = torch.randn(batch_size) * self.max_velocity
        delta_dot = torch.randn(batch_size) * self.max_steering_rate
        return torch.stack([v, delta_dot], dim=1)

