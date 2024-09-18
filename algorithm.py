"""
    Developed by Alexandros Tzikas
                alextzik@stanford.edu

"""

##########################################################################################
#                                  IMPLEMENTATION OF THE                                 #
#                                   PROPOSED ALGORITHM                                   #       
##########################################################################################

##########################################################################################
# Dependencies
import math
import numpy as np
from sklearn.covariance import EmpiricalCovariance
from tqdm import tqdm
import matplotlib.pyplot as plt
import os

from typing import Callable

import parameters as pars
from utils import two_sample_kl_estimator, compute_wasserstein_dist, plot_level_curves_normal, compute_heur_dist


##########################################################################################
class dynamics:
    """
        This class represents the dynamics equations for the state evolution

        args:
            - dim_state [int]: the dimension of the state
            - dim_input [int]: the dimension of the input
            - dyn_func [function]: propagates a state sample, using an input, according to the dynamics
    """
    def __init__(self, dim_state:int, dim_input, dyn_func:Callable[[np.ndarray, np.ndarray], np.ndarray]) -> None:
        self.dim_state = dim_state
        self.dim_input = dim_input
        self.dyn_func  = dyn_func

class State:
    def __init__(self) -> None:
        """This class contains the set of samples for a given node in the tree search.
        
            pars:
                - dim_state [int]: dimensionality of state
                - num_samples [int]: number of available samples
                - samples [(dim_state, num_samples) np.array]: the samples as columns of an array
        """
        self.dim_state = None
        self.num_samples = None
        self.samples = None

    def set(self, samples:np.ndarray) -> None:
        """
            Sets the state's particle set using a provided particle set
            args:
                - samples [(dim, num_samples) np.array]: contains the samples as columns
        """
        self.dim_state = samples.shape[0]
        self.num_samples = samples.shape[1]
        self.samples = samples

    def sample(self, mean:np.ndarray, covariance:np.ndarray, num_samples:int) -> None:
        """
        Sets the state's particle set by sampling a given Gaussian for a given
        number of samples

        args: 
                - mean [1d np.array]: the mean of the distribution
                - covariance [2d np.ndarray]: the covariance of the distribution
                - num_samples [int]: number of samples
        """
        self.num_samples = num_samples
        self.samples = np.random.multivariate_normal(mean=mean, cov=covariance, size=self.num_samples).T
        self.dim_state = self.samples.shape[0]

def transition(state:State, controller:tuple[np.ndarray, np.ndarray], dynamics:dynamics) -> State:
    """
    Performs the transition from the current state to a new state using the provided
    controller

    args:
            -  state [State]: the current state
            -  controller [tuple]: the matrix and bias for the controller
            -  dynamics [dynamics]: the dynamics instance
    
    return:
            - the next state
    """

    # next_samples = np.zeros(shape=(state.dim_state, state.num_samples))

    dt = 0.1
    A = np.array([[1, dt], [0, 1]])
    B = np.array([[dt, 0], [0, dt]])

    next_samples = A@state.samples + B@(controller[0]@state.samples + controller[1])

    # for sample_idx in range(state.num_samples):
    #     sample = state.samples[:, sample_idx].reshape(-1,1)
    #     next_samples[:, sample_idx] = dynamics.dyn_func(sample, controller[0]@sample + controller[1]).reshape(-1,)

    next_state = State()
    next_state.set(next_samples)

    return next_state

class Node:

    def __init__(self, state, dynamics:dynamics, parent=None, action=None):
        """The class used for the nodes of the MCTS

        args:
            - state: (instance of class State)
            - dynamics [dynamics]: instance of the dynamics
            - parent: (instance of class Node)
            - action: (list[K, b]) [we assume affine controllers]  action followed at the parent node to reach it (list[K, b])
        
        A node is determined by:
            - self.state: the state of the current node -- includes the set of particles (instance of class State)
            - self.parent node: parent node of current node (instance of class Node)
            - self.action: action followed at the parent node to reach it (list[K, b])
            - self.children: child nodes from current node (list[Node])
            - self.visits: number of times node is visited (int)
            - self.value: the node's Q-value (float)
        """
        self.state = state
        self.parent = parent

        self.action = action
        if not(action): # if node is root node
            self.action = [np.zeros(shape=(dynamics.dim_input, dynamics.dim_state)), np.zeros(shape=(dynamics.dim_input, 1))]

        self.children = []
        self.visits = 1 # initialize to 1 to avoid division by 0 in _action_prog_widen()
        self.value = 0.

        self.dynamics = dynamics

    def sample_action(self):
        """
        Samples an action that will lead to a new child node
        """

        K_mean = self.action[0]
        b_mean = self.action[1]

        prop_K = np.random.uniform(low=-1., high=1., size=K_mean.shape)
        prop_b = np.random.uniform(low=-1., high=1., size=b_mean.shape)

        new_K = prop_K
        new_b = prop_b

        new_action = [new_K, new_b]
        return new_action

class MCTS:
    def __init__(self, target_state:State, iterations:int=1000):
        """
        The MCTS algorithm is determined by the following parameters:
            - self.iterations: num of simulations to carry out
            - self.target_state: state consisting of samples from the target distribution (insance of State)
            - self.depth: max depth of the search tree
            - self.c_param: the UCB heuristic parameter that encourages exploration
            - the action_prog_widen parameters:
                - self.ka
                - self.ao
            - the cost normalizing parameters:
                - self.min_reward
                - self.max_reward
        """
        self.iterations = iterations

        self.target_state = target_state
        
        # tree depth
        self.depth = pars.TREE_DEPTH

        # UCB parameter
        self.c_param = pars.C_PARAMETER

        # action progressive widening
        self.ka = pars.APW_KA
        self.ao = pars.APW_A0

        # constants for normalizing cost between 0 and 1
        self.min_cost = pars.MIN_COST
        self.max_cost = pars.MAX_COST

    def plan(self, root) -> tuple[tuple[np.ndarray, np.ndarray], float]:
        """
        Deploys MCTS starting from an initial state at the root
            args: 
                root: (instance of class Node)
        It creates the root node of the tree and performs a number of 
            self.iterations simulations from the root node.
        It then computes the child node of the root with the highest Q-value and outputs the action that led to that 
            as the best action to take at the root node along with the value
        """
        for _ in range(self.iterations):
            self._simulate(root, self.depth)
            root.visits += 1

        choices_weights = [child.value
                                for child in root.children]
                        
        return root.children[choices_weights.index(min(choices_weights))].action, root.children[choices_weights.index(min(choices_weights))]

    def _simulate(self, node:Node, depth:int) -> float:
        """
        Performs one simulation of MCTS and expands the tree. 
        args:
            node (instance of class Node)
            depth (int): how many more levels to go down in the tree

        This is done recursively. The basis of this method is the POMCPOW algorithm from
            "Online algorithms for POMDPs with continuous state, action, and observation spaces"
                Sunberg et al.

        """
        # print(len(node.children))
        if depth == 0:
            return 0

        next_node = self._action_prog_widen(node)

        r = self._cost(node, next_node)
        q = r + self._simulate(next_node, depth-1)
        
        next_node.visits += 1
        next_node.value += (q-next_node.value)/next_node.visits

        return q


    def _action_prog_widen(self, node:Node) -> Node:
        """
        Selects a controller to branch out with in the MCTS tree, starting at
         node (instance of Node).
        """
        # print(len(node.children), self.ka*node.visits**(self.ao))
        if len(node.children) <= self.ka*node.visits**(self.ao):
        # if len(node.children) <= 10:
            new_action = node.sample_action()
            new_state = transition(node.state, new_action, node.dynamics)
            new_child = Node(new_state, node.dynamics, node, new_action)

            node.children.append(new_child)

        choices_weights = [
        child.value  - self.c_param * math.sqrt(math.log(node.visits) / child.visits)
        for child in node.children
        ]

        return node.children[choices_weights.index(min(choices_weights))]
    
    """
        Computes the cost for the transition between nodes node and next_node.
        args:
            node: parent node (instance of Node)
            next_node: next node (instance of Node)
    """
    def _cost(self, node, next_node):
        action = next_node.action

        state_prev = node.state
        state_next = next_node.state

        # KL_div = two_sample_kl_estimator(self.target_state, state_next)
        
        # res = np.mean(np.linalg.norm(next_node.state.samples, axis=0))
        
        # res = compute_wasserstein_dist(state_next.samples, self.target_state.mean, self.target_state.covariance)

        res = compute_heur_dist(state_next.samples, self.target_state.samples)


        # res = np.linalg.norm(state_next.mean-target_state.mean) + np.linalg.norm(state_next.covariance-target_state.covariance)

        # normalize result
        res = (res-self.min_cost) / (self.max_cost-self.min_cost)

        return res
    
##########################################################################################
##########################################################################################
#                                      METHOD DEPLOYMENT                                 #
#                                                                                        #       
##########################################################################################

# Example usage
    
def dyn_func(x, u):
    dt = 0.1
    A = np.array([[1, dt], [0, 1]])
    B = np.array([[dt, 0], [0, dt]])

    x_next = A@x + B@u

    return x_next

dyns = dynamics(2, 2, dyn_func)

num_steps = 50
state = State()
init_mean = np.array([-5, 5])
init_cov = np.eye(2)
state.sample(mean = init_mean, covariance=init_cov, num_samples=1000)
root = Node(state, dyns)

target_state = State()
target_mean = np.array([8, 9])
target_cov = np.array([[2, 1.5], [1.5, 2]])
target_state.sample(mean = target_mean, covariance=target_cov, num_samples=1000)

mcts = MCTS(target_state, iterations=1000)
wasserst_dists = []


# Main Loop
for t in tqdm(range(num_steps)):

    plt.plot(root.state.samples[0, :], root.state.samples[1, :], '*')
    plot_level_curves_normal(target_mean, target_cov, "viridis")
    plot_level_curves_normal(init_mean, init_cov, "viridis")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.xlim(-15, 15)
    plt.ylim(-15, 15)
    
    file_dir = os.path.dirname(os.path.realpath(__file__))
    log_dir = os.path.join(file_dir, "results")
    os.chdir(log_dir)
    plt.savefig(f"step_{t}.png")
    plt.close()

    next_action, next_root = mcts.plan(root)
    root = next_root

    # KLs.append(two_sample_kl_estimator(target_state, state))
    wasserst_dists.append(compute_heur_dist(root.state.samples, mcts.target_state.samples))
    # wasserst_dists.append(compute_wasserstein_dist(root.state.samples, np.mean(mcts.target_state.samples, axis=1), np.cov(mcts.target_state.samples)))

plt.plot(range(num_steps), wasserst_dists)
plt.ylabel("Instantaneous Cost (Wasserstein Distance)")
plt.xlabel("Timestep")
plt.show()