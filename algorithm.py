##########################################################################################
#                                  IMPLEMENTATION OF THE                                 #
#                                   PROPOSED ALGORITHM                                   #       
##########################################################################################

import math
import numpy as np
from sklearn.covariance import EmpiricalCovariance
from tqdm import tqdm
import matplotlib.pyplot as plt
import os

import parameters as pars
from utils import two_sample_kl_estimator, compute_wasserstein_dist, plot_level_curves_normal

# The class used to model the nodes in MCTS
class Node:
    """
        args:
            - state: (instance of class State)
            - parent: (instance of class Node)
            - action: (list[K, b]) [we assume affine controllers]
        A node is determined by:
            - self.state: includes the set of particles (instance of class State)
            - self.parent node: parent node of current node (instance of class Node)
            - self.action: action followed at the parent node to reach it (list[K, b])
            - self.children: child nodes from current node (list[Node])
            - self.visits: number of times node is visited (int)
            - self.value: the node's Q-value (float)
    """
    def __init__(self, state, parent=None, action=None):
        self.state = state
        self.parent = parent

        self.action = action
        if not(action): # if node is root node
            self.action = [np.zeros(shape=(self.state.dim_input, self.state.dim_state)), np.zeros(shape=(self.state.dim_input, 1))]

        self.children = []
        self.visits = 1 # initialize to 1 to avoid division by 0 in _action_prog_widen()
        self.value = 0.

    """
        Samples an action that will lead to a new child node: 
            we choose to sample around the controller used to arrive to the current node 
            because we do not expect consecutive optimal control policies to be 
            very different from each other
    """
    def sample_action(self):
        K_mean = self.action[0]
        b_mean = self.action[1]

        # prop_K = np.random.normal(size=K_mean.shape)
        # prop_b = np.random.normal(size=b_mean.shape)

        prop_K = np.random.uniform(low=-1., high=1., size=K_mean.shape)
        prop_b = np.random.uniform(low=-1., high=1., size=b_mean.shape)

        new_K = prop_K
        new_b = prop_b

        new_action = [new_K, new_b]
        return new_action

# The MCTS algorithm
class MCTS:
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
    def __init__(self, target_state, iterations=1000):
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

    """
        Deploys MCTS starting from an initial state at the root
            args: 
                root: (instance of class Node)
        It creates the root node of the tree and performs a number of 
            self.iterations simulations from the root node.
        It then computes the child node of the root with the highest Q-value and outputs the action that led to that 
            as the best action to take at the root node.
    """
    def plan(self, root):

        for _ in range(self.iterations):
            self._simulate(root, self.depth)
            root.visits += 1

        choices_weights = [child.value
                                for child in root.children]
                        
        return root.children[choices_weights.index(min(choices_weights))].action, root.children[choices_weights.index(min(choices_weights))]

    """
        Performs one simulation of MCTS and expands the tree. 
        args:
            node (instance of class Node)
            depth (int)

        This is done recursively. The basis of this method is the POMCPOW algorithm from
            "Online algorithms for POMDPs with continuous state, action, and observation spaces"
                Sunberg et al.
        
    """
    def _simulate(self, node, depth):
        # print(len(node.children))
        if depth == 0:
            return 0

        next_node = self._action_prog_widen(node)

        r = self._cost(node, next_node)
        q = r + self._simulate(next_node, depth-1)
        
        next_node.visits += 1
        next_node.value += (q-next_node.value)/next_node.visits

        return q

    """
        Selects a controller to branch out with in the MCTS tree, starting at
         node (instance of Node).
    """
    def _action_prog_widen(self, node):
        # print(len(node.children), self.ka*node.visits**(self.ao))
        if len(node.children) <= self.ka*node.visits**(self.ao):
        # if len(node.children) <= 10:
            new_action = node.sample_action()
            new_state = node.state.transition(new_action)
            new_child = Node(new_state, node, new_action)

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
        
        res = compute_wasserstein_dist(state_next.samples, self.target_state.mean, self.target_state.covariance)

        # res = np.linalg.norm(state_next.mean-target_state.mean) + np.linalg.norm(state_next.covariance-target_state.covariance)

        # normalize result
        res = (res-self.min_cost) / (self.max_cost-self.min_cost)

        return res
    
# Example game state class
class State:
    def __init__(self):
        self.dim_state = None
        self.dim_input = 2
        self.num_samples = None
        self.samples = None
        self.mean = None
        self.covariance = None

    """
        Sets the state's particle set using a provided particle set
        args:
            - samples: np.array[dim, num_samples] containing the samples as columns
    """
    def set(self, samples):
        self.dim_state = samples.shape[0]
        self.num_samples = samples.shape[1]
        self.samples = samples

        self.mean = np.mean(samples, axis=1)
        
        self.covariance = EmpiricalCovariance(assume_centered=False).fit(samples.T).covariance_

    """
        Sets the state's particle set by sampling a given Gaussian for a given
        number of samples
    """
    def sample(self, mean, covariance, num_samples):
        self.num_samples = num_samples
        self.samples = np.random.multivariate_normal(mean=mean, cov=covariance, size=self.num_samples).T
        self.dim_state = self.samples.shape[0]

        self.mean = mean
        self.covariance = covariance

    """
        Performs the transition from the current state to a new state using the provided
        controller
    """
    def transition(self, controller):
        us = np.matmul(controller[0], self.samples) + controller[1]

        # A = np.eye(self.dim_state)
        # B = 0.1*np.eye(self.dim_state)
        dt = 0.1
        A = np.array([[1, dt], [0, 1]])
        B = np.array([[dt, 0], [0, dt]])

        next_samples = np.matmul(A, self.samples) + np.matmul(B, us)
        next_state = State()
        next_state.set(next_samples)

        return next_state


##########################################################################################
##########################################################################################
#                                      METHOD DEPLOYMENT                                 #
#                                                                                        #       
##########################################################################################

# Example usage
num_steps = 50
state = State()
init_mean = np.array([-5, 5])
init_cov = np.eye(2)
state.sample(mean = init_mean, covariance=init_cov, num_samples=1000)
root = Node(state)

target_state = State()
target_mean = np.array([8, 9])
target_cov = np.array([[2, 1.5], [1.5, 2]])
target_state.sample(mean = target_mean, covariance=target_cov, num_samples=1000)

mcts = MCTS(target_state, iterations=10000)
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
    wasserst_dists.append(compute_wasserstein_dist(root.state.samples, mcts.target_state.mean, mcts.target_state.covariance))

plt.plot(range(num_steps), wasserst_dists)
plt.ylabel("Instantaneous Cost (Wasserstein Distance)")
plt.xlabel("Timestep")
plt.show()