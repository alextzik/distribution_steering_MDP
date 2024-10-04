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
import matplotlib.pyplot as plt
from scipy.stats import norm

from typing import Callable

import parameters as pars
from utils import compute_heur_dist, sample_orthogonal_mat

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 15

##########################################################################################
class dynamics:
    """
        This class represents the dynamics equations for the state evolution

        args:
            - dim_state [int]: the dimension of the state
            - dim_input [int]: the dimension of the input
            - dyn_func [function]: propagates a state sample, using an input, according to the dynamics
    """
    def __init__(self, dim_state:int, dim_input:int, dyn_func:Callable[[np.ndarray, np.ndarray], np.ndarray]) -> None:
        self.dim_state = dim_state
        self.dim_input = dim_input
        self.dyn_func  = dyn_func


class target_density:
    def __init__(self, weights:list, means:list, covs:list) -> None:
        """
            This class characterizes the target density as a GMM

            args:
                - weights list[float]: the weight of each component
                - means list[1d np.ndarray]: the means of the components
                - covs list[2d np.ndarray]: the covs of the components
        """

        self.weights = weights
        self.means = means
        self.covs = covs

    def compute_prob_contents(self, qs:np.ndarray, bs:np.ndarray) -> None:
        """
            Compute the prob content of the GMM in the halfspaces dictated by qs and bs

            qs [(dim, num_half-spaces) np.ndarray]: the normal vectors
            bs [(num_halfspaces, 1) np.ndarray]: the offsets
        """
        prob_contents = []
        for i in range(qs.shape[1]):
            prob_content = 0.

            q = qs[:, i].reshape(-1,1)
            b = bs[i, 0]

            for c in range(len(self.weights)):
                r_mean = q.T@self.means[c].reshape(-1,1) + b
                r_cov = q.T@self.covs[c]@q

                prob_content += self.weights[c]*(1-norm.cdf(x=0., loc=r_mean, scale=np.sqrt(r_cov)))

            prob_contents.append(prob_content.item())

        self.prob_contents = prob_contents

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

    next_samples = np.zeros(shape=(state.dim_state, state.num_samples))

    for sample_idx in range(state.num_samples):
        sample = state.samples[:, sample_idx].reshape(-1,1)
        next_samples[:, sample_idx] = dynamics.dyn_func(sample, controller[0]@sample + controller[1]).reshape(-1,)

    next_state = State()
    next_state.set(next_samples)

    return next_state

class Node:

    def __init__(self, state:State, dynamics:dynamics, parent=None, action:list=None):
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
        self.value = 1.

        self.dynamics = dynamics

    def sample_action(self):
        """
        Samples an action that will lead to a new child node
        """

        dim_state = self.action[0].shape[1]
        dim_input = self.action[0].shape[0]
        min_dim = np.minimum(dim_input, dim_state)

        sample_V = sample_orthogonal_mat(dim=dim_state)
        sample_U = sample_orthogonal_mat(dim=dim_input)
        sample_S = np.zeros(shape=(dim_input, dim_state))
        sample_S[0:min_dim, 0:min_dim] = np.random.uniform(low=0., high=0.5, size=(min_dim,))

        # theta = np.random.uniform(low=0., high=2*np.pi)
        # rot_matrix = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        # scale_coeff = np.random.uniform(low=0., high=1.)

        prop_K = sample_U @ sample_S @ sample_V.T 
        # prop_K = scale_coeff*rot_matrix
        prop_b = 3*np.random.standard_normal(size=(dim_input, 1))

        # prop_K = np.zeros(shape=(2,2))
        new_action = [prop_K, prop_b]
        return new_action

class MCTS:
    def __init__(self, target_state:target_density, qs:np.ndarray, bs:np.ndarray, iterations:int=1000):
        """
        The MCTS algorithm is determined by the following parameters:
            - self.iterations: num of simulations to carry out
            - self.target_state: state consisting of samples from the target distribution (instance of State)
            - self.depth: max depth of the search tree
            - self.c_param: the UCB heuristic parameter that encourages exploration
            - the action_prog_widen parameters:
                - self.ka
                - self.ao
            - qs: the normal vectors for the halfspaces in the distance heuristic
            - bs: the offsets for the halfspaces in the distance heuristic
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

        # distance heuristic
        self.qs = qs
        self.bs = bs

    def plan(self, root:Node) -> tuple[tuple[np.ndarray, np.ndarray], float]:
        """
        Deploys MCTS starting from an initial node at the root
            args: 
                root: (instance of class Node)
        It starts at the root node of the tree and performs a number of 
            self.iterations simulations from the root node.
        It then computes the child node of the root with the lowest Q-value (Q here denotes cost) and outputs the action that led to that 
            as the best action to take at the root node along with the value
        """
        for _ in range(self.iterations):
            root.visits += 1
            self._simulate(root, self.depth)

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
            return compute_heur_dist(node.state.samples, self.target_state, self.qs, self.bs)

        next_node = self._action_prog_widen(node)

        r = self._cost(node, next_node)
        next_node.visits += 1
        q = r + self._simulate(next_node, depth-1)
        
        next_node.value += (q-next_node.value)/next_node.visits

        return q


    def _action_prog_widen(self, node:Node) -> Node:
        """
        Selects a controller to branch out with in the MCTS tree, starting at
         node (instance of Node).
        """
        # print(len(node.children), self.ka*node.visits**(self.ao))
        # if len(node.children) <= self.ka*node.visits**(self.ao):
        if len(node.children) <= 100:
            new_action = node.sample_action()
            new_state = transition(node.state, new_action, node.dynamics)
            new_child = Node(new_state, node.dynamics, node, new_action)
            new_child.value = compute_heur_dist(new_child.state.samples, self.target_state, self.qs, self.bs)
            #compute_wasserstein_dist(new_state.samples, self.target_state.means[0], self.target_state.covs[0])
            
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
    def _cost(self, node:Node, next_node:Node):

        state_next = next_node.state

        # res = np.mean(np.linalg.norm(next_node.state.samples, axis=0))
        
        res = compute_heur_dist(state_next.samples, self.target_state, self.qs, self.bs)
        # res = compute_wasserstein_dist(state_next.samples, self.target_state.means[0], self.target_state.covs[0])

        return res
