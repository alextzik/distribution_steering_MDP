##########################################################################################
#                                  IMPLEMENTATION OF THE                                 #
#                                   PROPOSED ALGORITHM                                   #       
##########################################################################################

import math
import numpy as np
from tqdm import tqdm
from utils import two_sample_kl_estimator
import matplotlib.pyplot as plt

# The class used to model the nodes in MCTS
class Node:
    """
        args:
            - state: (instance of class State)
            - parent: (instance of class Node)
            - action: (list[K, b])
        A node is determined by:
            - self.state: set of particles (instance of class State)
            - self.parent node: parent node of current node (instance of class Node)
            - self.action: action followed at the parent node to reach it (list[K, b])
            - self.children: child nodes from current node (list[Node])
            - self.visits: number of times node is visited (int)
            - self.value: the node's Q-value (float)
            - self.count: count necessary for the action progressive widening in MCTS (int)
    """
    def __init__(self, state, parent=None, action=None):
        self.state = state
        self.parent = parent

        self.action = action
        if not(action): # if node is root node
            self.action = [np.zeros(shape=(self.state.dim_state, self.state.dim_state)), np.zeros(shape=(self.state.dim_state, 1))]

        self.children = []
        self.visits = 1 # initialize to 1 to avoid division by 0 in _action_prog_widen()
        self.value = 0.
        self.count = 0

    """
        Samples an action that will lead to a new child node: 
            we choose to sample around the controller used to arrive to the current node 
            because we do not expect consecutive optimal control functions to be 
            very different from each other
    """
    def sample_action(self):
        K_mean = self.action[0]
        b_mean = self.action[1]

        new_K = K_mean + np.random.normal(size=K_mean.shape)
        new_b = b_mean + np.random.normal(size=b_mean.shape)

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
            - self.target_samples: samples from the target distribution to which we will want to 
                                    reduce the KL divergence by choosing the controllers
                                    (np.array [dim, n])
    """
    def __init__(self, target_state, iterations=1000):
        self.iterations = iterations

        self.target_state = target_state
        
        # tree depth
        self.depth = 8

        # UCB parameter
        self.c_param = 10

        # action progressive widening
        self.ka = 2.
        self.ao = 0.5

    """
        Deploys MCTS starting from an initial state at the root
            args: 
                initial_state: (instance of class State)
        It creates the root node of the tree and performs a number of 
            self.iterations simulations from the root node.
        It then computes the child node of the root with the highest Q-value and outputs the action that led to that 
            as the best action to take at the root node.
    """
    def plan(self, initial_state):
        root = Node(initial_state)

        for _ in range(self.iterations):
            self._simulate(root, self.depth)

        choices_weights = [child.value
                                for child in root.children]
                
        return root.children[choices_weights.index(max(choices_weights))].action

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
        if depth == 0:
            return 0
        next_node = self._action_prog_widen(node)

        r = self._reward(node, next_node)
        q = r + self._simulate(next_node, depth-1)

        next_node.visits += 1
        node.visits += 1
        next_node.value += (q-next_node.value)/next_node.visits

        return q

    """
        Selects a controller to branch out with in the MCTS tree, starting at
         node (instance of Node).
    """
    def _action_prog_widen(self, node):
        if node.count <= self.ka*node.visits**(self.ao):
            new_action = node.sample_action()
            new_state = node.state.transition(new_action)
            new_child = Node(new_state, node, new_action)

            node.children.append(new_child)
            node.count += 1

        choices_weights = [
        child.value  + self.c_param * math.sqrt(math.log(node.visits) / child.visits)
        for child in node.children
        ]

        # print([
        # child.value 
        # for child in node.children
        # ])

        # print([
        # self.c_param * math.sqrt(math.log(node.visits) / child.visits)
        # for child in node.children
        # ])

        # print()

        return node.children[choices_weights.index(max(choices_weights))]
    
    """
        Computes the reward for the transition between nodes node and next_node.
        args:
            node: parent node (instance of Node)
            next_node: next node (instance of Node)
    """
    def _reward(self, node, next_node):
        action = next_node.action

        state_prev = node.state
        state_next = next_node.state

        KL_div = two_sample_kl_estimator(self.target_state, state_next)
        
        return -KL_div

# Example game state class
class State:
    def __init__(self):
        self.num_samples = None
        self.samples = None
        self.dim_state = None

    """
        Sets the state's particle set using a provided particle set
    """
    def set(self, samples):
        self.num_samples = samples.shape[1]
        self.samples = samples
        self.dim_state = samples.shape[0]

    """
        Sets the state's particle set by sampling a given Gaussian for a given
        number of samples
    """
    def sample(self, mean, covariance, num_samples):
        self.num_samples = num_samples
        self.samples = np.random.multivariate_normal(mean=mean, cov=covariance, size=self.num_samples).T
        self.dim_state = self.samples.shape[0]

    """
        Performs the transition from the current state to a new state using the provided
        controller
    """
    def transition(self, controller):
        us = controller[0]@self.samples + controller[1]

        A = np.eye(self.dim_state)
        B = 2*np.eye(self.dim_state)

        next_samples = A@self.samples + B@us
        next_state = State()
        next_state.set(next_samples)

        return next_state


##########################################################################################
##########################################################################################
#                                      METHOD DEPLOYMENT                                 #
#                                                                                        #       
##########################################################################################

# Example usage
state = State()
state.sample(mean = np.array([1, 1]), covariance=np.eye(2), num_samples=100)

target_state = State()
target_state.sample(mean = np.array([-1, 1]), covariance=3*np.eye(2), num_samples=100)

mcts = MCTS(target_state, iterations=100)
KLs = [two_sample_kl_estimator(target_state, state)]
# Main Loop
for t in tqdm(range(100)):
    next_action = mcts.plan(state)
    state = state.transition(next_action)
    KLs.append(two_sample_kl_estimator(target_state, state))

plt.plot(range(101), KLs)
plt.show()