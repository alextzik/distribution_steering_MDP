"""
    The implementation of the proposed algorithm is contained here.

"""

import math
import random
import numpy as np

# The class used to model the nodes in MCTS
class Node:
    """
        A node is determined by:
         - its state: set of particles, instance of class State
         - its parent node: instance of class Node
         - the action followed at the parent node to reach it: list[K, b]
         - its children: list[Node] 
         - the number of times it is visited
         - its Q-value 
         - a count necessary for the action progressive widening in MCTS.
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
            to encourage search in the good areas of the action space
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
            - iterations: num of simulations to carry out
            - depth: max depth of the search tree
            - c_param: the UCB heuristic parameter that encourages exploration
            - the action_prog_widen parameters:
                - self.ka
                - self.ao
    """
    def __init__(self, iterations=1000):
        self.iterations = iterations

        # tree depth
        self.depth = 8

        # UCB parameter
        self.c_param = 100

        # action progressive widening
        self.ka = 10.
        self.ao = 0.5

    """
        plan() takes as argument an initial_state (instance of State).
        It creates the root node of the tree and performs self.iterations
            simulations from the root node.
        It then computes the child node of the root with the highest Q-value and outputs the action that led to that 
            as the best action to take at the root node.
    """
    def plan(self, initial_state):
        root = Node(initial_state)

        for _ in range(self.iterations):
            self._simulate(root, self.depth)

        choices_weights = [child.value
                                for child in root.children]
        
        print(choices_weights)
        
        return root.children[choices_weights.index(max(choices_weights))].action


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

        return node.children[choices_weights.index(max(choices_weights))]
    
    def _reward(self, node, next_node):
        return 5.

# Example game state class
class State:
    def __init__(self):
        self.num_samples = None
        self.samples = None
        self.dim_state = None

    def set(self, samples):
        self.num_samples = samples.shape[1]
        self.samples = samples
        self.dim_state = samples.shape[0]

    def sample(self, mean, covariance, num_samples):
        self.num_samples = num_samples
        self.samples = np.random.multivariate_normal(mean=mean, cov=covariance, size=self.num_samples).T
        self.dim_state = self.samples.shape[0]

    def transition(self, controller):
        us = controller[0]@self.samples + controller[1]

        A = np.eye(self.dim_state)
        B = 2*np.eye(self.dim_state)

        next_samples = A@self.samples + B@us
        next_state = State()
        next_state.set(next_samples)

        return next_state

state = State()
state.sample(mean = np.array([1, 1]), covariance=np.eye(2), num_samples=100)
K = np.random.normal(size=(2,2))
b = np.random.normal(size=(2,1))
control = [K, b]

new_state = state.transition(control)

# Example usage
mcts = MCTS(iterations=1000)
next_action = mcts.plan(state)
print(next_action)