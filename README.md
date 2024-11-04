# Discrete-Time Distribution Steering using MDPs

The repo contains codebase for the paper "Discrete-Time Distribution Steering using Monte Carlo Tree Search", submitted to 
IEEE Robotics and Automation Letters.

## Summary

We propose an online planning algorithm that solves the discrete-time distribution steering problem. Our algorithm finds an adequate state-feedback controller at every timestep in order to guide
the distribution of the state. At every timestep, our algorithm
builds a tree of trajectories for the sample-based distribution of the state. At every node, various alternative control laws are explored.

We further propose a novel distance metric for the space of distributions. 

## Code Organization

The organization of the code is quite straight-forward. 

* ```algorithm.py```
    * Contains the implementation of our proposed online planning algorithm, along with supporting classes. Our algorithm is implemented as the ```MCTS``` class, which contains Algorithms 2-4 from the paper.

* ```baseline.py```
    * Contains the implementation of the baseline algorithm


* ```utils.py```
    * Contains the implementation of our novel distance metric ```compute_heur_dist``` (Algorithm 1 in paper)

* ```parameters.py```
    * Sets the necessary parameters

* ```run.py```
    * Runs the algorithm and collects the results. In this script, we specify the dynamics, the initial and target state distribution, the half-spaces to be used in the distance metric, and run our algorithm
    for a given number of steps.

## Further Notes

If interested in replicating our paper's results, please just switch to the corresponding branch. 
