# Project 3: Continuous Control


## Introduction

The purpose of this script is to train a Reinforcement Learning agent for the Tennis environment. The purpose of both players is to keep a ball in play as long as possible. 

The tasks in this environment are episodic and end with the ball hitting the ground.
Rewards of +0.1 are given for an agent hitting the ball over the net, whereas a reward of -0.01 is given for missing a ball. 
Each of the two agents receive a local state as 24 dimensional vector representing position and velocity of the ball. 
Each agent interacts with the environment by means of two inputs representing a movement to or from the net, and jumping.

The training is carried out in a distributed manner by means of [deep deterministic policy gradient (DDPG)](https://arxiv.org/abs/1509.02971).

The target is to reach an average score of 0.5 over past 100 episodes, where at each time instant the maximum score from the two agents is considered.

## Getting Started

1. Set up a Python 3.6 environment including the following packages:
    - `Torch 0.4.0`
    - `unityagents 0.4.0`
2. Download the environment
    - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux.zip)
    - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis.app.zip)
    - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86.zip)
    - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86_64.zip)     
3. Place the file into the root folder, and unzip (or decompress) it.


## Instructions

Executing `Tennis.py` will start the training of the agent. 
It will output a plot of the averaged score over 100 consecutive time frames, where each time instant of a score is taken to be the maximum of the accumulated rewards of both agents. 
Resulting weights of the trained actor and critic will be stored in the files `checkpoint_*.pth`.

## Sources

The implementation builds up on a code framework provided by [Udacity's Reinforcement Learning Exprert Nano degree](https://www.udacity.com/course/deep-reinforcement-learning-nanodegree--nd893).
It also uses the `Tennis`-environment of [Unity](https://unity.com/de/products/machine-learning-agents).
The implemented actor-critic reinforcement learning method implemented is [deep deterministic policy gradient (DDPG)](https://arxiv.org/abs/1509.02971).