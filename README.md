# Deep-Reinforcement-Learning


![Logo](/imgs/web-3706562_640.jpg)

In this repository I document my path of learning about Reinforcement Learning.
The goal is to understand, implement and document fundamental algorithms of Deep Reinforcement Learning.
Starting with Q-Learning (Table) going to Deep Q-Learning with several extensions (Experience Replay, Action Selection...) and in the end approching state of the art Deep Reinforcement Algorithms like A3C, A2C, PPO, TRPO, DDPG, D4PG and Multi-Agent DRL.
The algorithms are implemented in Python with the Deep Learning library Pytorch and tested in the Open Ai gym.

Below a list of Jupyter Notebooks with implementations

# Value Based / Offline Methods
## Discrete Action Space

- [Q-Learning](Q_Learning) &emsp;&emsp;&emsp; [Source/Paper](/Paper/DQN.pdf)

- [DQN](https://github.com/BY571/Reinforcement-Learning/tree/master/Deep%20Q_Learning) &emsp;&emsp;&emsp;&emsp; [Paper](/Paper/DQN.pdf)

- [Double DQN](https://github.com/BY571/Reinforcement-Learning/tree/master/Double%20DQN) &emsp;&emsp;&emsp;&emsp; [Paper](/Paper/Double_DQN.pdf)

- [Dueling DQN](https://github.com/BY571/Reinforcement-Learning/tree/master/Dueling%20Deep%20Q-Network) &emsp;&emsp;&emsp;&emsp; [Paper](/Paper/Dueling.pdf)

Distributional DQN [TODO]
[Source/Paper]

Noisy DQN [TODO]
[Source/Paper]

Rainbow [TODO]
[Source/Paper]

# Policy Based / Online Methods
## Discrete Action Space


Sarsa
[Source/Paper]


Vanilla Policy Gradient
[Source/Paper]


A2C
[Paper](/Paper/A3C.pdf)

A2C with gae* [TODO]

A2C multi environment


PPO
[Paper](/Paper/PPO.pdf)

PPO with gae*

PPO multi environment


## Continuous Action Space

[A2C](https://github.com/BY571/Reinforcement-Learning/blob/master/ContinousControl/A2C_conti_seperate_networks.ipynb)

A2C with gae* [TODO]

[A2C multi environment](https://github.com/BY571/Reinforcement-Learning/blob/master/ContinousControl/A2C_continuous_multienv.ipynb)


[PPO](https://github.com/BY571/Reinforcement-Learning/blob/master/ContinousControl/PPO_unity_Crawler.ipynb)

PPO with gae*

[PPO multi environment](https://github.com/BY571/Reinforcement-Learning/blob/master/ContinousControl/PPO_unity_Crawler.ipynb)




[DDPG](https://github.com/BY571/Udacity-DRL-Nanodegree-P2)
[Source/Paper]


D4PG [TODO]
[Source/Paper]

gae* = Generalized Advanted Estimation [Source](/Paper/GAE.pdf)
________________________________________________

# Multi-Agent Deep Reinforcement Learning

[Multi-Agent-DDPG](https://github.com/BY571/Udacity-DRL-Nanodegree-P3-Multiagent-RL-)

# Hyperparameter Tuning

Gridsearch

Random Forest [TODO]

Genetic Algorithm [TODO]

====================================


