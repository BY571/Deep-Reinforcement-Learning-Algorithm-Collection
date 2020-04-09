# Deep-Reinforcement-Learning


![Logo](/imgs/web-3706562_640.jpg)

In this repository I document my path of learning about Reinforcement Learning.
The goal is to understand, implement and document fundamental algorithms of Deep Reinforcement Learning.
Starting with Q-Learning (Table) going to Deep Q-Learning with several extensions (Experience Replay, Action Selection...) and in the end approching state of the art Deep Reinforcement Learning Algorithms like A3C, A2C, PPO, TRPO, DDPG, D4PG and SAC. As well covering topics of Multi-Agent DRL and Black-Box Optimization. The current focus lays on model free algorithms but in the future i will try to study and implement some model based algorithms.

All algorithms are implemented in Python with the Deep Learning library Pytorch and tested in the Open Ai gym.

Below a list of Jupyter Notebooks with implementations

# Value Based / Offline Methods
## Discrete Action Space

- [Q-Learning](Q_Learning) &emsp;&emsp;&emsp; [Source/Paper](/Paper/DQN.pdf)

- [DQN](https://github.com/BY571/Reinforcement-Learning/tree/master/Deep%20Q_Learning) &emsp;&emsp;&emsp;&emsp; [Paper](/Paper/DQN.pdf)

- [Double DQN](https://github.com/BY571/Reinforcement-Learning/tree/master/Double%20DQN) &emsp;&emsp;&emsp;&emsp; [Paper](/Paper/Double_DQN.pdf)

- [Dueling DQN](https://github.com/BY571/DQN-Atari-Agents) &emsp;&emsp;&emsp;&emsp; [Paper](/Paper/Dueling.pdf)

- [N-Step DQN](https://github.com/BY571/DQN-Atari-Agents)

- [Categorical DQN - C51](https://github.com/BY571/DQN-Atari-Agents) &emsp;&emsp;&emsp;&emsp;[Paper](https://github.com/BY571/Reinforcement-Learning/blob/master/Paper/Distributional%20DQN.pdf)

- [Noisy DQN](https://github.com/BY571/DQN-Atari-Agents)
&emsp;&emsp;&emsp;&emsp; [Paper](/Paper/Noisy_networks.pdf)

- [Rainbow](https://github.com/BY571/DQN-Atari-Agents)
&emsp;&emsp;&emsp;&emsp;[Paper] (https://arxiv.org/pdf/1710.02298.pdf)

## Continuous Action Space

- [DDPG](https://github.com/BY571/Udacity-DRL-Nanodegree-P2)
[Source/Paper]


- D4PG [TODO]
[Source/Paper]

- [Twin Delayed DDPG (TD3)](https://github.com/BY571/Reinforcement-Learning/blob/master/ContinousControl/TD3_conti.ipynb)
&emsp;&emsp;&emsp;&emsp;[Paper](https://github.com/BY571/Reinforcement-Learning/blob/master/Paper/TD3.pdf)

- [Soft Actor Critic (SAC-newest 2019 version)](https://github.com/BY571/Reinforcement-Learning/blob/master/ContinousControl/SAC.ipynb)
&emsp;&emsp;&emsp;&emsp;[Paper](https://github.com/BY571/Reinforcement-Learning/blob/master/Paper/SAC_2019.pdf)
_________________________________________________
# Policy Based / Online Methods
## Discrete Action Space


- [Sarsa](https://github.com/BY571/Reinforcement-Learning/blob/master/Temporal%20Difference%20(Sarsa%2C%20Sarsamax%2C%20Expeted%20Sarsa)/Temporal_Difference.ipynb)
[Source/Paper]


- [Vanilla Policy Gradient](https://github.com/BY571/Reinforcement-Learning/blob/master/Policy%20Gradient%20Algorithms/Policy_Gradien_%2B_Baseline_mean.ipynb) [+LSTM](https://github.com/BY571/Reinforcement-Learning/blob/master/Policy%20Gradient%20Algorithms/PolicyGradient_LSTM.ipynb)
[Source/Paper]


- A2C
[Paper](/Paper/A3C.pdf)

- A2C with gae* [TODO]

- A2C multi environment


- PPO
[Paper](/Paper/PPO.pdf)

- PPO with gae*

- [PPO with gae and curiosity driven exploration (single, digit inputs)](https://github.com/BY571/Reinforcement-Learning/blob/master/PPO_gae_curios.ipynb) [Paper](/Paper/)

- PPO multi environment


## Continuous Action Space

- [A2C](https://github.com/BY571/Reinforcement-Learning/blob/master/ContinousControl/A2C_conti_seperate_networks.ipynb)

- A2C with gae* [TODO]

- [A2C multi environment](https://github.com/BY571/Reinforcement-Learning/blob/master/ContinousControl/A2C_continuous_multienv.ipynb)


- [PPO](https://github.com/BY571/Reinforcement-Learning/blob/master/ContinousControl/PPO_unity_Crawler.ipynb)

- [PPO with gae*](https://github.com/BY571/Reinforcement-Learning/blob/master/ContinousControl/ROBOSCHOOL_PPO_GAE.ipynb)[PPO with gae multi](https://github.com/BY571/Reinforcement-Learning/blob/master/ContinousControl/PPO_conti_gae_multi.ipynb)

- [PPO+curiosity&single](https://github.com/BY571/Reinforcement-Learning/blob/master/ContinousControl/PPO_conti_gae_curios.ipynb)[+curiosity&multi](https://github.com/BY571/Reinforcement-Learning/blob/master/PPO_conti_gae_curio_multi.ipynb)

- [PPO multi environment](https://github.com/BY571/Reinforcement-Learning/blob/master/ContinousControl/PPO_unity_Crawler.ipynb)




gae* = Generalized Advanted Estimation [Source](/Paper/GAE.pdf)
________________________________________________

# Upside-Down-Reinforcement-Learning
Discrete action space implementation of [⅂ꓤ](https://github.com/BY571/Upside-Down-Reinforcement-Learning)


# Black-Box Optimization

- [Evolution Strategies]() [with mulit processing](https://github.com/BY571/Reinforcement-Learning/blob/master/Black-Box%20Optimization/Evolution_Strategies_parallel+novelty/README.md) [and novelty search](https://github.com/BY571/Reinforcement-Learning/blob/master/Black-Box%20Optimization/Evolution_Strategies_parallel+novelty/README.md)

- [Genetic Algorithm - GARNE](https://github.com/BY571/GARNE-Genetic-Algorithm-with-Recurrent-Network-and-Novelty-Exploration/blob/master/README.md) 
  - Genetic Algorithm implementation with LSTM, Multiprocessing over several CPUs and Novelty Search for Exploration
__________________________________________
# Multi-Agent Deep Reinforcement Learning

- [Multi-Agent-DDPG](https://github.com/BY571/Udacity-DRL-Nanodegree-P3-Multiagent-RL-)

# Hyperparameter Tuning

Gridsearch

Random Forest [TODO]

Genetic Algorithm [TODO]

====================================


