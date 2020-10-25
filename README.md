# Deep-Reinforcement-Learning


![Logo](/imgs/web-3706562_640.jpg)

Collection of Deep Reinforcement Learning Algorithms in PyTorch.


Below a list of Jupyter Notebooks with implementations

# Value Based / Offline Methods
## Discrete Action Space

- [Q-Learning](Q_Learning) &emsp;&emsp;&emsp; [Source/Paper](/Paper/DQN.pdf)

- [DQN](https://github.com/BY571/Reinforcement-Learning/tree/master/Deep%20Q_Learning) &emsp;&emsp;&emsp;&emsp; [Paper](/Paper/DQN.pdf)

- [Double DQN](https://github.com/BY571/Reinforcement-Learning/tree/master/Double%20DQN) &emsp;&emsp;&emsp;&emsp; [Paper](/Paper/Double_DQN.pdf)

- [Dueling DQN](https://github.com/BY571/DQN-Atari-Agents) &emsp;&emsp;&emsp;&emsp; [Paper](/Paper/Dueling.pdf)

- [N-Step DQN](https://github.com/BY571/DQN-Atari-Agents)

- [Noisy DQN](https://github.com/BY571/DQN-Atari-Agents)
&emsp;&emsp;&emsp;&emsp; [Paper](/Paper/Noisy_networks.pdf)

- [Rainbow](https://github.com/BY571/DQN-Atari-Agents)
&emsp;&emsp;&emsp;&emsp;[Paper](https://arxiv.org/pdf/1710.02298.pdf)

## Distributional RL 

- [Categorical DQN - C51](https://github.com/BY571/DQN-Atari-Agents) &emsp;&emsp;&emsp;&emsp;[Paper](https://github.com/BY571/Reinforcement-Learning/blob/master/Paper/Distributional%20DQN.pdf)

- [QR-DQN](https://github.com/BY571/QR-DQN)

- [IQN](https://github.com/BY571/IQN-and-Extensions)

- [FQF](https://github.com/BY571/FQF-and-Extensions)


## Continuous Action Space

- [NAF - Normalized Advantage Function](https://github.com/BY571/Normalized-Advantage-Function-NAF-)

-[Soft-DQN] TODO
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

______________________________________________

# Actor-Critic Algorithms 

- [DDPG](https://github.com/BY571/Udacity-DRL-Nanodegree-P2)
[Source/Paper]

- [D4PG](https://github.com/BY571/D4PG)
[Source/Paper](https://arxiv.org/pdf/1804.08617.pdf)

- [Twin Delayed DDPG (TD3)](https://github.com/BY571/Reinforcement-Learning/blob/master/ContinousControl/TD3_conti.ipynb)
&emsp;&emsp;&emsp;&emsp;[Paper](https://github.com/BY571/Reinforcement-Learning/blob/master/Paper/TD3.pdf)

- [Soft Actor Critic (SAC-newest 2019 version)](https://github.com/BY571/Reinforcement-Learning/blob/master/ContinousControl/SAC.ipynb)
&emsp;&emsp;&emsp;&emsp;[Paper](https://github.com/BY571/Reinforcement-Learning/blob/master/Paper/SAC_2019.pdf)

________________________________________________

# Upside-Down-Reinforcement-Learning
Discrete and continuous action space implementation of [⅂ꓤ](https://github.com/BY571/Upside-Down-Reinforcement-Learning)

________________________________________________
# Munchausen Reinforcement Learning

Implementierungen von Munchausen RL

- [M-DQN](https://github.com/BY571/Munchausen-RL)

- [M-IQN](https://github.com/BY571/IQN-and-Extensions)

- [M-FQF](https://github.com/BY571/FQF-and-Extensions)

- [M-SAC](https://github.com/BY571/Soft-Actor-Critic-and-Extensions)


________________________________________________

# Model-Based RL

__________________________________________________

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


