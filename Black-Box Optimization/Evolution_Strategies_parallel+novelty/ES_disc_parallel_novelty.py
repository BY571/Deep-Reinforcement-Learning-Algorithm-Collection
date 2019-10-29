# -*- coding: utf-8 -*-
"""
Created on Wed Oct  9 10:24:39 2019

@author: Z0014354
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

import numpy as np
import matplotlib.pyplot as plt
import gym
import torch.multiprocessing as mp
import collections
from collections import deque
import copy
from tensorboardX import SummaryWriter
from sklearn.neighbors import NearestNeighbors

ITERS_PER_UPDATE = 10
NOISE_STD = 0.1 #0.04 higher std leeds to better exploration - more stable learning
LR = 2e-2
PROCESSES_COUNT = 6 # amount of worker default 6
HIDDEN_SIZE = 5   # 6
K_NEIGHBORS = 10
ENV_NAME =  "CartPole-v0"   #"Alien-ram-v0"
RewardsItem = collections.namedtuple('RewardsItem', field_names=['seed', 'pos_reward', 'neg_reward', 'steps'])



class Model(nn.Module):
    def __init__(self, state_size, action_size, idx, hidden_size=HIDDEN_SIZE):
        super(Model, self).__init__()
        self.idx = idx
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        probs = torch.softmax(self.fc3(x), dim=1)
        return probs


def evaluate(env, brain):
    """
    Runs an evaluation on the given brain.
    """
    state = env.reset()
    rewards = 0
    steps = 0
    while True:
        state = torch.from_numpy(state).unsqueeze(0).float()
        probs = brain(state)
        action = probs.max(dim = 1)[1]
        next_state, reward, done, _ = env.step(action.data.numpy()[0])
        rewards += reward
        steps  += 1
        state = next_state
        if done:
            break

    return rewards, steps


def sample_noise(brain):
    """
    Samples noise from a normal distribution in the shape of the brain parameters. Output are two noisy parameters: + noise and - noise (for better and more stable learning!)
    """
    pos = []
    neg = []
    for param in brain.parameters():
        noise_t = torch.tensor(np.random.normal(size = param.data.size()).astype(np.float32))
        pos.append(noise_t)
        neg.append(-noise_t)
    return pos, neg


def eval_with_noise(env, brain, noise, noise_std):
    """
    Evaluates the current brain with added parameter noise

    """
    for p, p_n in zip(brain.parameters(), noise):
        p.data += noise_std * p_n
    r, s = evaluate(env, brain)
    for p, p_n in zip(brain.parameters(), noise):
        p.data -= noise_std * p_n
    return r, s


def worker_func(worker_id, params_queue, rewards_queue, noise_std):
    """
    Worker function that gathers pos and negative rewards for the optimization process and puts them in the rewards_queue with the network parameter seed:
        >> rewards_queue.put(RewardsItem(seed=seed, pos_reward=pos_reward, neg_reward=neg_reward, steps=pos_steps+neg_steps)) <<
    """
    #print("worker: {} has started".format(worker_id))
    env = gym.make(ENV_NAME)
    net = Model(env.observation_space.shape[0], env.action_space.n, "worker")
    net.eval()
    while True:
        params = params_queue.get()
        if params is None:
            break

        # set parameters of the queue
        net.load_state_dict(params)

        for _ in range(ITERS_PER_UPDATE):
            seed = np.random.randint(low=0, high=65535)
            np.random.seed(seed)
            noise, neg_noise = sample_noise(net)
            pos_reward, pos_steps = eval_with_noise(env, net, noise, noise_std)
            neg_reward, neg_steps = eval_with_noise(env, net, neg_noise, noise_std)
            #print(_, "\n",noise, pos_reward, neg_reward)
            rewards_queue.put(RewardsItem(seed=seed, pos_reward=pos_reward, neg_reward=neg_reward, steps=pos_steps+neg_steps))

    pass


def train_step(brain, novelty, batch_noise, batch_rewards, step_idx):
    """
    Optimizes the weights of the NN based on the rewards and noise gathered
    """
    # normalize rewards to have zero mean and unit variance
    norm_reward = np.array(batch_reward)
    norm_reward -= np.mean(norm_reward)
    s = np.std(norm_reward)
    if abs(s) > 1e-6:
        norm_reward /= s

    weighted_noise = None
    for noise, reward in zip(batch_noise, norm_reward):
        if weighted_noise is None:
            weighted_noise = [(W*reward* p_n) + ((1-W)*novelty*p_n) for p_n in noise]  # combining reward and novelty
        else:
            for w_n, p_n in zip(weighted_noise, noise):
                w_n += (W*reward* p_n) + ((1-W)*novelty*p_n)


    for p, p_update in zip(brain.parameters(), weighted_noise):
        update = p_update / (len(batch_reward)*NOISE_STD)
        p.data += LR * update


def test_current_params(env, net):
    """
    Runs the current network parameters on the env to visually monitor the progress.
    """
    state = env.reset()

    while True:
        env.render()
        state = torch.from_numpy(state).unsqueeze(0).float()
        probs = brain(state)
        action = probs.max(dim = 1)[1]
        state, reward, done, _ = env.step(action.data.numpy()[0])

        if done:
            break

def get_behavior_char(env, net):
    """
    Returns the initial behavior characterization value b_pi0 for a network.
    The value is defined in this case as the final state of agent in the environment.
    
    >>> Important to find a good behavior characterization. Depents on the environment! <<< -> final state, step count ... 
    
    """
    state = env.reset()
    step_count = 0
    while True:
        state = torch.from_numpy(state).unsqueeze(0).float()
        probs = brain(state)
        action = probs.max(dim = 1)[1]
        state, reward, done, _ = env.step(action.data.numpy()[0])
        step_count += 1
        if done:
            break
    #print(step_count)
    return  np.array([step_count]) #state 


def get_kNN(archive, bc, n_neighbors):
    """
    Searches and samples the K-nearest-neighbors from the archive and a new behavior characterization
    returns the summed distance between input behavior characterization and the bc in the archive
    
    """

    archive = np.concatenate(archive)
    neigh = NearestNeighbors(n_neighbors=n_neighbors)
    neigh.fit(archive)
    distances, idx = neigh.kneighbors(X = bc, n_neighbors=n_neighbors)
    #k_nearest_neighbors = archive[idx].squeeze(0)

    return sum(distances.squeeze(0))
    
    

# =============================================================================
# def calc_novelty(b_pi_theta, archive):
#     """
#     calculates the novelty of a given arcive of behavior characterizations.
#     returns the mean distance between the initial behavior characterizations and all new gathered behavior characterizations.
#     """
#     # distance loss function:
#     distance = nn.MSELoss() #nn.PairwiseDistance()
#     # creates arcive vector for distance calc
#     archive_v = torch.cat(archive)
#     # create a vector of initial behavior characterizations in the shape of the arcive length
#     b_pi_theta_v = torch.cat([b_pi_theta for i in range(len(archive))])
# 
#     return torch.sqrt(distance(b_pi_theta_v, archive_v)).mean()
# =============================================================================

def calc_noveltiy_distribution(novelties):
    """
    Calculates the probabilities of each model parameters of being selected as its
    novelty normalized by the sum of novelty across all policies:

    P(theta_m) for each element in the meta_population M - m element M

    """
    probabilities = [round((novel/(sum(novelties))),4) for novel in novelties]
    return probabilities


if __name__ == "__main__":

    env = gym.make(ENV_NAME)
    #env.seed(2)
    MPS = 2 # meta population size
    meta_population = [Model(env.observation_space.shape[0],env.action_space.n, idx=i) for i in range(MPS)]

    # create arcive for models
    archive = []
    writer = SummaryWriter()
    iterations = 300 #1500 # max iterations to run

    delta_reward_buffer = deque(maxlen=10)  # buffer to store the reward gradients to see if rewards stay constant over a defined time horizont ~> local min
    W = 1

    params_queues = [mp.Queue(maxsize=1) for _ in range(PROCESSES_COUNT)]
    rewards_queue = mp.Queue(maxsize=ITERS_PER_UPDATE)
    workers = []

    for idx, params_queue in enumerate(params_queues):
        proc = mp.Process(target=worker_func, args=(idx, params_queue, rewards_queue, NOISE_STD))
        proc.start()
        workers.append(proc)

    print("All started!")
    step_idx = 0
    reward_history = []
    reward_max =[]
    reward_min = []
    reward_std = []

    old_m_reward = 0

    for step_idx in range(iterations):

        ########################## NOVELTY BRAIN SELECTION #############################
        # select new network from the meta population based on its probability:
        if len(archive) > 0:
            novelties = []
            S = np.minimum(K_NEIGHBORS, len(archive))
            for model in meta_population:
                b_pi_theta = torch.from_numpy(get_behavior_char(env, model)).unsqueeze(0).float()
                distance = get_kNN(archive, b_pi_theta.numpy(), S)
                novelty = distance / S
                if novelty <= 1e-3:
                    novelty = 5e-3
                novelties.append(novelty)

            #print("novelties:", novelties)
            
            probs = calc_noveltiy_distribution(novelties)
            #print("probs: ", probs )
            probs = np.array(probs)
            probs /= probs.sum()   # norm so that sum up to one - does without as well but np gives error because of rounding
            brain_idx = np.random.choice(list(range(MPS)),p=probs) # select new brain based on novelty probabilities
            brain = meta_population[brain_idx]
            novelty = novelties[brain_idx]
        else:
            brain_idx = np.random.randint(0, MPS)
            brain = meta_population[brain_idx]
            novelty = 1
        ###################################################################################

        # broadcasting network params
        params = brain.state_dict()
        for q in params_queues:
            q.put(params)

        batch_noise = []
        batch_reward = []
        batch_steps_data = []
        batch_steps = 0
        results = 0

        while True:
            #print(rewards_queue.qsize())
            while not rewards_queue.empty():
                reward = rewards_queue.get_nowait()
                np.random.seed(reward.seed) # sets the seed of the current worker rewards
                noise, neg_noise = sample_noise(brain)
                batch_noise.append(noise)
                batch_reward.append(reward.pos_reward)
                batch_noise.append(neg_noise)
                batch_reward.append(reward.neg_reward)
                results += 1
                batch_steps += reward.steps

            if results == PROCESSES_COUNT * ITERS_PER_UPDATE:
                break

        step_idx += 1
        m_reward = np.mean(batch_reward)

        reward_gradient_mean = np.mean(delta_reward_buffer)
        r_koeff = abs(m_reward - reward_gradient_mean)
        # if last few rewards are almost konstant -> stuck in loc minima -> decrease W for exploration: higher novelty weight
        if r_koeff < 1.5:
            W = np.maximum(0, W-0.05)
        else:
            W = np.minimum(1, W+0.05)
        delta_reward_buffer.append(m_reward)
        old_m_reward = m_reward

        writer.add_scalar("mean_reward", np.mean(batch_reward), step_idx)
        writer.add_scalar("max_reward", np.max(batch_reward), step_idx)
        writer.add_scalar("min_reward", np.min(batch_reward), step_idx)
        writer.add_scalar("std", np.std(batch_reward), step_idx)
        writer.add_scalar("novelty", novelty, step_idx)
        writer.add_scalar("novelty_w", W, step_idx)
# =============================================================================
#         if m_reward > -250:
#             print("\nSolved the environment in {} steps".format(step_idx))
#             break
# =============================================================================
        train_step(brain, novelty, batch_noise, batch_reward, step_idx)
        # select new behavior:
        b_pix = torch.from_numpy(get_behavior_char(env, brain)).unsqueeze(0).float()
        # append new behavior to specific brain archive
        archive.append(b_pix.numpy())

        print("\rStep: {}, Mean_Reward: {:.2f}, Novelty: {:.2f}, W: {:.2f} r_koeff: {:.2f}".format(step_idx, m_reward, novelty, W, r_koeff), end = "", flush = True)

#        if step_idx % 10 == 0:
#            test_current_params(env, brain)

    for worker, p_queue in zip(workers, params_queues):
        p_queue.put(None)
        worker.join()
