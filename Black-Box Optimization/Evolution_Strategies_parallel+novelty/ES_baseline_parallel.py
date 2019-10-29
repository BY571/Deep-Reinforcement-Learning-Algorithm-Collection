# -*- coding: utf-8 -*-
"""
Created on Wed Oct  9 10:24:39 2019

@author: Z0014354
"""

import numpy as np
import matplotlib.pyplot as plt
import gym
import multiprocessing as mp
import collections
import copy

ITERS_PER_UPDATE = 10
NOISE_STD = 0.01
LR = 1e-3
PROCESSES_COUNT = 8 # amount of worker
HIDDEN_SIZE = 4
ENV_NAME = "CartPole-v0" 
RewardsItem = collections.namedtuple('RewardsItem', field_names=['seed', 'pos_reward', 'neg_reward', 'steps'])



class Model(object):

    def __init__(self, stateCnt, actionCnt, hidden_size = HIDDEN_SIZE):
        # inits zero weights
        self.weights = [np.zeros(shape=(stateCnt, hidden_size)), np.zeros(shape=(hidden_size, hidden_size)), np.zeros(shape=(hidden_size,actionCnt))]

    def predict(self, inp):
        out = np.expand_dims(inp.flatten(), 0)
        out = out / np.linalg.norm(out)
        weight_len = len(self.weights)
        for idx, layer in enumerate(self.weights):
            # hidden activation
            if idx < weight_len - 1:
                out = self.activation(np.dot(out, layer))
            # outout activation
            else:
                out = self.activation(np.dot(out, layer), type_="output_layer")
        return out[0]
    
    def activation(self,x, type_="hidden"):
        if type_ == "hidden":
            # relu
            return np.maximum(x,0)
            
            # softmax
            #return (np.exp(x))/sum(np.exp(x))
            
            #softplus
            #return np.log(1 + np.exp(x))
            
            #sigmoid
            #return 1/(1+np.exp(-x))
            
            # tanh
            #return np.tanh(x)
        else:
            # softnmax
            #return (np.exp(x))/sum(np.exp(x))
            
            # relu
            return np.maximum(x,0)
        
    def get_weights(self):
        return self.weights

    def set_weights(self, weights):
        self.weights = weights
        
        
def evaluate(env, brain):
    """
    Runs an evaluation on the given brain.
    """
    state = env.reset()
    rewards = 0
    steps = 0
    while True:
        state = np.expand_dims(state, axis=0)
        action_prob = brain.predict(state)
        action = action_prob.argmax()  # for discrete action space
        
        next_state, reward, done, _ = env.step(action)
        rewards += reward
        steps  += 1
        state = next_state
        if done:
            break
      
    return rewards, steps


def sample_noise(brain):
    """
    Sampling noise to a positive and negative noise buffer.
    """
    pos = []
    neg = []
    for param in brain.get_weights():
        noise_t = np.random.normal(size = param.shape)
        pos.append(noise_t)
        neg.append(-noise_t)
    return pos, neg


def eval_with_noise(env, brain, noise, noise_std):
    """
    Evaluates the current brain with added parameter noise
  
    """
  
    old_params = copy.deepcopy(brain.get_weights())
    new_params = []
    for p, p_n in zip(brain.get_weights(), noise):
        p += noise_std*p_n
        new_params.append(p)
    brain.set_weights(new_params)
    r, s = evaluate(env, brain)
    brain.set_weights(old_params)
    return r, s 


def worker_func(worker_id, params_queue, rewards_queue, noise_std):
    #print("worker: {} has started".format(worker_id))
    env = gym.make(ENV_NAME)
    net = Model(env.observation_space.shape[0], env.action_space.n)

    while True:
        params = params_queue.get()
        if params is None:
            break

        # set parameters of the queue - equal to: net.load_state_dict(params)
        net.set_weights([param for param in params])
        
        for _ in range(ITERS_PER_UPDATE):
            seed = np.random.randint(low=0, high=65535)
            np.random.seed(seed)
            noise, neg_noise = sample_noise(net)
            pos_reward, pos_steps = eval_with_noise(env, net, noise, noise_std)
            neg_reward, neg_steps = eval_with_noise(env, net, neg_noise, noise_std)
            #print(_, "\n",noise, pos_reward, neg_reward)
            
            rewards_queue.put(RewardsItem(seed=seed, pos_reward=pos_reward, neg_reward=neg_reward, steps=pos_steps+neg_steps))

    pass


def train_step(brain, batch_noise, batch_rewards, step_idx):
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
            weighted_noise = [reward * p_n for p_n in noise]
        else:
            for w_n, p_n in zip(weighted_noise, noise):
                w_n += reward * p_n
        

    for p, p_update in zip(brain.get_weights(), weighted_noise):
        update = p_update / (len(batch_reward)*NOISE_STD)
        p += LR * update
        
        
        
if __name__ == "__main__":

    env = gym.make(ENV_NAME)
    #env.seed(2)
    brain = Model(env.observation_space.shape[0], env.action_space.n)

    iterations = 100 # max iterations to run 

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
    reward_std = []


    for step_idx in range(iterations):
        # broadcasting network params
        params = brain.get_weights()
        for q in params_queues:
            q.put(params)

        batch_noise = []
        batch_reward = []
        batch_steps_data = []
        batch_steps = 0
        results = 0
        while True: 
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
        reward_history.append(m_reward)
        reward_max.append(np.max(batch_reward))
        reward_std.append(np.std(batch_reward))
        if m_reward > 199:
            print("\nSolved the environment in {} steps".format(step_idx))
            break
        train_step(brain, batch_noise, batch_reward, step_idx)

        print("\rStep: {}, Mean_Reward: {:.2f}".format(step_idx, m_reward), end = "", flush = True)


    for worker, p_queue in zip(workers, params_queues):
        p_queue.put(None)
        worker.join()

    plt.figure(figsize = (11,7))
    plt.plot(reward_history, label = "Mean Reward", color = "orange")
    plt.plot(reward_max, label = "Max Reward", color = "blue")
    plt.plot(reward_std, label = "Reward std", color = "green")
    plt.xlabel("Steps")
    plt.ylabel("Rewards")
    plt.legend()
    plt.show()