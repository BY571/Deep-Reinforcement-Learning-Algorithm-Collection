# -*- coding: utf-8 -*-
"""
Created on Wed Dec  4 10:33:09 2019

@author: Z0014354

PPO with GAE implementation of Sebastian Dittert
"""

import gym
import math
import torch 
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torch.nn.utils import clip_grad_norm_
from collections import deque
from tensorboardX import SummaryWriter
import MultiPro
import argparse
import time 


def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)

class Critic(nn.Module):
    def __init__(self, input_shape, hidden_size):
        super(Critic, self).__init__()
        self.layer1 = nn.Linear(input_shape, hidden_size)
        self.layer2 = nn.Linear(hidden_size, hidden_size)
        self.layer3 = nn.Linear(hidden_size, 1)
        self.reset_parameters()
    
    def forward(self,x):
        x = torch.tanh(self.layer1(x))
        x = torch.tanh(self.layer2(x))
        x = self.layer3(x)      
        
        return x
    
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.layer1.weight)
        nn.init.xavier_uniform_(self.layer2.weight)
        #nn.init.xavier_uniform_(self.layer3.weight)
      
class Actor(nn.Module):
    def __init__(self, input_shape, output_shape, action_high_low, hidden_size):
        super(Actor, self).__init__()
        self.layer1 = nn.Linear(input_shape, hidden_size)
        self.layer2 = nn.Linear(hidden_size,hidden_size)

        self.mean = nn.Linear(hidden_size, output_shape)
        self.variance = nn.Linear(hidden_size, output_shape)
        self.action_high_low = action_high_low
        #self.reset_parameters()

    def forward(self, x):

        x = torch.tanh(self.layer1(x))
        head = torch.tanh(self.layer2(x))
        
        mean = torch.tanh(self.mean(head)) # tanh squashed output to the range of -1..1
        variance = F.softplus(self.variance(head)) # log(1 + e^x) has the shape of a smoothed ReLU
        sigma = torch.sqrt(variance.cpu())
        m = Normal(mean.cpu(), sigma)
        actions = m.sample()
        logprobs = m.log_prob(actions) #for the optimization step we create a new distribution based on the new mean and variance - still taking the logprobs based on the old actions!

        return actions, logprobs, m
            

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.layer1.weight)
        nn.init.xavier_uniform_(self.layer2.weight)
        nn.init.xavier_uniform_(self.mean.weight)
        #nn.init.xavier_uniform_(self.variance.weight)

    
    
class Agent():
  def __init__(self,
               state_size,
               action_size,
               action_high_low,
               hidden_size,
               LR_A=3e-4,
               LR_C=3e-4,
               gamma=0.99,
               lambda_=0.95,
               mini_batch_size=512,
               ppo_epochs=5):
    
    self.state_size = state_size
    self.actor = Actor(state_size, action_size, action_high_low, hidden_size).to(device)
    self.action_high = action_high_low[0]
    self.action_low = action_high_low[1]
    self.critic = Critic(state_size, hidden_size).to(device)

    self.gamma = gamma
    self.lambda_ = lambda_
    self.mini_batch_size = mini_batch_size
    self.ppo_epochs = ppo_epochs
    

    self.optimizer_a = optim.Adam(params=self.actor.parameters(), lr=LR_A) #RMSprop
    self.optimizer_c = optim.Adam(params=self.critic.parameters(), lr=LR_C)
       

  def test_net(self, env, count = 10):
      """
      Tests the agents performance with current weights.
      """
      rewards = 0.0
      steps = 0
      entropys = 0.0

      for _ in range(count):
          obs = env.reset()

          while True:
              obs_v = torch.from_numpy(obs).float()
              action, _, dist = self.actor(obs_v.to(device))
              entropy = dist.entropy().detach().cpu().numpy()
              action = action.cpu().numpy()
              action = np.clip(action*self.action_high, self.action_low, self.action_high)
              obs, reward, done, info = env.step(action)

              rewards += reward
              entropys += entropy.mean()
              steps += 1
              if done:
                  break

      return rewards/count, entropys/count, steps/count 




  def compute_gae(self, next_value, rewards, masks, values):
      """
      lambda => 1: high variance, low bias
      lambda => 0: low variance, high bias
      """

      rewards_batch = list(zip(*rewards))
      masks_batch = list(zip(*masks))
      values_batch = torch.cat((torch.stack(values, dim=1).squeeze(2), next_value.squeeze(0)),dim=1)
      
      out_discounted_rewards = []
      out_advantage = []
      for rewards, masks, values  in zip(rewards_batch, masks_batch, values_batch):
      
        gae = 0
        disc_returns = []
        advantage = []
        for step in reversed(range(len(rewards))):
            # d = r_t +gamma*V(s_t+1) - V(s)
            delta = rewards[step] + self.gamma * values[step + 1] * masks[step] - values[step]
            # sum(lambda*gamma)^t* delta_t+1
            gae = delta + self.gamma * self.lambda_ * masks[step] * gae

            disc_returns.insert(0, gae + values[step]) # adding values since we want the returns and not the advantage yet!  A(a,s) = Q"returns" - V(s)
            advantage.insert(0, gae)
            
        out_discounted_rewards.append(disc_returns)
        out_advantage.append(advantage)
        
      return torch.FloatTensor(out_discounted_rewards).flatten().unsqueeze(1), torch.FloatTensor(out_advantage).flatten().unsqueeze(1)


  def ppo_iter(self, states, actions, log_probs, advantage, discounted_rewards):
      batch_size = len(states)

      for i in range(batch_size // self.mini_batch_size):
          rand_ids = np.random.randint(0, batch_size, self.mini_batch_size)

          yield states[rand_ids], actions[rand_ids], log_probs[rand_ids], advantage[rand_ids], discounted_rewards[rand_ids]



  def ppo_update(self, states, actions, log_probs, advantage, discounted_rewards, eps_clip=0.2):
    """

    """

    a_loss_batch = []
    c_loss_batch = []


    for _ in range(self.ppo_epochs):
      for states_i, old_actions, old_logprobs, advantage_i, discounted_reward_i  in self.ppo_iter(states, actions, log_probs, advantage, discounted_rewards):

        self.optimizer_c.zero_grad()
        #train critic
        new_value = self.critic(states_i.to(device))

        c_loss = .5 * (discounted_reward_i - new_value).pow(2).mean() 
        c_loss.backward()
        #print("C: ", c_loss)
        clip_grad_norm_(self.critic.parameters(),CLIP_GRAD)
        self.optimizer_c.step()

        #train actor
        self.optimizer_a.zero_grad()
        _, _, dist = self.actor(states_i.to(device))
        new_logprobs = dist.log_prob(old_actions)
        entropy = dist.entropy()
        
        ratio = torch.exp(new_logprobs - old_logprobs.detach())
        surr = ratio * advantage_i
        clip = torch.clamp(ratio, 1.0 - eps_clip, 1.0 + eps_clip) 
        a_loss = torch.min(surr, clip*advantage_i )
        a_loss = (- a_loss - ENTROPY_BONUS * entropy).mean()
        clip_grad_norm_(self.actor.parameters(),CLIP_GRAD)
        a_loss.backward(retain_graph=True)
        
        self.optimizer_a.step()

        c_loss_batch.append(c_loss.detach().numpy())
        a_loss_batch.append(a_loss.detach().numpy())      

    return np.array(c_loss_batch).mean(), np.array(a_loss_batch).mean()



def main(args):
    torch.multiprocessing.freeze_support()
    t0 = time.time()
    ENV = args.env #"MountainCarContinuous-v0"  #Pendulum-v0 LunarLanderContinuous-v0
    
    env = gym.make(ENV)#Creating the Environment
    writer = SummaryWriter("runs/"+args.info)
    n_cpu = args.worker
    
    envs = MultiPro.SubprocVecEnv([lambda: gym.make(ENV) for i in range(n_cpu)])
    seed = args.seed
    
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    env.seed(seed)
            
    
    state_size  = env.observation_space.shape[0]
    action_size = env.action_space.shape[0]
    action_high_low = (env.action_space.high[0], env.action_space.low[0])
    
    agent = Agent(state_size, action_size, action_high_low= action_high_low, hidden_size=args.layer_size, LR_A=args.lr, LR_C=args.lr, gamma=args.gamma, lambda_=args.lambda_, mini_batch_size=args.mini_batch_size, ppo_epochs=args.ppo_updates)
    
    max_episodes = args.ep
    plot_rewards = []
    max_steps = int(args.max_steps/n_cpu)
    
    # calc reshape stacking size
    shape = (max_steps*n_cpu, state_size)
    
    for ep in range(max_episodes+1):
        states = envs.reset()
    
        done = False
        
        state_batch = []
        value_batch = []
        action_batch = []
        logprob_batch = []
        rewards_batch = []
        masks = []
        for step in range(max_steps):
    
            states = torch.from_numpy(states).float()

            action, logprob, _  = agent.actor(states.to(device))  
            value = agent.critic(states.to(device))
            action_v = action.cpu().numpy()

            action_v = np.clip(action_v*env.action_space.high[0], env.action_space.low[0], env.action_space.high[0])
            next_states, reward, done, _ = envs.step(action_v)

            state_batch.append(states)
            value_batch.append(value)
            logprob_batch.append(logprob)
            action_batch.append(action)
            rewards_batch.append(torch.from_numpy(reward).float())  
            masks.append(torch.from_numpy(1 - done).float())
    
            states = next_states

    
            if np.any(done):
              states = envs.reset()
        
        # stack all gathered data

        state_batch = torch.stack(state_batch, dim=1).reshape(shape)
        actions_batch = torch.stack(action_batch, dim=1).reshape(max_steps*n_cpu,action_size)
        logprob_batch = torch.stack(logprob_batch, dim=1).reshape(max_steps*n_cpu,action_size).detach()    

        
        # calculate advantage:
        next_value = agent.critic(torch.from_numpy(next_states).float())
        discounted_rewards, advantage = agent.compute_gae(next_value, rewards_batch, masks, value_batch)

        # normalize advantage:
        advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-5)
        
        c_loss, a_loss = agent.ppo_update(states=state_batch, actions=actions_batch, log_probs=logprob_batch, advantage=advantage.detach()  , discounted_rewards=discounted_rewards.detach())
        writer.add_scalar("critic_loss", c_loss, ep)
        writer.add_scalar("actor_loss", a_loss, ep)
    
        
        if ep != 0 and ep % 5 == 0:
          test_rewards, test_entropy, test_steps = agent.test_net(env)
          writer.add_scalar("entropy",test_entropy, ep)
          writer.add_scalar("max_reward",test_rewards, ep)
          plot_rewards.append(test_rewards)
    
          print("\rEpisode: {} | Ep_Reward: {:.2f} | Average_100: {:.2f}".format(ep, test_rewards, np.mean(plot_rewards[-100:])), end = "", flush = True)
          
    envs.close()      
    t1 = time.time()
    plt.pause(60)
    env.close()
    print("training took {} min!".format((t1-t0)/60))
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("-env", type=str,default="Pendulum-v0", help="Environment name")
    parser.add_argument("-info", type=str, help="Information or name of the run")
    parser.add_argument("-ep", type=int, default=200, help="The amount of training episodes, default is 200")
    parser.add_argument("-seed", type=int, default=0, help="Seed for the env and torch network weights, default is 0")
    parser.add_argument("-lr", type=float, default=5e-4, help="Learning rate of adapting the network weights, default is 5e-4")
    parser.add_argument("-entropy_bonus", type=float, default=1e-3,  help="Entropy bonus for exploration - default is 1e-2")
    parser.add_argument("-layer_size", type=int, default=64, help="Number of nodes per neural network layer, default is 64")
    parser.add_argument("-worker", type=int, default=8, help="Number of parallel worker -default is 8")
    parser.add_argument("-lambda_", type=float, default=0.95, help="GAE lambda")
    parser.add_argument("-g", "--gamma", type=float, default=0.99, help="discount factor gamma, default is 0.99")
    parser.add_argument("-CG", "--clip_grad", type=float, default=0.25, help="Clip the gradients for updating the network parameters, default is 0.25")
    parser.add_argument("-ms", "--max_steps", type=int, default=2048, help="Maximum steps that are taken by the agent in the environment before updating")
    parser.add_argument("-mbs", "--mini_batch_size", type=int, default=256, help="Mini Batch size for the ppo updates, default is 256")
    parser.add_argument("-updates", "--ppo_updates", type=int, default=7, help="Number of PPO updates, default is 7")
    args = parser.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using: ", device)    

    ENTROPY_BONUS = args.entropy_bonus
    CLIP_GRAD = args.clip_grad
    main(args)
