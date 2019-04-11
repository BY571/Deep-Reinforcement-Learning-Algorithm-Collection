import gym
from gym import wrappers
import time
import math
import torch 
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import pybullet_envs
from Parallel_processing import SubprocVecEnv

GAMMA = 0.99
ENTROPY_BETA = 0.001  
CLIP_GRAD = 0.1
ENV_ID = "MinitaurBulletEnv-v0"
#RENDER = True
spec = gym.envs.registry.spec(ENV_ID)
#spec._kwargs["render"] = RENDER

num_envs = 12

def make_env():
    def _thunk():
        env = gym.make(ENV_ID)
        return env

    return _thunk

envs = [make_env() for i in range(num_envs)]
envs = SubprocVecEnv(envs)
env = gym.make(ENV_ID)
env = wrappers.Monitor(env, "Saved_Videos/", resume=True, force = True, video_callable=lambda episode_id: episode_id%100 ==0)
#adapt wrapper to save monitorings


class A2C_Conti(nn.Module):
  def __init__(self, input_shape, output_shape):
    super(A2C_Conti, self).__init__()
    
    self.net = nn.Sequential(nn.Linear(input_shape, 256),
                             nn.ReLU()
                             )
    self.critic = nn.Sequential(nn.Linear(256, 1))          # estimated Value of the state
    self.mean = nn.Sequential(nn.Linear(256, output_shape),
                              nn.Tanh())                    # tanh squashed output to the range of -1..1
    self.variance =nn.Sequential(nn.Linear(256, output_shape),
                                 nn.Softplus())             # log(1 + e^x) has the shape of a smoothed ReLU
    
  def forward(self,x):
    x = self.net(x)
    return self.mean(x), self.variance(x), self.critic(x)

def test_net(net, env, count = 10):
  rewards = 0.0
  steps = 0
  for _ in range(count):
    obs = env.reset()
    while True:
      obs_v = torch.FloatTensor(obs).to(device)
      mean_v, var_v, v_ = net(obs_v)
      action = calc_actions(mean_v, var_v)
      obs, reward, done, info = env.step(action)
      rewards += reward
      steps += 1
      if done:
        break
  return rewards/count, steps/count


def calc_actions(mean, variance):
  mean = mean.cpu().data.numpy()
  sigma = torch.sqrt(variance).cpu().data.numpy()
  actions = np.random.normal(mean, sigma)
  actions = np.clip(actions, -1, 1)
  return actions


def compute_returns(terminal_reward, rewards,masks, gamma=GAMMA):
    R = terminal_reward
    returns = []
    for step in reversed(range(len(rewards))):
        R = rewards[step] + gamma * R * masks[step]
        returns.insert(0, R)
    return returns



def run_optimization(next_state, model,action_batch, mean_batch, variance_batch, values_batch, rewards_batch, masks):
    next_state = torch.FloatTensor(next_state).to(device)
    m_, v_, next_value = model(next_state)
    
    
    discounted_rewards = torch.cat(compute_returns(next_value, rewards_batch, masks)).detach()
    #print(discounted_rewards.shape)
    
    action_v    = torch.cat(action_batch)     # shape [n_steps * nr_envs, 8]
    mean_v      = torch.cat(mean_batch)       # shape [n_steps * nr_envs, 8]  8 output_values
    variance_v  = torch.cat(variance_batch)   # shape [n_steps * nr_envs, 8]
    values_v    = torch.cat(values_batch)     # shape [n_steps * nr_envs, 1]  only one value of the critic
    
    
    # A(s,a) = Q(s,a)- V(s)
    advantage = discounted_rewards - values_v.detach() 
    
    # calc log(pi):
    # torch.clamp to prevent division on zero if variance is to small 
    log_pi = -((action_v.cpu() - mean_v.cpu())**2)/(2 * variance_v.clamp(min = 1e-3).cpu()) - torch.log(torch.sqrt(2* math.pi * variance_v.cpu()))
    
    # policy_loss
    policy_loss  = -(log_pi * advantage.cpu()).mean() 

    
    value_loss = F.mse_loss(values_v, discounted_rewards) 
    #value_loss = advantage.pow(2).mean()   # same as above
    
    # calculate entropy
    #entropy = - torch.log10(torch.sqrt(2*math.pi*math.e*variance_v)).mean()  # directly as in the book
    entropy = (-(torch.log(2*math.pi*variance_v)+1)/2).mean()
    
    actor_loss_list.append(policy_loss)
    value_loss_list.append(value_loss)
    entropy_list.append(entropy)
    
    loss =  policy_loss.to(device) + value_loss + ENTROPY_BETA * entropy

    optimizer.zero_grad()
    loss.backward()
    #clip_grad_norm_(model.parameters(),CLIP_GRAD)
    optimizer.step()



if __name__ == "__main__":
    start = time.time()
    use_cuda = torch.cuda.is_available()
    device   = torch.device("cuda" if use_cuda else "cpu")
    input_shape  = envs.observation_space.shape[0]
    output_shape = envs.action_space.shape[0]

    model = A2C_Conti(input_shape, output_shape).to(device)
    optimizer = optim.Adam(params = model.parameters(),lr = 5e-5)

    max_frames   = 20000
    frame_idx    = 0

    actor_loss_list = []
    value_loss_list = []
    entropy_list = []

    plot_rewards = []
    step_distance = []
    steps = 0
    n_steps = 5

    state = envs.reset()

    while frame_idx < max_frames:
        
        action_batch = []
        mean_batch = []
        variance_batch = []
        values_batch    = []
        rewards_batch   = []
        masks     = []
        entropy = 0
        print("{:.2f} percent of training Progress".format(frame_idx/max_frames *100))

        for _ in range(n_steps):
            state = torch.FloatTensor(state).to(device)
            mean, variance, value = model(state)
            #print(mean.shape)
            #print(variance.shape)
            #print(mean.data.numpy)
            
            action = calc_actions(mean, variance) 
            next_state, reward, done, _ = envs.step(action)
            
            action_batch.append(torch.FloatTensor(action))                # [nr_of_envs, 8]
            mean_batch.append(mean)                                       # [nr_of_envs, 8]
            variance_batch.append(variance)                               # [nr_of_envs, 8]
            values_batch.append(value)                                    # [nr_of_envs, 8]
            rewards_batch.append(torch.FloatTensor(reward).unsqueeze(1).to(device))  # appends vector in shape [nr_of_envs, 1]
            masks.append(torch.FloatTensor(1 - done).unsqueeze(1).to(device))        # appends vector in shape [nr_of_envs, 1]
            
            #break  # for debugging
            
            state = next_state
            frame_idx += 1
            
            if frame_idx % 500 == 0:
                test_rewards, test_steps = test_net(model, env)
                step_distance.append(test_steps)
                plot_rewards.append(test_rewards)
        #break
        run_optimization(next_state, model,action_batch, mean_batch, variance_batch, values_batch, rewards_batch ,masks)
        #break
        
        
        
    end = time.time()
    print("training took {} minutes".format((end-start)/60 ))
    # PLOTTING RESULTS

    plt.figure(figsize = (20,7))
    plt.subplot(1,5,1)
    plt.title("policy_loss")
    plt.plot(actor_loss_list)
    plt.subplot(1,5,2)
    plt.title("value_loss")
    plt.plot(value_loss_list)
    plt.subplot(1,5,3)
    plt.title("entropy")
    plt.plot(entropy_list)
    plt.subplot(1,5,4)
    plt.title("rewards")
    plt.plot(plot_rewards)
    plt.subplot(1,5,5)
    plt.title("steps")
    plt.plot(step_distance)
    plt.show()