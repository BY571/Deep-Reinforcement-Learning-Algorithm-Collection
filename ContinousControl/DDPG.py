import random
from tensorboardX import SummaryWriter
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np 
import roboschool
import gym
from gym import wrappers
import pybullet_envs
import time

class NormalizedActions(gym.ActionWrapper):

    def _action(self, action):
        low_bound   = self.action_space.low
        upper_bound = self.action_space.high
        
        action = low_bound + (action + 1.0) * 0.5 * (upper_bound - low_bound)
        action = np.clip(action, low_bound, upper_bound)
        
        return action

    def _reverse_action(self, action):
        low_bound   = self.action_space.low
        upper_bound = self.action_space.high
        
        action = 2 * (action - low_bound) / (upper_bound - low_bound) - 1
        action = np.clip(action, low_bound, upper_bound)
        
        return action

class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0
    
    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done
    
    def __len__(self):
        return len(self.buffer)
      
class OUNoise(object):
    def __init__(self, action_space, mu=0.0, theta=0.15, max_sigma=0.3, min_sigma=0.3, decay_period=100000):
        self.mu           = mu            # mean value -> as "goal state" 0.0 in the sense of no noise
        self.theta        = theta
        self.sigma        = max_sigma     # variance of the noise 
        self.max_sigma    = max_sigma
        self.min_sigma    = min_sigma
        self.decay_period = decay_period
        self.action_dim   = action_space.shape[0]
        self.low          = action_space.low
        self.high         = action_space.high
        self.reset()
        
    def reset(self):
        self.state = np.ones(self.action_dim) * self.mu
        
    def evolve_state(self):
        x  = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(self.action_dim)
        self.state = x + dx
        return self.state
    
    def get_action(self, action, t=0):
        ou_state = self.evolve_state()
        self.sigma = self.max_sigma - (self.max_sigma - self.min_sigma) * min(1.0, t / self.decay_period)
        return np.clip(action + ou_state, self.low, self.high), ou_state
    
#https://github.com/vitchyr/rlkit/blob/master/rlkit/exploration_strategies/ou_strategy.py


class Actor(nn.Module):
  def __init__(self, input_shape, action_shape):
    super(Actor, self).__init__()
    self.actor = nn.Sequential(nn.Linear(input_shape[0],400),
                               nn.LayerNorm(400),
                               nn.ReLU(),
                               nn.Linear(400,300),
                               nn.LayerNorm(300),
                               nn.ReLU(),                               
                               nn.Linear(300,action_shape[0]),
                               nn.Tanh())
  def forward(self, x):
    state = torch.FloatTensor(x).to(device)
    return self.actor(state)

class Critic(nn.Module):
  def __init__(self, input_shape, action_shape):
    super(Critic, self).__init__()
    
    self.critic1 = nn.Sequential(nn.Linear(input_shape[0],400),
                                  #nn.LayerNorm(256),
                                  nn.ReLU())
    self.critic2 = nn.Sequential(nn.Linear(400+ action_shape[0], 300),
                                  #nn.LayerNorm(256),
                                  nn.ReLU(),
                                  nn.Linear(300,1))
  def forward(self,state, action):
    x = self.critic1(state)
    comb = torch.cat([x,action], dim = 1)
    return self.critic2(comb)

def update_and_optimize(batch_size):
  state, action, reward, next_state, done = replay_buffer.sample(batch_size)
  state_v = torch.FloatTensor(state).to(device)        # shape[batch_size,3]
  action_v = torch.FloatTensor(action).to(device)      # shape[batch_size,1]
  reward_v = torch.FloatTensor(reward).unsqueeze(1).to(device)    # shape [batch_size,1]
  next_state_v = torch.FloatTensor(next_state).to(device) # shape [batch_size,3]
  done_v = torch.FloatTensor(np.float32(done)).unsqueeze(1).to(device) # shape [batch_size,1]
  
  # update critic:
  critic_optim.zero_grad()
  Q_v = critic(state_v, action_v)
  next_action = target_actor(next_state).to(device)
  target_Q = target_critic(next_state_v, next_action.detach())
  discounted_target_Q = (reward_v + 0.99 * target_Q * (1.0 - done_v)).to(device)
  loss = critic_loss(Q_v, discounted_target_Q.detach())
  writer.add_scalar("Critic loss", loss, frame_idx)
  writer.add_scalar("Target_Q", target_Q.mean(), frame_idx)
  loss.backward()
  critic_optim.step()
  
  # update actor:
  actor_optim.zero_grad()
  current_action = actor(state_v.cpu())
  actor_loss = -critic(state_v, current_action.to(device)).mean()
  writer.add_scalar("Actor loss", actor_loss, frame_idx)
  actor_loss.backward()
  actor_optim.step()
  
  # Softupdate 
  soft_tau = 0.01
  for target_param, param in zip(target_critic.parameters(), critic.parameters()):
    target_param.data.copy_(
                target_param.data * (1.0 - soft_tau) + param.data * soft_tau
            )

    for target_param, param in zip(target_actor.parameters(), actor.parameters()):
      target_param.data.copy_(
                target_param.data * (1.0 - soft_tau) + param.data * soft_tau
            )


if __name__ == "__main__":
    start = time.time()
    # use cuda
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    print("Device: ",device)
    ENV_ID = "HalfCheetahBulletEnv-v0"  #HalfCheetahBulletEnv-v0 #MinitaurBulletEnv-v0
    env = gym.make(ENV_ID)
    #env = gym.make("RoboschoolHalfCheetah-v1") #RoboschoolHalfCheetah-v1
    env = wrappers.Monitor(env, "Saved_Videos/", resume=True, force = True, video_callable=lambda episode_id: episode_id% 5 ==0)
                                        #, video_callable=lambda x: True, force=True
    env = NormalizedActions(env)

    action_space = env.action_space.shape
    observation_space = env.observation_space.shape

    critic = Critic(observation_space, action_space).to(device)
    actor = Actor(observation_space, action_space).to(device)
    target_actor = actor
    target_critic = critic
    target_actor.load_state_dict(actor.state_dict())
    target_critic.load_state_dict(critic.state_dict())
    critic_optim = optim.Adam(critic.parameters(), lr = 0.001, weight_decay=1e-2)
    actor_optim = optim.Adam(actor.parameters(), lr = 0.0001)

    critic_loss = nn.MSELoss()

    replay_buffer_size = 1000000
    replay_buffer = ReplayBuffer(replay_buffer_size)

    writer = SummaryWriter()

    noise = OUNoise(env.action_space)
    batch_size = 128
    max_frames = 80000 #100000~32 min --300000 ~47 min
    frame_idx = 0
    rewards = []

    while frame_idx < max_frames:
        state = env.reset()
        noise.reset()
        ou_states = []
        episode_reward = 0
        done = False 
        step = 0
        print("Training Progress: {:.2f}".format(frame_idx/max_frames *100))
        while not done:
            action = actor(state)
            action, ou_state = noise.get_action(action.cpu().detach().numpy(), frame_idx) #step
            ou_states.append(ou_state)

            next_state, reward, done, _ = env.step(action)
            


            replay_buffer.push(state, action, reward, next_state, done)
            if len(replay_buffer) > batch_size:# and frame_idx % 10 == 0:
                update_and_optimize(batch_size)
            
            state = next_state
            episode_reward += reward
            frame_idx += 1
            step += 1

            
            if done:
                writer.add_scalar("Rewards", episode_reward, frame_idx)
                writer.add_scalar("Steps", step, frame_idx)
                writer.add_scalar("OU_state", np.array(ou_states).mean(), frame_idx)
        
    end = time.time()
    writer.close()
    print("------------------------------\nTraining for {:.2f} minutes".format((end-start)/60))
