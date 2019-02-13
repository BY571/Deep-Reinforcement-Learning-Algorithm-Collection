import math, random
from collections import deque
import cv2

import gym
from gym import wrappers
import wrapper
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd 
import torch.nn.functional as F
from IPython.display import clear_output

import matplotlib.pyplot as plt

USE_CUDA = torch.cuda.is_available()
Variable = lambda *args, **kwargs: autograd.Variable(*args, **kwargs).cuda() if USE_CUDA else autograd.Variable(*args, **kwargs)

class ReplayBuffer(object):
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        state      = np.expand_dims(state, 0)
        next_state = np.expand_dims(next_state, 0)
            
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        state, action, reward, next_state, done = zip(*random.sample(self.buffer, batch_size))
        return np.concatenate(state), action, reward, np.concatenate(next_state), done
    
    def __len__(self):
        return len(self.buffer)

class CnnDQN(nn.Module):
    def __init__(self, input_shape, num_actions):
        super(CnnDQN, self).__init__()
        
        self.input_shape = input_shape
        self.num_actions = num_actions
        
        self.convolutional_layers = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        
        self.value_layer = nn.Sequential(
            nn.Linear(self.feature_size(), 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )
        self.advantage_layer = nn.Sequential(
            nn.Linear(self.feature_size(), 512),
            nn.ReLU(),
            nn.Linear(512, self.num_actions)
        )

    def forward(self, x):
        x = self.convolutional_layers(x)
        x = x.view(x.size(0), -1)
        value = self.value_layer(x) # shape [1,1]
        value = value.expand(x.size(0), self.num_actions) # shape [1,6] 
        advantage = self.advantage_layer(x) #shape [1,6]
        advantage_mean = advantage.mean(1)#shape [1]
        advantage_mean = advantage_mean.unsqueeze(1) #shape[1,1]
        advantage_mean = advantage_mean.expand(x.size(0), self.num_actions) #shape [1,6]
        Q = value + advantage - advantage_mean
        #print("Q-Values: ",Q)
        return Q
    
    def feature_size(self):
        #Calculate the output size of the CNN
        return self.convolutional_layers(autograd.Variable(torch.zeros(1, *self.input_shape))).view(1, -1).size(1)
    
    def act(self, state, epsilon,action_space):
        if random.random() > epsilon:
            with torch.no_grad():
                state   = Variable(torch.FloatTensor(np.float32(state)).unsqueeze(0))
                q_value = self.forward(state)
                action  = q_value.max(1)[1].data[0] #.max(1)  maxdata: values--[0] and idx--[1] 
        else:
            action = random.randrange(action_space)
        return action
    
def update_target(current_model, target_model):
    target_model.load_state_dict(current_model.state_dict())

def save_model(model, idx):
    torch.save(model, "Saved_models/")

def epsilon_by_frame(frame_idx):
    epsilon_start = 1.0
    epsilon_final = 0.01 #0.01
    epsilon_decay = 30000 #30000
    eps = epsilon_final + (epsilon_start - epsilon_final) * math.exp(-1. * frame_idx / epsilon_decay)
    return eps

def compute_td_loss(batch_size,current_model,target_model,opti,loss_func,gamma,replay_buffer):
    state, action, reward, next_state, done = replay_buffer.sample(batch_size)
                                                                # shapes for normal image-- stacked (4,84,84) ...
    state      = Variable(torch.FloatTensor(np.float32(state))) #shape (1,84,84)
    next_state = Variable(torch.FloatTensor(np.float32(next_state))) #shape (1,84,84)
    action     = Variable(torch.LongTensor(action)) #shape [32] -- has to be long for gather function
    reward     = Variable(torch.FloatTensor(reward)) #shape [32]
    done       = Variable(torch.FloatTensor(done)) #shape [32]
    
    q_values      = current_model(state) #shape [32,6]
    next_q_values = current_model(next_state) #shape [32,6]
    next_q_state_values = target_model(next_state) #shape [32,6]
    
    q_value       = q_values.gather(1, action.unsqueeze(1)).squeeze(1) #shape [32] gathers q_values by the index of action
    next_q_value = next_q_state_values.gather(1, torch.max(next_q_values, 1)[1].unsqueeze(1)).squeeze(1) #shape [32] torch.max(nqv,1) gives the maxvalues--[0] and idx--[1] 
    expected_q_value = reward + gamma * next_q_value * (1 - done)  # shape [32]
    
    
    # DeepMind took nn.SmoothL1Loss()
    #loss = (q_value - Variable(expected_q_value.data)).pow(2).mean() #standard loss  -- .data to get rid of grad_fn=<AddBackward0>
    loss = loss_func(q_value,Variable(expected_q_value.data))
    
    opti.zero_grad()
    loss.backward()
    opti.step()
    return loss

def plot(frame_idx, rewards, losses):
    plt.close()
    plt.figure(figsize=(20,5))
    plt.subplot(121)
    plt.title("frames {}. reward: {}" .format(frame_idx, np.round(np.mean(rewards[-10:]),2)))
    plt.plot(rewards)
    plt.subplot(122)
    plt.title("loss")
    plt.plot(losses)
    plt.ylim(0,1)
    plt.draw()
    plt.pause(0.0001)

def processing(img):
    img = np.expand_dims(cv2.resize(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY), (84,84)),axis= 0)
    img = img.astype(np.uint8)
    #print(img.dtype)
    return img

def main():
    plt.ion()
    env = wrapper.make_atari("RiverraidNoFrameskip-v4", monitor=True,epidsode_capture=50)
    env = wrapper.wrap_deepmind(env,frame_stack=True, pytorch_img = True)
    action_space = env.action_space.n
    current_model = CnnDQN(env.observation_space.shape, action_space)#env.observation_space.shape
    target_model  = CnnDQN(env.observation_space.shape, action_space)    

    if USE_CUDA:
        current_model = current_model.cuda()
        target_model  = target_model.cuda()

#DeepMind took optim.RMSprop(current_model.parameters(), lr=0.000)
    #opti = optim.Adam(current_model.parameters(), lr=0.0001) 
    opti = optim.RMSprop(current_model.parameters(), lr=0.0001)
    loss_func = nn.SmoothL1Loss()

    replay_initial = 10000
    replay_buffer = ReplayBuffer(100000)

    num_frames = 1000000
    batch_size = 32
    gamma      = 0.99

    losses = []
    all_rewards = []
    episode_reward = 0

    state = env.reset() # shape normal:(1,84,84) -stacked (4,84,84)
    # Manuel Stacking
    #state = processing(state)
    #state = np.stack((state,state,state,state),axis = 1).squeeze(0)
    #assert state.shape == (4,84,84)
    for frame_idx in range(1, num_frames + 1):
        
        epsilon = epsilon_by_frame(frame_idx)
        print("Training :: Frame {} :: Epsilon {} ".format(frame_idx, round(epsilon,2)))
        action = current_model.act(state, epsilon,action_space)
        next_state, reward, done, _ = env.step(action)
        # Manuel Stacking
        #next_state = processing(next_state)
        #next_state = np.append(next_state, state[1:, :, :],axis= 0)
        #assert next_state.shape == (4,84,84)
        replay_buffer.push(state, action, reward, next_state, done)
        
        state = next_state
        episode_reward += reward
        
        if done:
            state = env.reset()
            # Manuel Stacking
            #state = processing(state)
            #state = np.stack((state,state,state,state),axis = 1).squeeze(0)
            all_rewards.append(episode_reward)
            episode_reward = 0
            
        if len(replay_buffer) > replay_initial:
            loss = compute_td_loss(batch_size,current_model, target_model,opti,loss_func,gamma,replay_buffer)
            losses.append(loss.item()) 
            
        if frame_idx % 10000 == 0:
            plot(frame_idx, all_rewards, losses)
            
        if frame_idx % 1000 == 0:
            update_target(current_model, target_model)
        
        #if frame_idx % 100000 ==0:
        #    save_model(current_model, frame_idx)

if __name__ == "__main__":
    main()