import gym
from gym import wrappers
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

import numpy as np
from collections import deque
import random
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import style
import time

class Network(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Network,self).__init__()
        self.linear1 = nn.Linear(input_dim, 40)
        self.linear2 = nn.Linear(40, 40)
        self.linear3 = nn.Linear(40, output_dim)

    def forward(self,x):
        x = self.linear1(x)
        x = F.relu(x)
        x = self.linear2(x)
        x = F.relu(x)
        out = self.linear3(x)
        return out

class Agent:
    def __init__(self, state_size, action_size):
        
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=5000)
        self.gamma = 0.95    # discount rate
        self.epsilon = 0.4  # exploration rate
        self.epsilon_start = self.epsilon
        self.learning_rate = 0.001
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") #activate device
        
        # Our DQN and the Target Network
        self.model = Network(state_size, action_size).to(self.device)
        self.target_model = Network(state_size, action_size).to(self.device)

        self.criteria = nn.MSELoss()
        self.opt = optim.Adam(self.model.parameters(), lr=self.learning_rate)
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def update_target(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def adapt_epsilon(self,ep):
        # Epsilon starts at 0.5 linear increasing to 0.99 by ep 4000:
        # linear: epsilon = 0.0001225*ep+self.epsilon_start
        # exponent (4000 eps): epsilon = self.epsilon_start + (ep/5714)**2
        if ep == 0:
            pass
        if self.epsilon < 0.98:
            self.epsilon = self.epsilon_start + (ep/3800)**2 #4500
    
    def act(self, state, status = "Train"):
        if status == "Play": 
            self.epsilon = 0.95
        if np.random.rand() > self.epsilon:
            return random.randrange(self.action_size)
        
        act_values = self.model(Variable(torch.Tensor(state)).to(self.device)).cpu().data.numpy()
        return np.argmax(act_values[0])

    def give_epsilon(self):
        return self.epsilon

    def replay(self, batch_size):
        if len(self.memory) < batch_size: 
            return
        minibatch = random.sample(self.memory, batch_size)

        for state, action, reward, next_state, done in minibatch:
            target = reward
            self.model.train()
            if not done:
                next_state_v = Variable(torch.Tensor(next_state))
                target = self.target_model(next_state_v.to(self.device)).cpu() # target has to be on cpu for numpy
                target = target.data.numpy()[0]
            target_actual = self.target_model(Variable(torch.Tensor(state)).to(self.device)).cpu().data.numpy()
            target_actual[0][action] = reward + self.gamma *np.amax(target)
            
            self.opt.zero_grad()
            out = self.model(Variable(torch.Tensor(state)).to(self.device))
            loss = self.criteria(out, Variable(torch.Tensor(target_actual)).to(self.device))
            loss.backward()
            self.opt.step()
            



def play(Ep,agent, status = "train"):
    # for active plotting:
    learning_graph = []
    epsilons = []
    learning_graph_live = deque(maxlen = 180)
    epochs_live = deque(maxlen = 180)
    epsilons_live = deque(maxlen = 180)

    batch_size = 64
    env = gym.make("CartPole-v1")
    env = wrappers.Monitor(env, "Saved_Videos/", resume=True, video_callable=lambda episode_id: episode_id%250==0)
    action_space = env.action_space.n 
    state_space = env.observation_space.shape[0]
    if agent == None:
        agent = Agent(state_space,action_space)
    for ep in range(Ep):
        state = env.reset()
        state = np.reshape(state,[1,state_space]) 
        done = False
        score = 0
        agent.adapt_epsilon(ep) # Increasing the epsilon linear - adjustable to non linear, log,...
        while not done:

            if status == "play":
                env.render()
            action = agent.act(state, status)
            new_state, reward, done, _ = env.step(action)
            new_state  = np.reshape(new_state,[1,state_space])
            agent.remember(state, action, reward, new_state, done)
            state = new_state
            score +=1

            if done:
                break
        
        
        
        print("Episode {}# Score: {}# Epsilon {}".format(ep, score + 1,agent.give_epsilon()))
        # Update Target Network
        if ep % 200 == 0:
            agent.update_target()
            print("Updated Target Network!")
        agent.replay(batch_size)
        # Live plot
        learning_graph.append(score)
        epsilons.append(agent.give_epsilon()*100)
        learning_graph_live.append(score)
        epochs_live.append(ep)
        epsilons_live.append(agent.give_epsilon()*100)

        plt.plot(epochs_live, learning_graph_live,"b")
        plt.plot(epochs_live, epsilons_live,"r")
        plt.xlabel("Epoch")
        plt.ylabel("Score / Epsilon")
        plt.title("Score Live Plot")
        plt.show()
        plt.pause(0.00000001)
        plt.clf()
        
    return learning_graph, epsilons, agent

def main():
    Episodes = 4000 #4001
    graph,epsilons,agent = play(Episodes,None, "train")
    plt.plot(graph, "b")
    plt.plot(epsilons, "r")
    plt.xlabel("Episoden")
    plt.ylabel("Score / Epsilon")
    plt.show()

    print("Do you want to save the model?")
    answer = input("Y/N\n")
    if answer == "Y":
        name = input("give a name for the model: \n")
        agent.save_learnings(name)
    else:
        pass
    

    print("Soll der Agent getestet werden?\n")
    n = input("Wie viele Episoden sollen gespielt werden?")
    x,y, ag = play(int(n),agent,status = "play")

if __name__ == "__main__":
    fig = plt.figure()
    plt.ion()
    main()
