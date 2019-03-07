import torch.nn as nn
import torch
import torch.optim as optim
from torch.autograd import Variable
import gym
from gym import wrappers
import numpy as np
import matplotlib.pyplot as plt
from torch.distributions import Categorical


class Policy(nn.Module):
  def __init__(self,input_shape,action_shape):
    super().__init__()
    
    self.model = nn.Sequential(
        nn.Linear(input_shape[0],64),
        nn.ReLU(),
        nn.Linear(64,32),
        nn.ReLU(),
        nn.Linear(32,action_shape),
        nn.Softmax(dim = 1)
    )
  def forward(self,x):
    return self.model(x)    

def action(model, s):
  # simple pytorch aproach for action-selection and log-prob calc 
  #https://pytorch.org/docs/stable/distributions.html
  prob = model(s)
  m = Categorical(prob)
  a = m.sample()
  # log p(a∣π(s))
  log_p = m.log_prob(a)
  #print(a.item(), log_p)
  return a.item(), log_p

  # naive own numpy aproach attenion! grad gets lost by transforming prob to numpy:
  #possible_actions = [i for i in range(len(prob.data.detach().numpy()[0]))]
  # choose accordingly to probability:
  #action = np.random.choice(possible_actions, p = prob.data.detach().numpy()[0])
  #calculate the log-prob for the chosen action:
  #grad = prob[0][action].grad_fn
  #log_prob = np.log(prob.data.detach().numpy()[0][action])
  # transform to torch Tensor:
  #log_prob = torch.Tensor([log_prob]).unsqueeze(0)
  #log_prob = Variable(log_prob,requires_grad=True)
  #log_prob.backward()
  #print(log_prob)
  #print(action,log_prob)
  #return action, log_prob

def policy_optimization(ep, model, optimizer,batch_rewards,log_probs):
  R = 0
  gamma = 0.99
  policy_loss = []
  rewards = []
  #calc discounted Rewards
  for r in batch_rewards[::-1]: # reverses the list of rewards 
    R = r + gamma * R
    rewards.insert(0, R) # inserts the current rewart to first position
    
  rewards = torch.tensor(rewards)
  # standardization to get data of zero mean and varianz 1, stabilizes learning 
  #-- attention scaling rewards looses information of special events with higher rewards - addapting on different environments  
  rewards = (rewards - rewards.mean()) / (rewards.std() + ep)
  for log_prob, reward in zip(log_probs, rewards):
    policy_loss.append(-log_prob * reward) #baseline+
  
  optimizer.zero_grad()
  policy_loss = torch.cat(policy_loss).sum()
  policy_loss.backward()
  optimizer.step()
  
def run(episodes,model,env):
  optimizer = optim.Adam(model.parameters(), lr = 1e-2)
  rewards = []
  steps_taken = []
  
  for i in range(episodes):
    done = False
    ep_rewards = 0
    batch_rewards = []
    log_probs = []
    state = env.reset()
    steps = 0
    while not done:
      a, log_p = action(model, torch.Tensor(state).unsqueeze(0))
      log_probs.append(log_p)
      new_state, reward, done, info = env.step(a)
      batch_rewards.append(reward)
      ep_rewards += reward
      steps +=1
      
      

      state = new_state
      
      
    rewards.append(ep_rewards)
    steps_taken.append(steps)
    print("Episode: {} --- Rewards: {} --- Steps: {}".format(i, ep_rewards, steps))
    policy_optimization(i, model, optimizer, batch_rewards,log_probs)

  return steps_taken
  
def main():
  USE_CUDA = torch.cuda.is_available()
  Variable = lambda *args, **kwargs: autograd.Variable(*args, **kwargs).cuda() if USE_CUDA else autograd.Variable(*args, **kwargs)
  
  env = gym.make("Acrobot-v1")
  env = wrappers.Monitor(env, "Saved_Videos/", resume=True, force = True, video_callable=lambda episode_id: episode_id%40==0)
  obs_shape = env.observation_space.shape
  action_shape = env.action_space.n
  episodes = 240
  model = Policy(obs_shape, action_shape)
  steps = run(episodes, model, env)

  plt.plot(steps)
  plt.xlabel("Episodes")
  plt.ylabel("Steps needed to reach goal")
  plt.show()

if __name__ == "__main__":
  #Argparse:
  main()
