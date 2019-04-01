import numpy as np 
import gym
from gym import wrappers
import torch
import torch.nn as nn
from torch import optim
from collections import namedtuple
from tensorboardX import SummaryWriter
# Memory
Episode = namedtuple("Episode", field_names  = ["reward","steps"])
EpisodeStep = namedtuple("EpisodeStep", field_names = ["state", "action"])


class Network(nn.Module):
    def __init__(self, input_shape, output_shape):
        super(Network, self).__init__()

        
        self.net = nn.Sequential(
            nn.Linear(input_shape, 128),
            nn.ReLU(),
            nn.Linear(128, output_shape)
        )

    def forward(self,x):
        return self.net(x)

def filter_batch(batch, percentile = 70):
    rewards = list(map(lambda s: s.reward, batch))
    reward_bound = np.percentile(rewards, percentile)
    reward_mean = float(np.mean(rewards))

    train_obs = []
    train_act = []
    for example in batch:
        if example.reward < reward_bound:
            continue
        train_obs.extend(map(lambda step: step.state, example.steps))
        train_act.extend(map(lambda step: step.action, example.steps))
    train_obs_vector = torch.FloatTensor(train_obs)
    train_act_vector = torch.LongTensor(train_act)
    return train_obs_vector, train_act_vector, reward_bound, reward_mean

def iterative_batches(env, network, batch_size = 16):
    batch = []
    episode_reward = 0.0
    episode_steps = []
    state = env.reset()
    softmax = nn.Softmax(dim =1)

    while True: 
        state_vector = torch.Tensor([state])
        action_probs_vector = softmax(network(state_vector))
        
        action_probs = action_probs_vector.data.numpy()[0]
        action = np.random.choice(len(action_probs), p = action_probs)

        next_state, reward, done, _ = env.step(action)
        episode_reward += reward
        episode_steps.append(EpisodeStep(state = state, action = action))

        if done:
            batch.append(Episode(reward = episode_reward, steps = episode_steps))
            episode_reward = 0.0
            episode_steps = []
            next_state = env.reset()
            if len(batch) == batch_size:
                yield batch
                batch = []
        state = next_state

if __name__ == "__main__":
    env = gym.make("CartPole-v0")
    env = gym.wrappers.Monitor(env, directory = "mon", force = True)
    output_shape = env.action_space.n
    input_shape = env.observation_space.shape[0]

    network = Network(input_shape = input_shape, output_shape = output_shape)
    objective = nn.CrossEntropyLoss()
    optimizer = optim.Adam(params = network.parameters(), lr = 0.01)
    writer = SummaryWriter()

    for iter_no, batch in enumerate(iterative_batches(env, network)):
        state_vector, action_vector, reward_bound, reward_mean = filter_batch(batch)
        optimizer.zero_grad()
        action_values_vector = network(state_vector)
        loss_vector = objective(action_values_vector, action_vector)
        loss_vector.backward()
        optimizer.step()
        print("{}: loss = {}, reward_mean = {}, reward_boundary = {}".format(iter_no, loss_vector.item(), reward_mean, reward_bound))
        writer.add_scalar("loss", loss_vector.item(), iter_no)
        writer.add_scalar("reward mean", reward_mean, iter_no)
        writer.add_scalar("reward boundary", reward_bound, iter_no)
        if reward_mean > 199:
            print("Solved CartPole Problem!")
            break
    writer.close()