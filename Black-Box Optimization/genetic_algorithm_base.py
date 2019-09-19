import copy
import torch
import torch.nn as nn 
from torch.utils.tensorboard import SummaryWriter
import numpy as np 
import argparse
import gym 


class Network(nn.Module):
    def __init__(self,state_size,action_size,hidden_layer,seed):
        super(Network, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.hidden_layer = hidden_layer
        
        self.net = nn.Sequential(
                nn.Linear(self.state_size, self.hidden_layer),
                nn.ReLU(),
                nn.Linear(self.hidden_layer, self.hidden_layer),
                nn.ReLU(),
                nn.Linear(self.hidden_layer, self.action_size),
                nn.Softmax(dim = 1))

    def forward(self, x):
        return self.net(x)

def evaluate(env, net):
    """
    Plays a round of the game and returns the obtained reward
    """
    state = env.reset()
    rewards = 0
    while True:
        state = torch.from_numpy(state).unsqueeze(0).float()
        action_prob = net(state)
        action = action_prob.max(dim=1)[1] #argmax
        next_state, reward, done, info = env.step(action.data.numpy()[0])
        rewards += reward 
        state = next_state
        if done:
            break
    return rewards

def mutate_parent(net):
    """
    Mutates the parent neural nets by adding noise sampled by a normal distribution.

    """
    new_net = copy.deepcopy(net)
    for parameter in new_net.parameters():
        noise = torch.tensor(np.random.normal(size=parameter.data.size()).astype(np.float32))
        parameter.data += NOISE_STD * noise 
    return new_net


if __name__ == "__main__":
    # parse input values like
    # - Noise standard deviation [NOISE_STD]
    # - Population size [POPULATION_SIZE]
    # - Parents count [PARENTS_COUNT] 

    parser = argparse.ArgumentParser(description = "Noise, Population size, Parents count")
    parser.add_argument("--noise",type = float,default=1e-2)
    parser.add_argument( "--ps",type=int,default=50)
    parser.add_argument( "--pc",type=int,default=10)

    args = parser.parse_args()
    NOISE_STD = args.noise  
    POPULATION_SIZE = args.ps
    PARENTS_COUNT = args.pc 

    #print(f"Noise: {NOISE_STD}, PopS: {POPULATION_SIZE}, PARENTS_COUNT: {PARENTS_COUNT}")
    np.random.seed(seed=42)
    torch.manual_seed(42)
    writer = SummaryWriter(comment="-CartPole")
    env = gym.make("CartPole-v0")
    gen_idx = 0
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n 

    nets = [Network(state_size, action_size, hidden_layer=32, seed=3) for _ in range(POPULATION_SIZE)] 
    population = [(net, evaluate(env, net)) for net in nets]

    while True:
        population.sort(key=lambda p: p[1], reverse=True)    # sorts the fitness from highest to lowest
        rewards = [p[1] for p in population[:PARENTS_COUNT]]    # takes the fitness of the top x-parents 
        reward_mean = np.mean(rewards)
        reward_max = np.max(rewards)
        reward_std = np.std(rewards)

        writer.add_scalar("reward_mean", reward_mean, gen_idx)
        writer.add_scalar("reward_max", reward_max, gen_idx)
        writer.add_scalar("reward_std", reward_std, gen_idx)
        print(f"Generation: {gen_idx} | Reward_mean: {reward_mean} | Reward_max: {reward_max} | Reward_std: {reward_std}")

        if reward_mean > 199:
            print("Solved the environment in {} generations".format(gen_idx))
            break
            writer.close()

        prev_population = population 
        population = [population[0]]    # list of the nets

        for _ in range(POPULATION_SIZE-1):
            parent_idx = np.random.randint(0, PARENTS_COUNT)    #sample the new population from the top x-parents
            parent = prev_population[parent_idx][0]
            net = mutate_parent(parent)
            population.append((net, evaluate(env, net)))

        gen_idx += 1
        
         
