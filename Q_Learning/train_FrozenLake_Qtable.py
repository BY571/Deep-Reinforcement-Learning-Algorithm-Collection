import pandas as pd 
import numpy as np 
import gym
import time
import matplotlib.pyplot as plt
import argparse 



def make_Q_table(actions,n_states):
    table = pd.DataFrame(
        np.zeros((n_states, actions)), columns = list(range(actions)))     # q_table initial values
    # print(table)    # show table
    return table
    
def choose_action(state, q_table):
    state_actions = q_table.iloc[state, :]
    if (np.random.uniform() > EPSILON) or ((state_actions == 0).all()):  # act non-greedy or state-action have no value
        action_name = np.random.choice(ACTIONS)
    else:   # act greedy
        action_name = state_actions.idxmax()    # replace argmax to idxmax as argmax means a different function in newer version of pandas
    #print("Action_choosen: "+str(action_name))
    return action_name


def RL(ACTIONS,N_SPACE):
    q_table = make_Q_table(ACTIONS,N_SPACE)
    reward_list = []
    try_list = []
    
    for episode in range(EPISODES):
        S = env.reset()
        rewards = 0
        steps = 0
        for one_try in range(TRYS):  #how long one epidsode lasts

            env.render()
            A = choose_action(S, q_table)

            S_,R,done,info = env.step(A)
            #print(S_)
            #time.sleep(1)
            print()
            q_old = q_table.loc[S, A]  #Current Q-Value of the state
            q_learned = R + GAMMA * q_table.iloc[S_, :].max()
            q_table.loc[S, A] += ALPHA * (q_learned - q_old)  # update
            S = S_  # move to next state
            rewards += R
            steps = one_try
            if done:
                print("Episode finished after {} timesteps".format(one_try+1))
                steps = one_try+1
                break
        reward_list.append(rewards)
        try_list.append(steps+1)


    return q_table,reward_list,try_list






parser = argparse.ArgumentParser()
parser.add_argument("-e", "--Episoden",type = int,help ="Die Anzahl der zu trainierenden Episoden")
parser.add_argument("-a", "--Alpha",type = float,help ="Learning Rate ~0.1")
parser.add_argument("-g", "--Gamma",type = float,help ="Discount Factor ~0.9")
parser.add_argument("-eps", "--Epsilon",type = float,help ="Epsilon- for the Epsilon-Greedy decision process ~0.9")

args = parser.parse_args()

EPISODES = args.Episoden
TRYS = 100
EPSILON = args.Epsilon # epsilon greedy
ALPHA = args.Alpha # learning rate
GAMMA = args.Gamma #discount factor

if __name__ =="__main__":


    env = gym.make("FrozenLake-v0")
    print(gym.__version__)
    env.reset()
    # getting space and action
    ACTIONS = env.action_space.n   #env.unwrapped.get_action_meanings() to get a list of the action names
    N_SPACE = env.observation_space.n
    #print(ACTIONS)
    #print(N_SPACE)
    q_table,rlist,steps = RL(ACTIONS,N_SPACE)
    
    plt.plot(rlist)
    plt.title("Received Rewards")
    plt.xlabel("Epochs")
    plt.ylabel("Rewards")
    plt.show()

    plt.plot(steps)
    plt.title("Needed steps to finish one episode")
    plt.xlabel("Epochs")
    plt.ylabel("Steps")
    plt.show()


    
    
    
    print("Q-Table: \n")
    print(q_table)

    print("\nDo you want to save the Q-Table? \n")
    answer = input("[y/n]")

    if answer == "y":
        q_table.to_pickle("./Q_Table_E{}_a{}_g{}_eps{}.pkl".format(EPISODES,ALPHA,GAMMA,EPSILON))
    else:
        pass

