import pandas as pd 
import numpy as np 
import gym
import time

EPISODES = 5000
TRYS = 100
EPSILON = 0.9 # epsilon greedy
ALPHA = 0.1 # learning rate
GAMMA = 0.9 #discount factor




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
    for episode in range(EPISODES):
        S = env.reset()
        for one_try in range(TRYS):  #how long one epidsode lasts

            env.render()
            A = choose_action(S, q_table)

            S_,R,done,info = env.step(A)
            #print(S_)
            #time.sleep(1)
            q_old = q_table.loc[S, A]  #Current Q-Value of the state
            q_learned = R + GAMMA * q_table.iloc[S_, :].max()
            q_table.loc[S, A] += ALPHA * (q_learned - q_old)  # update
            S = S_  # move to next state  
            if done:
                print("Episode finished after {} timesteps".format(one_try+1))
                break


    return q_table


if __name__ =="__main__":
    env = gym.make("FrozenLake-v0")
    print(gym.__version__)

    env.reset()
    # getting space and action
    ACTIONS = env.action_space.n   #env.unwrapped.get_action_meanings() to get a list of the action names
    N_SPACE = env.observation_space.n
    #print(ACTIONS)
    #print(N_SPACE)
    q_table = RL(ACTIONS,N_SPACE)
    print("Q-Table: \n")
    print(q_table)


