import pickle as pkl 
import numpy as np
import pandas as pd
import time
import gym 
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-e", "--Episoden",type = int,help ="Die Anzahl der zu trainierenden Episoden")
parser.add_argument("-v", "--Video",type = bool,help ="Sollen die Versuche in einem Video aufgezeichnet werden?")
parser.add_argument("-q", "--Q_Table",type = str,help ="Name der Q_table mit der gespielt werden soll")


args = parser.parse_args()

EPISODES = args.Episoden
TRYS = 100
AUFZEICHNUNG = args.Video 
Q_Table_name = args.Q_Table 




def load_Qtable(Q_table):
    Q = pd.read_pickle(Q_table)
    return Q


def choose_action(state,Q_table):
    state_actions = Q_table.iloc[state, :]
    action_name1 = state_actions.idxmax()
    state_actions.pop(action_name1)
    action_name2 = state_actions.idxmax()
    if (np.random.uniform() > 0.4):
        print("Best action choosen!")
        return action_name1
    else:
        print("Second-best-action choosen!")
        return action_name2 

def play():
    Q_Table = load_Qtable(Q_Table_name)
    for episode in range(EPISODES):
        S = env.reset()
        for one_try in range(TRYS):  #how long one epidsode lasts

            env.render()
            A = choose_action(S, Q_Table)
            print("Action choosen: {}".format(A))
            S_,R,done,info = env.step(A)
            #print(S_)
            time.sleep(2)

            # Addapting for further learning
            #print()
            #q_old = q_table.loc[S, A]  #Current Q-Value of the state
            #q_learned = R + GAMMA * q_table.iloc[S_, :].max()
            #q_table.loc[S, A] += ALPHA * (q_learned - q_old)  # update
            #S = S_  # move to next state
            
            if done:
                print("Episode finished after {} timesteps".format(one_try+1))
                break


if __name__ =="__main__":


    env = gym.make("FrozenLake-v0")
    print(gym.__version__)
    env.reset()

    play()

    # 0 - Down
    # 1 - 