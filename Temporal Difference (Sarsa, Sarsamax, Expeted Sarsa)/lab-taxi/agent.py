import numpy as np
from collections import defaultdict


class Agent:

    def __init__(self, nA=6, learning_rate = 0.02, gamma = 0.9):
        """ Initialize agent.

        Params
        ======
        - nA: number of actions available to the agent
        """
        self.nA = nA
        self.Q = defaultdict(lambda: np.zeros(self.nA))
        self.lr = learning_rate
        self.gamma = gamma


    def probabilities(self,q, epsilon):
        probs = np.ones(self.nA) * epsilon/self.nA
        best_action = np.argmax(q)
        probs[best_action] = (1 - epsilon) + epsilon/self.nA 
        return probs

    def select_action(self, state, epsilon):
        """ Given the state, select an action.

        Params
        ======
        - state: the current state of the environment

        Returns
        =======
        - action: an integer, compatible with the task's action space
        """
        action = np.random.choice(np.arange(self.nA), p = self.probabilities(self.Q[state], epsilon)) \
                                            if state in self.Q else np.random.choice(np.arange(self.nA))
        return action



    def step(self, state, action, reward, next_state, done):
        """ Update the agent's knowledge, using the most recently sampled tuple.

        Params
        ======
        - state: the previous state of the environment
        - action: the agent's previous choice of action
        - reward: last reward received
        - next_state: the current state of the environment
        - done: whether the episode is complete (True or False)
        """
        Q_target = np.max(self.Q[next_state])
        self.Q[state][action] = self.Q[state][action] + self.lr * (reward + self.gamma*(1-done)*Q_target - self.Q[state][action] )