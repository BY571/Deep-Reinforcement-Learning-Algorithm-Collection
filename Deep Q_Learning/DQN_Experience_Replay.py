import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam
import numpy as np 
from collections import deque
from keras.models import load_model
import random
import matplotlib.pyplot as plt
import gym
from gym import wrappers



class AI():
    def __init__(self, state_size, action_size, memory_size, learning_rate, gamma):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen = memory_size)
        

        # HYPERPARAMETER
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon  = 0.5
        self.epsilon_start = self.epsilon

        self.brain = self.build_brain()


    def build_brain(self):
        model = Sequential()
        model.add(Dense(self.state_size, activation='relu'))
        model.add(Dense(25, activation='relu'))
        #model.add(Dropout(0.3))
        model.add(Dense(25, activation='relu'))
        #model.add(Dropout(0.3))
  #      model.add(Dense(12, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss = "mse", optimizer = Adam(lr=self.learning_rate))
        return model


    def load_model(self, name):
        """
        Loads an existing Model
        Input: string of the model name - h5 data
        """
        brain = load_model(name)
        return None

    def save_learnings(self, model_name):#
        """
        Input string of Modelname
        """
        self.brain.save(model_name+".h5")
    
    def adapt_epsilon(self,ep):
        # Epsilon starts at 0.5 linear increasing to 0.99 by ep 4000:
        # linear: epsilon = 0.0001225*ep+self.epsilon_start
        # exponent (4000 eps): epsilon = self.epsilon_start + (ep/5714)**2
        if ep == 0:
            pass
        self.epsilon = self.epsilon_start + (ep/5714)**2 
    
    def act(self, state, status = "train"):
        if status == "train": 
            if np.random.rand() > self.epsilon:
                return random.randrange(self.action_size)
        return np.argmax(self.brain.predict(state)[0]) 
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay(self):
        batch_size = 32
        if len(self.memory) < batch_size: 
            return

        samples = random.sample(self.memory, batch_size) 
        for state, action, reward, next_state, done in samples:
            target = reward
            
            if not done:    
                target = reward + self.gamma * np.amax(self.brain.predict(next_state)[0]) # Predict the future/target value
                #print(target)
            Q_target_shape = self.brain.predict(state) # normal Q- Value prediction for the training-shape 
            Q_target_shape[0][action] = target # replacing the best Q-Value with the target 
            self.brain.fit(state, Q_target_shape, epochs=1, verbose=0) # training with the new Target value (loss = sum(Q_target-Q)exp2)




def play(Ep, agent, status = "train"):
    
    learning_graph = []
    env = gym.make("CartPole-v1")
    env = wrappers.Monitor(env, "Saved_DQN_ER_Models/", resume=True, video_callable=lambda episode_id: episode_id%250==0)
    action_space = env.action_space.n 
    state_space = env.observation_space.shape[0]
    if agent == None:
        agent = AI(state_space,action_space,memory_size = 5000,learning_rate = 0.001,gamma = 0.95) #2500 mem
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
        print("Episode {}# Score: {}".format(ep, score + 1))
        if ep == 250 or ep % 500 == 0:
                # save model eacht 500 ep for videos
                agent.save_learnings(str(ep)+","+str(score))
        agent.replay() 
        learning_graph.append(score)
    return learning_graph, agent

def main():
    Episodes = 4001 #4001
    graph,agent = play(Episodes,None)
    plt.plot(graph)
    plt.xlabel("Episoden")
    plt.ylabel("Score")
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
    x,y = play(int(n),agent,status = "play")

if __name__ == "__main__":
    main()
