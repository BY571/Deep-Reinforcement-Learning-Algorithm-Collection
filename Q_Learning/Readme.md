[image1]: ./Img/Q_value.png "Calculation Equation"
[image2]: ./Img/Q_table10000.png "Calculation Equation"
[image3]: ./Img/Receivedrewards.png "Calculation Equation"
[image4]: ./Img/steps_taken.png "Calculation Equation"





# Q-Learning and Q-Table

## Creating the Q-Table
The Q-Table gets created by the number of states (n_states) and the number of actions (n_actions) and form a matrix: n_states x n_actions 

This even shows the limitations of normal Q-learning with a Q-Table. The number of states has to be finit and not too large. Further, the states are not allowed to change during the game. 

## Calculating the Q-Values

The Q-Values get calculated each step by this formula:

![alt text][image1]

Here are as well some limitations. Since the Q-Values are dependent on the given rewards and most of the time the only reward is given when reaching the goal state, there has to be a way to reach the goal state by random actions. Otherwise the Q-Table will stay as a table of zeros.

## Testing on Open AI Gyms environment FrozenLake
After Training of 10000 Epochs the following Q-Table got calculated:

![alt text][image2]

Also by looking at the received rewards over the epochs, one can see that after epoch ~1500 almost every following try received an reward of 1 or better won the game.

![alt text][image3]

Same with the steps taken,one can see the increase in taken steps. which happens, since the game doesnt get stopped earlier by failing.

![alt text][image4]

