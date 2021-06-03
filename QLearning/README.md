#QLearning

Q Learning is a value-based reinforcement learning algorithm, and as a reinforcement learning algorithm,
it is dependent on an agent being given positive and/or negative rewards depending on its actions.

One way to acomplish the learning part, is by introducing a table, which the agent updates depending on
its experience of the environment, with the maximum expected future reward, for each action at each state.

The Q-table, as it were, will be composed of rows for states and columns for actions, each Q-table score will
be the maximum expected future reward that the agent will get if it takes a particular action at a particular state.

The Q-Learning algorithm employs the an action-value function, which takes a state and an action as inputs, and returns
the expected future reward:

Q^pi(st, at) = E[ R_(t+1) + yR_(t+2) + (y^2)(R_(t+3)) + ... | st, at ]

Before exploring the environment the Q-table gives the same arbitrary value (usualy 0), and as we explore, the Q-table will give
us an increasingly better approximation, by iteratively updating Q(s,a) using the Bellman Equation.
