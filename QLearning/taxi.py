import numpy as np
import gym
import random
import time

class QLearner:
    q_action_size = 0
    q_state_size  = 0

    def __init__(self, action_size, state_size, default_value=0):
        self.q_action_size = action_size
        self.q_state_size  = state_size
        self.q_table = np.full((state_size, action_size), default_value)

    def get_action(self, state, epsilon=1.0):
        exp_exp_tradeoff = random.uniform(0,1)

        if(exp_exp_tradeoff > epsilon):
            return np.argmax(self.q_table[state,:])
        else:
            return -1

    def update(self, state, action, learning_rate, gamma, reward):
        self.q_table[state, action] = self.q_table[state, action] + learning_rate * (reward + gamma * np.max(self.q_table[new_state, :]) - self.q_table[state, action])


# Print iterations progress
def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end = printEnd)
    # Print New Line on Complete
    if iteration == total:
        print()


if __name__ == "__main__":
    env = gym.make("Taxi-v3")

    agent = QLearner(env.action_space.n, env.observation_space.n)

    epsilon = 1.0

    total_eps = 10000
    printProgressBar(0, total_eps, prefix = 'Progress:', suffix = 'Complete', length = 50)

    # Learn
    for episode in range(total_eps): #episodes
        state = env.reset()
        step  = 0
        done  = False

        for step in range(99):
            action = agent.get_action(state, epsilon)
            if(action == -1):
                action = env.action_space.sample()

            new_state, reward, done, info = env.step(action)

            agent.update(state, action, 0.7, 0.618, reward)

            state = new_state

            if done:
                break

        printProgressBar(episode + 1, total_eps, prefix = 'Learning:', suffix = 'Complete', length = 50)
        epsilon = 0.01 + (1.0 - 0.01)*np.exp(-0.01*episode)


    # Play
    rewards = []
    for episode in range(100):
        state = env.reset()
        step = 0
        done = False
        total_rewards = 0

        for step in range(99):
            #env.render()
            action = agent.get_action(state, -1)

            new_state, reward, done, info = env.step(action)
            total_rewards += reward

            if done:
                rewards.append(total_rewards)
                break
            state = new_state

    env.close()
    print ("Score over time: " +  str(sum(rewards)/100))

    action_size = env.action_space.n
