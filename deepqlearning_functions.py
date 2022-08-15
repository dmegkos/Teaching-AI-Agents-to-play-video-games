# These functions are used in DRL_Part_2 (Deep Q-Learning)

# Import libraries
# General imports
import numpy as np
import random
from collections import namedtuple
# Neural Networks imports
import torch
import torch.nn as nn
import torch.nn.functional as F
# Visualization imports
import matplotlib.pyplot as plt
# Environment import
import gym
# Seed for reproducibility
torch.manual_seed(10)


#### Define Environment functionality class ####
# Code inspired by https://deeplizard.com/learn/video/jkdXDinWfo8
# A class that loads and handles an OpenAI Gym game
class EnvManager():
    def __init__(self, env_name, device):
        self.env = gym.make(env_name).unwrapped  # load environment
        self.env.reset()  # reset to initial state
        self.done = False  # track if the episode has ended
        self.current_state = None  # storing the current state
        self.device = device  # cpu or gpu

    # reset environment and set the current state
    def reset(self):
        self.current_state = self.env.reset()

    # close the environment
    def close(self):
        self.env.close()

    # render game image
    def render(self, mode='human'):
        return self.env.render(mode)

    # get the number of environment's available actions
    def num_actions_available(self):
        return self.env.action_space.n

    # excecute the given action and get the reward
    def take_action(self, action):
        self.current_state, reward, self.done, _ = self.env.step(action.item())
        return torch.tensor([reward], device=self.device)

    # get the state after excecuting the action
    def get_state(self):
        if self.done:  # if the action ended the episode
            return torch.zeros_like(torch.tensor(self.current_state), device=self.device).float()
        else:
            # return the new state after the action
            return torch.tensor(self.current_state, device=self.device).float()

    # number of features in a state, for the NN
    def n_state_features(self):
        return self.env.observation_space.shape[0]


#### Define our Deep Q Network class ####
# Code inspired by Lab 06
# Three hidden layers
# takes as input the state features
# outputs q values for each action
class DQN(nn.Module):
    def __init__(self, n_state_features, hidden_size, n_actions, isdueling):
        super().__init__()
        self.isdueling = isdueling  # Boolean for Dueling architecture
        # Input -> First hidden layer
        self.fc1 = nn.Linear(n_state_features, hidden_size)
        # First hidden layer -> Second hidden layer
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        # Second hidden layer -> Third hidden layer
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        # Check if network architecture is Dueling Network
        if self.isdueling:
            # The Value V(s) of being at the state
            self.V = nn.Linear(hidden_size, 1)
            # The Advantage A(s,a) of taking the action at the state
            self.A = nn.Linear(hidden_size, n_actions)
        else:
            # Third hidden layer -> Output
            self.out = nn.Linear(hidden_size, n_actions)

    def forward(self, x):
        # Pass input through layers and activate it
        # using ReLu activation function
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        # Check if network architecture is Dueling Network
        if self.isdueling:
            V = self.V(x)  # The Value V(s)
            A = self.A(x)  # The Advantage A(s,a)
            x = V + (A - A.mean())  # Combine streams
            return x
        else:
            x = self.out(x)
            return x


#### Define an Experience Class for Experience Replay ####
# Code inspired by Lab 06
# Agent's experiences will be saved in a named tuple
# Each experience will have the state, action, the new state and the reward
Experience = namedtuple(
    'Experience', ('state', 'action', 'new_state', 'reward'))


#### Process Tensors ####
# Code inspired by https://deeplizard.com/learn/video/kF2AlpykJGY
# and https://deeplizard.com/learn/video/ewRw996uevN
def extract_tensors(experiences):
    # Convert batch of Experiences to Experience of batches
    # using zip
    batch = Experience(*zip(*experiences))

    t1 = torch.stack(batch.state)
    t2 = torch.cat(batch.action)
    t3 = torch.cat(batch.reward)
    t4 = torch.stack(batch.new_state)

    return (t1, t2, t3, t4)


#### Define Replay Memory Class for storing and sampling experiences ####
# Code inspired by Lab 06
# This class will store agent's new experiences
# and sample experiences based on a batch size
class ReplayMemory():
    def __init__(self, capacity):
        self.capacity = capacity  # the capacity of the replay memory
        self.memory = []
        self.position = 0  # counter for position

    # store an experience to the memory
    def store(self, experience):
        if len(self.memory) < self.capacity:  # if there is room available
            self.memory.append(experience)  # store experience
        else:  # if the experience memory is full
            # replace oldest experience
            self.memory[self.position % self.capacity] = experience
        self.position += 1  # increase counter

    # sample a batch of experiences from the memory
    def get(self, batch_size):
        return random.sample(self.memory, batch_size)

    # check if there are enough experiences to sample
    def batch_available(self, batch_size):
        return len(self.memory) >= batch_size


#### Define Epsilon Greedy Class ####
# Gets as input the minimum, maximum and decay rate values of epsilon
# Returns epsilon greedy strategy based on current step
class EpsilonGreedyValue():
    def __init__(self, max_epsilon, min_epsilon, epsilon_decay_rate):
        self.max_epsilon = max_epsilon
        self.min_epsilon = min_epsilon
        self.epsilon_decay_rate = epsilon_decay_rate

    # get epsilon greedy strategy
    def epsilon_strategy(self, current_step):
        return self.min_epsilon + (self.max_epsilon - self.min_epsilon) * \
            np.exp(-1. * current_step * self.epsilon_decay_rate)


#### Define our Agent Class ####
# The agent will choose an action from the list of
# available actions based on the epsilon greedy strategy
class DRLAgent():
    def __init__(self, strategy, n_actions, device):
        self.current_step = 0  # store n of steps
        self.strategy = strategy  # epsilon greedy strategy
        self.n_actions = n_actions  # available actions
        self.device = device  # cpu or gpu

    # agent selects an action using the policy network
    # based on the current state
    def select_action(self, state, policy_net):
        # get epsilon greedy strategy
        rate = self.strategy.epsilon_strategy(self.current_step)
        self.current_step += 1

        # exploration vs exploitation
        if rate > random.random():  # explore
            # select a random action
            action = random.randrange(self.n_actions)
            return torch.tensor([action]).to(self.device)
        else:  # exploit
            with torch.no_grad():
                # pass the state to the network and get the max q value action
                return policy_net(state).unsqueeze(dim=0).argmax(dim=1).to(self.device)


#### Define Q-Values Calculator class ####
# This class will handle the calculation the Q Values
# Gets states and actions sampled from replay memory
# Policy network will return the current state Q-Values
# Target network will return the next state Q-Values
# Code source https://deeplizard.com/learn/video/ewRw996uevM
class QValues():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @staticmethod
    # Use the policy network to get current state q values
    def get_current(policy_net, states, actions):
        return policy_net(states).gather(dim=1, index=actions.unsqueeze(-1))

    @staticmethod
    # Use the target network to get next state q values
    def get_next(target_net, next_states):
        # find if there are final states to the given batch
        # get their location
        final_state_locations = next_states.flatten(start_dim=1) \
            .max(dim=1)[0].eq(0).type(torch.bool)
        # get the location of states that are not final
        non_final_state_locations = (final_state_locations == False)
        # filter the non final states
        non_final_states = next_states[non_final_state_locations]
        # get the batch size
        batch_size = next_states.shape[0]
        # create torch to store the values of the next states
        values = torch.zeros(batch_size).to(QValues.device)
        # pass the states that are not final to the target net
        # get the output and store to values
        values[non_final_state_locations] = target_net(
            non_final_states).max(dim=1)[0].detach()
        return values


#### Function that calculates and plots cumulative moving average ####
# Inputs: list containing rewards or timesteps per episode, x and y labels
# Outputs: CMA Plot
def plot_cma(rt_ep_list, ylabel):
    # Program to calculate cumulative moving average
    # Inspired by https://www.geeksforgeeks.org/how-to-calculate-moving-averages-in-python/

    # Convert list to array
    rt_ep_array = np.asarray(rt_ep_list)

    i = 1
    # Initialize an empty list to store cma
    cma = []

    # Store cumulative sums of array in cum_sum array
    cum_sum = np.cumsum(rt_ep_array)

    # Loop through the array elements
    while i <= len(rt_ep_array):

        # Calculate the cumulative average by dividing
        # cumulative sum by number of elements till
        # that position
        window_average = round(cum_sum[i-1] / i, 2)

        # Store the cumulative average of
        # current window in moving average list
        cma.append(window_average)

        # Shift window to right by one position
        i += 1

    # Plot evaluation metrics
    plt.figure()
    plt.title("Cumulative Moving Average")
    plt.xlabel("Episodes")
    plt.ylabel(ylabel)
    pcma = plt.plot(cma)


#### Function that calculates rewards moving average ####
# Inputs: list containing rewards, window size and a boolean for plotting
# Outputs: Final Moving Average or Plot
def reward_ma(rt_ep_list, w_size, plot_metric):
    # Program to calculate moving average using numpy
    # Inspired by https://www.geeksforgeeks.org/how-to-calculate-moving-averages-in-python/

    # Convert list to array
    rt_ep_array = np.asarray(rt_ep_list)
    window_size = w_size

    i = 0
    # Initialize an empty list to store moving average
    moving_averages = []

    # Loop through the array
    # consider every window of size w_size
    while i < len(rt_ep_array) - window_size + 1:

        # Calculate the average of current window
        window_average = round(np.sum(rt_ep_array[
            i:i+window_size]) / window_size, 2)

        # Store the average of current
        # window in moving average list
        moving_averages.append(window_average)

        # Shift window to right by one position
        i += 1

    # if flagged to plot the moving average
    if plot_metric:
        # Plot evaluation metrics
        plt.figure()
        plt.title("Moving Average")
        plt.xlabel("Episodes")
        plt.ylabel("Rewards")
        pma = plt.plot(moving_averages)
    else:
        # return the last reward moving average
        return moving_averages[-1]
