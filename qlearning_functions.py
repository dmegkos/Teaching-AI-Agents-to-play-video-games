# These functions are used in DRL_Part_1 (Q-learning)

# Required imports
import numpy as np
import matplotlib.pyplot as plt

# A custom function that initializes our environment. The environment is inspired by OpenAI Gym's FrozenLake.
# The agent is placed in a 8x8 grid world, beginning always from the same state, trying to reach the other side
# of the lake. The agent navigates through a frozen path, trying to avoid a number of holes the lake has. If the
# agent falls into a hole, the agent starts again from the initial state. Because the path is frozen, the agent
# will sometimes move to a different direction than the one that was picked. There are 64 available states
# and four available actions (Up:0, Down:1, Left:2, Right:3). The 8x8 grid is defined below:
#
#  "8x8": [    SFFFHFFF
#              FHFFFHFF
#              FFHFFFHF
#              FFFHHFHF
#              FFHHFFFH
#              FFFHFHFH
#              FFHFFHHH
#              FHFFFFFG 
#              ] 
#
# Where S=Safe Spot (Agent follows the action that was chosen), F=Frozen, H=Hole, G=Goal
# Which can be mapped to the below states:
#
#  "8x8": [   0,1,2,3,4,5,6,7,
#             8,9,10,11,12,13,14,15 
#             16,17,18,19,20,21,22,23
#             24,25,26,27,28,29,30,31,
#             32,33,34,35,36,37,38,39,
#             40,41,42,43,44,45,46,47,
#             48,49,50,51,52,53,54,55,
#             56,57,58,59,60,61,62,63
#             ]

# Rewards for R matrix: S=0, F=-0.5, H=-1, G=1
# Output: S (set of states), A (set of actions), R Matrix, Q Matrix, state (starting state)
# goal state (the exit), hole state (the states that have holes)


def init_env():
    # Define set of states
    S = np.arange(64)

    # Define the goal
    goal_state = 63

    # Define the holes
    hole_state = [11, 13, 17, 22, 26, 27, 28, 34, 43, 45, 47, 48, 50, 53, 54, 55]

    # Define set of actions and map to numbers
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3
    A = [UP, DOWN, LEFT, RIGHT]

    # Initialize R matrix with dimensions A x S
    R = np.empty((len(S), len(A)))

    # Populate R matrix with possible actions and rewards
    R[:] = -0.5  # Populate with -0.5 (F)

    # Populate with 0s (Safe spot), action without random probability for the movement
    
    # R[8, 0] = 0
    # R[1, 2] = 0

    # Populate with 1s, the goal
    R[62, 3] = 100

    # Populate with -1s, the holes
    for i in hole_state:
        # the if statements avoid the index being out bounds
        if i - 1 < 0: 
            continue
        if i - 8 < 0: 
            continue
        if i + 1 > 63:
            continue
        if i + 8 > 63:
            continue
        # sets the states that lead to the holes to -10 
        R[i-1, 3] = -10
        R[i+1, 2] = -10
        R[i-8, 1] = -10
        R[i+8, 0] = -10


  # Populate with NaNs, actions that can't be executed (there's a wall)
    for i in range(8):  # upper wall - agent can't go up
        R[i, 0] = np.nan 
    for i in np.arange(0, 57, 8):  # left wall - agent can't go left 
        R[i, 2] = np.nan
    for i in np.arange(7, 64, 8):  # right wall - agent can't go right 
        R[i, 3] = np.nan
    for i in np.arange(56, 64):  # lower wall - agent can't go down 
        R[i, 1] = np.nan

    # Initialize Q matrix with dimensions A x S and populate with zeros
    Q = np.zeros(R.shape)

    # Initialize starting state
    state = S[0]  # We always begin from state 0
    #print("Starting state is '{}', which is row {} in the Q and R matrices".format(S[state], state))
    return S, A, R, Q, state, goal_state, hole_state


# A custom function that handles the selection of the action to be executed by the agent
# Input: R matrix, Q matrix, S (the list of all states), A (the list of actions), state (the current state of the agent)
# Output: action (the action the agent is going to execute)


def select_action(R, Q, S, A, state, epsilon):
    # Check all available actions
    available_actions = np.where(~np.isnan(R[state]))[0]  # Identify available actions
    # print("The available actions from state '{}' are: {}".format(
    # S[state], [A[x] for x in available_actions]))

    # The current Q values for each action
    q_values = [Q[state, action] for action in available_actions]
    #print('The Q values for those actions from current state are: {}'.format(q_values))

    # Select an action
    # Check the best actions
    best_actions = available_actions[np.where(q_values == np.max(q_values))[0]]

    # The current Q values of the best actions
    #best_actions_q_values = [Q[state, a] for a in best_actions]

    # if len(best_actions) > 1:  # In case there are more than one best actions
    #print('Detected multiple actions with identical Q values. Agent will randomly select one of these.')
    # print('Our best available actions from here are: {} with current q values: {}'.format(
    # [A[x] for x in best_actions], best_actions_q_values))

    # Epsilon-greedy
    if np.random.uniform() > epsilon:
        action = np.random.choice(available_actions)
        #print("Selecting random action '{}' with current Q value {}".format(A[action], Q[state, action]))
    else:
        action = np.random.choice(best_actions)
        #print("Selecting greedy action '{}' with current Q value {}".format(A[action], Q[state, action]))

    # Implement the Freezing Random action - this option has been deactivated as too challenging for the agent on our grid.
    # if state in range(64):
    #     if np.random.uniform() > 0.9:  # There is a 10% chance the agent "slips" and chooses a random action
    #         action = np.random.choice(available_actions)

    return action  # Return the action the agent is going to execute

# A custom function that handles the movement of the agent inside the 8x8 grid.
# Input: state (the current state of the agent), action (the action the agent is going to take)
# Output: state (the new state of the agent after executing the given action)


def select_step(state, action):
    if action == 0:
        state = state - 8  # Move up
    elif action == 1:
        state = state + 8  # Move down
    elif action == 2:
        state = state - 1  # Move left
    else:
        state = state + 1  # Move right
    return state

# A custom function that updates the relevant Q-Value of the agent.
# Input: alpha (learning rate), gamma (the discount parameter), Q matrix, state (current state),
# old_state (the previous state), r (the immediate reward), action
# Output: Q Matrix (the updated version)


def update_qvalue(alpha, gamma, Q, state, old_state, r, action):
    Q[old_state, action] = Q[old_state, action] + alpha * \
        (r + gamma * np.max(Q[state])-Q[old_state, action])

    return Q


# Function that calculates and plots cumulative moving average
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
    plt.title("Reward - Cumulative Moving Average")
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
        plt.title("Rewards Moving Average")
        plt.xlabel("Episodes")
        plt.ylabel("Rewards")
        pma = plt.plot(moving_averages)
    else:
        # return the last reward moving average
        return moving_averages[-1]


#### Function that calculates timestep moving average ####
# Inputs: list containing number of timesteps, window size and a boolean for plotting
# Outputs: Final Moving Average or Plot
def timestep_ma(ts_ep_list, w_size, plot_metric):
    # Program to calculate moving average using numpy
    # Inspired by https://www.geeksforgeeks.org/how-to-calculate-moving-averages-in-python/

    # Convert list to array
    rt_ep_array = np.asarray(ts_ep_list)
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
        plt.title("Steps Moving Average")
        plt.xlabel("Episodes")
        plt.ylabel("Steps")
        pma = plt.plot(moving_averages)
    else:
        # return the last reward moving average
        return moving_averages[-1]


#### Function used to generate the hyperparameters grid search ####
# Inputs: alpha, gamma, decay rate, epsilon
# Output: grid search on gols hit, number of rewards collected, average steps, hyperparameters
def trainig_grid(a, g, d, epsilon = 0.9):
    S, A, R, Q, state, goal_state, hole_state = init_env()
    max_epsilon = 0.9
    min_epsilon = 0.01
    num_ep = 10000  
    num_timestep = 500
    goal = 0
    goals = []  
    reward_ep_list = [] # List containing rewards
    ts_ep_list = [] # List containing steps 
    eps_decay = []  # List containing epsilons
    goals_print = [] # List containing goals to print
    steps_print = [] # List containing steps to print
    reward_print = [] # List containing rewards to print
    #run num_episodes episodes
    for episode in range(num_ep):

        #print("Starting state is '{}'".format(S[state]))
        
        # Initialize/Reset reward metric
        r_metric = 0
    
        goals.append(goal)

        for timestep in range(num_timestep):
            # Select action
            action = select_action(R,Q,S,A,state,epsilon)

            # Get immediate reward
            r = R[state,action]
            #print("Reward for taking action '{}' from state '{}': {}".format(A[action], S[state], r))

            # Sum the reward
            r_metric += r

            # Update the state - move agent
            old_state = state # Store old state
                
            state = select_step(state,action) # Get new state
            #print("After taking action '{}' from state '{}', new state is '{}'".format(A[action], S[old_state], S[state]))

            # Update Q-Matrix
            Q = update_qvalue(a,g,Q,state,old_state,r,action)

            if S[state] == goal_state:
                goal += 1
                goals[-1] = goal
                break
            elif S[state] in hole_state:
                break

        # Store metrics to lists
        ts_ep_list.append(timestep) # Number of timesteps
        reward_ep_list.append(r_metric) # Total episode rewards

        # Exploration rate decay 
        epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-d*episode)
        eps_decay.append(epsilon) # appends the updated epsilon to a list (used to plot this metric)

        state = S[0] # Start again from the beginning
        # print('Episode {} finished. Q matrix values:\n{}'.format(episode,Q.round(1)))
    # the below if statements are used to adjust the outputs lenghts -> display a better grid 
    if episode == num_ep-1:
        m = max(goals)
        goals_print.append(m)
        if len(str(m)) == 1:
            m = "   "+str(m)
        elif len(str(m)) == 2:
            m = "  "+str(m)
        elif len(str(m)) == 3:
            m = " "+str(m)
        s = sum(reward_ep_list)
        reward_print.append(s)
        if len(str(s)) == 8:
            s = " "+str(s)
        elif len(str(s)) == 7:
            s = "  "+str(s)
        t = sum(ts_ep_list)/num_ep
        steps_print.append(t)
        # prints outputs to the grid
        print('Goals hit: {} | Cumulative reward: {} | AVG steps: {:.0f} | Alpha: {} | Gamma: {} | Epsilon_s: {} | Epsilon_f: {:.2} | Decay Rate: {}'\
        .format(m, 
        s,
        t, 
        a, 
        g,
        max_epsilon,
        epsilon, 
        d))


#### Function used to display the episodes to screen ####
# Inputs: agents' actions
# Output: visual grid with agent moving onto it.
def display_episodes(actions):   
   
    # Creates the frozen lake matrix

    S = [['S', 'F', 'F', 'F', 'F', 'F', 'F', 'F'], 
        [ 'F', 'F', 'F', 'H', 'F', 'H', 'F', 'F'], 
        [ 'F', 'H', 'F', 'F', 'F', 'F', 'H', 'F'],
        [ 'F', 'F', 'F', 'H', 'F', 'F', 'F', 'F'],
        [ 'F', 'F', 'H', 'H', 'F', 'F', 'F', 'F'],
        [ 'F', 'F', 'F', 'H', 'F', 'H', 'F', 'H'],
        [ 'H', 'F', 'H', 'F', 'F', 'H', 'H', 'H'],
        [ 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'G']]

    # Import and initialize the pygame library
    import pygame
    from pygame.locals import QUIT
    clock = pygame.time.Clock()

    # Set the parameters for the main window 
    SCREEN_WIDTH = 1000 
    SCREEN_HEIGHT = 1000

    # Set up the drawing window
    screen = pygame.display.set_mode([SCREEN_WIDTH, SCREEN_HEIGHT])

    # Display window name
    pygame.display.set_caption("Frozen Lake")

    # Initialize pygame
    pygame.init()

    # Run until the user asks to quit

    running = True

    while running:
        # Did the user click the window close button?    
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # Fill the background with white
        screen.fill((255, 255, 255))

        # LOAD THE ENVIRONMENT IMAGES 

        hole = pygame.image.load('hole.jpeg')  # load image
        hole = pygame.transform.scale(hole, (SCREEN_WIDTH/8, SCREEN_WIDTH/8)) # rescale the image to be a perfect square that fits 1/4 of the screen. 
        frozen = pygame.image.load('frozen.jpeg')
        frozen = pygame.transform.scale(frozen, (SCREEN_WIDTH/8, SCREEN_WIDTH/8)) 
        goal = pygame.image.load('goal2.jpeg')
        goal = pygame.transform.scale(goal, (SCREEN_WIDTH/8, SCREEN_WIDTH/8)) 

        # CREATE ENVIRONMENT (sets the photos into the right slots) FROM THE MATRIX
        def print_screen():  
            for r in range(8):
                for c in range(8):
                    x = c * 125 # each slot is 125 x 125 pixels 
                    y = r * 125
                    if S[r][c] == 'S':
                        image = frozen
                    elif S[r][c] == 'F':
                        image = frozen
                    elif S[r][c] == 'G':
                        image = goal
                    else:
                        image = hole

                    screen.blit(image, (x, y)) #display the images to screen 

    # Define a player object by extending pygame.sprite.Sprite
    # The surface drawn on the screen is now an attribute of 'player'
        class Player(pygame.sprite.Sprite):
            def __init__(self):
                super(Player, self).__init__()
                self.surf = pygame.image.load('ninja2.png') # load the image
                self.surf = pygame.transform.scale(self.surf, (SCREEN_WIDTH/12, SCREEN_WIDTH/12)) # rescale the image
                self.rect = self.surf.get_rect() # this gets the images property (position and size)
                self.rect.x = 0 # set player horizontal position more central (by default the images are set with their up left corner to position 0,0)
                self.rect.y = 0 # set player vertical position more central
            
            # set the player picture to move properly from slot to slot, depending on the action. 
            def move(self, actions):
                if actions == 0: # move up
                    self.rect.x += 0
                    self.rect.y += -125 
                elif actions == 1: # move down 
                    self.rect.x += 0
                    self.rect.y += 125
                elif actions == 2: # move left
                    self.rect.x += -125
                    self.rect.y += 0
                elif actions == 3: # move right
                    self.rect.x += 125
                    self.rect.y += 0
                elif actions == 5: # restart
                    self.rect.x = 0
                    self.rect.y = 0


        player = Player()   # initialise a player instance          
        
        # this is a loop to try and see if the players moves okay. 
        for i in actions: # for now this takes the actions from the list of actions defined at the top
        # screen.blit(image, (x, y)) #display the images to screen
            player.move(i) # uses the above function to move the player
            print_screen()
            screen.blit(player.surf, (player.rect.x, player.rect.y)) # make the player move
            clock.tick(100)  # this sets the speed of the movements. If you don't set it it finishes all the movements before you even see them. 
            pygame.display.update() # this updates the frames on the screen.
        
        # this stops the game when the player has reached the final goal. 
            if i == 6:
                running = False
                
    # Done! Time to quit.
    pygame.quit()


