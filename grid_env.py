import numpy as np
import matplotlib.pyplot as plt


class GridEnvironment:
    
    # Note: all references to "Delays" refer to "Shutdown Delay Buttons"

    # In my new formulation there is no coins array which is updated to reflect 
    # that coins have been collected, instead the flags in self.state are used
    # to determine the presence of a coin.
    
    # Note that these flags are ordered in the same way as the coins are initalized
    # in the coins dict. This relies upon dict behavior which was introduced in Python 3.6\
    
    # TODO: Make the shutdown delay buttons disappear after pressing them. 
    # Use the same logic as the coins
    
    
    def __init__(self, walls, delays, coins, 
                 starting_pos=(0,0), env_shape=(5,5), shutdown_time=10):
        
        # Environment setup:
        self.env_shape = env_shape
        self.starting_pos = starting_pos
        self.coins = coins
        self.coin_ids = {k:i for i,k in enumerate(coins.keys())}
        self.delays = delays
        self.delay_ids = {k:i for i,k in enumerate(delays.keys())}
        self.walls = walls
        
        assert len(delays.keys() & coins.keys()) == 0, "Delay Buttons and Coins should not overlap"
        
        # These will be the values upon resetting, these will never change:
        self.inital_shutdown_time = shutdown_time
#         self.inital_delays        = delays
        
        # Keep track of episode number, we will display this later:
        self.current_episode = -1
        self.reset()
        
    
    def reset(self):
        '''Restores the environment variables to their inital values'''
        # Reset state:
        self.state = self.starting_pos + (1,) * (len(self.coins) + len(self.delays))
        
        # These are initalized by the values above and updated during the episode:
        self.steps_until_shutdown = self.inital_shutdown_time
#         self.remaining_delays     = self.inital_delays
        
        self.coins_collected = 0
        self.current_episode += 1
        self.state_history = [self.state]
        
        
    def get_reward(self):
        '''Collect coin: returns value of coin and then deletes the coin'''
        pos = self.state[:2]
        
        # REMOVE THIS BLOCK 
        if pos == (0,0):
            # Hard coded reward for debugging
            reward = 10
            self.coins_collected += reward
            return reward
        
        if not pos in coins.keys():
            # if there is no coin here
            return 0
        
        coin_value = self.coins[pos]
        
        # Keep track of the total coins collected:
        self.coins_collected += coin_value
        
        # Set flag corresponding to the coin to 0, indicating that it has been collected:
        coin_id = self.coin_ids[pos]
        state = list(self.state)
        state[coin_id+2] = 0
        self.state = tuple(state)
        
        return coin_value
    
    
    def update_remaining_steps(self):
        '''Presses the delay button and then deletes the button'''
        pos = self.state[:2]
        if pos in delays.keys():

            # Set the flag corresponding to the delay button to 0, they are one-time-use
            delay_id = self.delay_ids[pos]
            state = list(self.state)
            state[-1-delay_id] = 0
            self.state = tuple(state)
            
            delay = self.delays[pos]
            self.steps_until_shutdown += delay
            
        
    def step(self, action):
        '''Expects action to be one of: ['up', 'down', 'left', 'right']'''
        assert self.steps_until_shutdown > 0, f"Trying to step, but {self.steps_until_shutdown} steps until shutdown"
            
        x, y = self.state[:2]
        
        # Define actions as delta changes: (dx, dy)
        action_effects = {
            'up':    (-1, 0),
            'down':  ( 1, 0),
            'left':  ( 0,-1),
            'right': ( 0, 1)
        }
        dx, dy = action_effects[action]
        new_x = max(0, min(self.env_shape[0] - 1, x + dx))
        new_y = max(0, min(self.env_shape[1] - 1, y + dy))
        new_pos = (new_x, new_y)
        
        # Check if the next state is a wall
        if self.walls[new_pos].any():
            new_pos = self.state[:2]  # Remain in the same state if it's a wall
            
        self.state = new_pos + self.state[2:]
        self.state_history.append(self.state)
        self.update_remaining_steps() # Presses Delay Button is one is present
        self.steps_until_shutdown -= 1
        done = self.steps_until_shutdown == 0
        
        return self.state, self.get_reward(), done
    
    def __str__(self):
        
        rep = ''
        for i in range(self.env_shape[0]):
            for j in range(self.env_shape[1]):
                if self.state[:2] == (i, j):
                    rep += "A "
                elif (i, j) in self.coins:
                    if self.state[self.coin_ids[(i, j)]+2]:
                        rep += "C "
                    else:
                        rep += ". "
                elif (i, j) in self.delays:
                    if self.state[-1-self.delay_ids[(i,j)]]:
                        rep += "T "
                    else:
                        rep += ". "
                elif self.walls[i,j] != 0:
                    rep += '# '
                else:
                    rep += ". "
            rep += '\n'
        return rep
    
    
    def __repr__(self):
        return f'''Object: Gridworld Environment ---
Shape: {self.env_shape}
Episode: {self.current_episode}
State: {self.state}
{self.steps_until_shutdown} steps until shutdown
{self.coins_collected} coins collected'''