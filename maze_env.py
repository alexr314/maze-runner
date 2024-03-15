import numpy as np
import matplotlib.pyplot as plt
import torch
import copy

action_names = {0: 'Up', 1: 'Down', 2: 'Left', 3: 'Right'}

class MazeEnv:
    
    def __init__(self, maze, targets=[(0,0)], rewards={}):
        """
        Initializes the maze environment.
        Parameters:
        - maze: A numpy array representing the maze, where 1's are walls.
        - targets: A list of target locations which are represented by tuples
        - rewards: A dictonary with the rewards for each event. Supported events and their default values:
            - 'step': Default reward, a small penalty for not reaching the target yet to encourage shorter solutions. 
                Default value: -0.01
            - 'hit_wall': Reward for hitting a wall. 
                Default value: -0.1
            - 'reach_target' Reward for reaching the target square. 
                Default value: 1.0
        """
        self.maze = torch.tensor(maze, dtype=torch.float32)
        self.n_rows, self.n_cols = self.maze.shape
        self.agent_position = None
        self.target_positions = targets
        self.reset()
        self.step_reward   = rewards.get('step', -0.01)
        self.wall_reward   = rewards.get('hit_wall', -0.1)
        self.target_reward = rewards.get('reach_target', 1.0)
        
        
    def reset(self):
        """
        Resets the environment to start a new episode.
        """
        # Get free spaces that are not walls or the target
        maze_temp = copy.deepcopy(self.maze)
        for target in self.target_positions:
            maze_temp[target] = 1
        free_spaces = torch.where(maze_temp == 0)
        
        # Randomly place the agent in a free space (0)
        idx = np.random.choice(len(free_spaces[0]))
        self.agent_position = (int(free_spaces[0][idx]), int(free_spaces[1][idx]))
        return self.get_state()

    def get_state(self):
        """
        Returns the current state of the environment.
        """
        state = torch.zeros((self.n_rows, self.n_cols, 2))
        state[:, :, 0] = self.maze
        state[self.agent_position[0], self.agent_position[1], 1] = 1
        return state

    def step(self, action):
        """
        Updates the environment based on the agent's action.
        :param action: An integer representing the action taken by the agent.
        
        Note: The verticle coordinates of the actions may appear to be backwards,
        but this reflects the way they appear in the array and thus in imshow: 
        Namely, higher row-index values appear lower in the picture.
        
        Returns: (state, reward, done)
        - state: np.array with shape (*maze.shape, 2)
        - reward: float
        - done: bool (use this to stop your episode)
        """
        # Define action effects
        action_effects = {
            0: (-1, 0),  # Up
            1: (1, 0),   # Down
            2: (0, -1),  # Left
            3: (0, 1)    # Right
        }
        
        # Calculate new position
        effect = action_effects[action]
        new_position = (self.agent_position[0] + effect[0], self.agent_position[1] + effect[1])
        
        # Initialize default reward and done flag
        reward = self.step_reward  # Small negative reward for each move
        done = False

        # Check if new position is the target
        if new_position in self.target_positions:
            reward = self.target_reward  # Positive reward for reaching the target
            done = True
        # Check if new position is within bounds and not a wall
        elif 0 <= new_position[0] < self.n_rows and 0 <= new_position[1] < self.n_cols and self.maze[new_position] == 0:
            self.agent_position = new_position
        else:
            # A larger negative reward for hitting a wall
            reward = self.wall_reward

        # Return the new state, reward, and done flag
        return self.get_state(), reward, done
        
    def display(self):
        maze = self.maze.numpy().copy()
        padded_maze = np.pad(maze, 1, 'constant', constant_values=1)
        im = 255*(1-np.stack([padded_maze]*3, axis=2))
        
        # Draw blue agent pixel:
        im[tuple(x+1 for x in self.agent_position)] = [0,100,255]
        # Draw red target pixels:
        for pos in self.target_positions:
            im[pos[0]+1, pos[1]+1] = [255,0,0]
        
        plt.figure(figsize=[2,2])
        plt.imshow(im.astype(np.uint8))
        plt.axis('off'); plt.show()
        
