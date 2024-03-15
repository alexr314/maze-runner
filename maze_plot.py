import torch
import numpy as np
import matplotlib.pyplot as plt

def get_policy_table(maze, policy_net):
    """Evaluates the policy at every valid state in the maze.
    Returns a (*maze.shape, 4) array of probabilities of each action
    Use it in conjunction with plot_policy if the policy is represented by a neural net.
    """
    # Handling if maze is np array
    if type(maze) != torch.Tensor:
        maze = torch.tensor(maze)

    policy_table = np.zeros((*maze.shape, 4))

    # For each possible agent position
    for agent_position in np.stack(np.unravel_index(range(np.prod(maze.shape)), maze.shape)).T:

        # Don't plot if in walls:
        if maze[tuple(agent_position)]:
            continue

        # compute state
        state = torch.zeros((*maze.shape, 2))
        state[:, :, 0] = maze
        state[agent_position[0], agent_position[1], 1] = 1
        state_tensor = torch.flatten(state).float().unsqueeze(0)  # Add batch dimension

        # Pass the state through the policy network to get action probabilities
        with torch.no_grad():  # Disable gradient calculation
            action_probs = policy_net(state_tensor)

        policy_table[tuple(agent_position)] = action_probs
    
    return policy_table


def draw_policy(maze, policy_table, targets=[]):
    """Given a maze and a policy_table array of shape (*maze, 4)
    This visually represents the probability of each action at each state 
    by drawning arrows, whose opacity corresponds to the respective probability.
    Also pass targets list: [[x1,y1], [x2,y2], ...]
    
    If the policy is a neural network, get the policy_table by using get_policy_table
    Example usage:
    >>> policy_table = get_policy_table(maze, policy_net)
    >>> draw_policy(maze, policy_table)
    """
    # Draw the maze
    padded_maze = np.pad(maze, 1, 'constant', constant_values=1)
    
    im = 255*(1-np.stack([padded_maze]*3, axis=2))
    
    # Draw red target pixels:
    for target in targets:
        im[tuple(x+1 for x in target)] = [255,180,180]
    
    plt.imshow(im)
    
#     # Handling if maze is np array
#     if type(maze) != torch.Tensor:
#         maze = torch.tensor(maze)   # I don't think this matters now...

    # For each possible agent position
    all_positions = np.stack(np.unravel_index(range(np.prod(maze.shape)), maze.shape)).T
    for agent_position in all_positions:
        
        # Skip plotting if in walls:
        if maze[tuple(agent_position)]:
            continue
            
        action_probs = policy_table[tuple(agent_position)]

        # Draw arrows:
        color = 'r'
        
        # Up
        plt.arrow(*reversed(agent_position+1), 0, -.12, head_width=.25, lw=3, 
                  ec='none', fc=color, alpha=float(action_probs[0]))
        # Down
        plt.arrow(*reversed(agent_position+1), 0, .12, head_width=.25, lw=3, 
                  ec='none', fc=color, alpha=float(action_probs[1]))
        # Left
        plt.arrow(*reversed(agent_position+1), -.12, 0, head_width=.25, lw=3, 
                  ec='none', fc=color, alpha=float(action_probs[2]))
        # Right
        plt.arrow(*reversed(agent_position+1), .12, 0, head_width=.25, lw=3, 
                  ec='none', fc=color, alpha=float(action_probs[3]))

    plt.axis('off')