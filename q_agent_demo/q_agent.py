import numpy as np

class Q_Agent:
    
    def __init__(self, env_shape, n_flags, actions, 
                 learning_rate=0.1, discount_factor=0.95, 
                 epsilon=0.9, epsilon_decay=0.999, epsilon_min=0.1):
        
        self.q_values = np.zeros(shape=(*env_shape, *([2]*n_flags), len(actions)))
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.actions = actions

    def choose_action(self, state):
        if np.random.random() < self.epsilon:
            return np.random.choice(self.actions)
        else:
            action_index = np.argmax(self.q_values[state])
            return self.actions[action_index]

    def update_q_values(self, state, action, reward, next_state):
        old_q_value = self.q_values[state + (self.actions.index(action),)]
        future_reward = np.max(self.q_values[next_state])
        self.q_values[state + (self.actions.index(action),)] = old_q_value + self.learning_rate * (reward + self.discount_factor * future_reward - old_q_value)

    def update_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        else:
            self.epsilon = self.epsilon_min
            