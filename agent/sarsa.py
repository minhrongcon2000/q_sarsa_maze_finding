import gym
import numpy as np

class SARSA:
    def __init__(self, env: gym.Env, discounted_factor: float) -> None:
        self.__discount_factor = discounted_factor
        self.__maze_size = env.observation_space.high[1] - env.observation_space.low[1] + 1
        self.__q_table = np.zeros((self.__maze_size ** 2, env.action_space.n))
        self.__optimal_policy = np.random.randint(0, env.action_space.n, size=(self.__maze_size, self.__maze_size))
        
    def __coord2state(self, coord):
        x, y = coord
        x, y = int(x), int(y)
        return x * self.__maze_size + y
    
    def __state2coord(self, state):
        y = state % self.__maze_size
        x = (state - y) // self.__maze_size
        return x, y
    
    def update_policy(self, current_coord, current_action, current_reward, next_coord, next_action, learning_rate, exploration_rate):
        current_state = self.__coord2state(current_coord)
        next_state = self.__coord2state(next_coord)
        
        self.__q_table[current_state, current_action] = self.__q_table[current_state, current_action] + learning_rate * (current_reward + self.__discount_factor * self.__q_table[next_state, next_action] - self.__q_table[current_state, current_action])
        
        if np.random.binomial(1, 1 - exploration_rate) > 0:
            self.__optimal_policy[current_coord[0], current_coord[1]] = np.argmax(self.__q_table[current_state])
        else:
            self.__optimal_policy[current_coord[0], current_coord[1]] = np.random.randint(0, self.__q_table.shape[1])
    
    @property
    def optimal_policy(self):
        return self.__optimal_policy
    
    @property
    def Q_table(self):
        return self.__q_table