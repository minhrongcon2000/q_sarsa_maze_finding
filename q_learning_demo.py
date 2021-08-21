import gym
import gym_maze
from agent.q_learning import QLearning

env = gym.make("maze-sample-10x10-v0")
env = gym.wrappers.Monitor(env, "q_learning", force=True)
agent = QLearning(env, discount_factor=0.9)
episodes = 10000
horizon = 10000
epsilon = 0.8
alpha = 1

for episode in range(episodes):
    observation = env.reset()
    for t in range(horizon):
        env.render()
        x, y = observation
        x, y = int(x), int(y)
        
        action = agent.optimal_policy[x, y]
        action = int(action)
        observation, reward, done, _ = env.step(action=action)
        x_next, y_next = observation
        x_next, y_next = int(x_next), int(y_next)
        
        if done:
            break
        
        agent.update_policy((x, y), action, reward, (x_next, y_next), 
                            learning_rate=alpha, exploration_rate=epsilon)
    else:
        env.stats_recorder.save_complete()
        env.stats_recorder.done = True
    
    print("Episode {} completed".format(episode + 1))
    print(agent.Q_table)
    epsilon = 1 / (episode + 1)
    alpha = 1 / (episode + 1)
env.close()
