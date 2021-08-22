# q_sarsa_maze_finding
A Python implementation of Q-Learning and SARSA algorithm on finding optimal path through maze.

## Experimentation setup
This is a naive implementation of two algorithm, Q-Learning and SARSA, on finding optimal path through a maze environment. The environment in use is credited to this repository [here](https://github.com/MattChanTK/gym-maze). In this case, the pre-generated environment _MazeEnvSample10x10_ is used.

For SARSA and Q-Learning, the algorithm is setup to become GLIE and step-size (or learning rate, if you familiar with this term) is setup to satisfy the Robbins-Munro condition; hence, both are decayed by a function 1/T, where T is the number of episodes.

## Dependencies:
- [OpenAI Gym](https://github.com/openai/gym)
- [gym_maze](https://github.com/MattChanTK/gym-maze)
- [numpy](https://github.com/numpy/numpy)

## Results
For Q-Learning: https://youtu.be/KPwNsHukWtc
For SARSA: https://youtu.be/oruyR6-R0OM
