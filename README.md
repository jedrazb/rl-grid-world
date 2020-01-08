# Deep Q-learning agent in the grid world
Deep Q-learning agent which finds a path to the goal in a grid world. This exercise was done as a coursework for course C424 at Imperial College London.

<p align="center"><img src="https://media.giphy.com/media/fAo4EH4rFJO7xPjKyS/giphy.gif" width="400"></p>

--------
Dependencies: `numpy`, `cv2`, `torch`

To start the training run:
```
python train_and_test.py
```

-------

Following techniques were used:
* Deep Q-network (created in PyTorch)
* Models with continous (`agent_radians.py`) and discrete actions (`agent.py`)
* Prioritised experience replay buffer
* Epsilon greedy policy
* Target network
* Sampling using Cross Entropy Method

