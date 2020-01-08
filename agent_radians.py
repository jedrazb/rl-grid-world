import collections
from multiprocessing import Pool

import numpy as np
import torch


class Agent:

    # Function to initialise the agent
    def __init__(self):
        # Set the episode length (you will need to increase this)
        self.episode_length = 250
        # Reset the total number of steps which the agent has taken
        self.num_steps_taken = 0
        # Episodes count
        self.episodes = 0
        # The state variable stores the latest state of the agent in the environment
        self.state = None
        # Store the initial state of the agent
        self.init_state = None
        # The action variable stores the latest action which the agent has applied to the environment
        self.action = None
        self.dqn = DQN()
        # Action is expressed in radians, step size constant
        # to minimize the steps taken to reach the goal
        self.step_size = 0.02
        self.action_space_start = -0.5 * np.pi
        self.action_space_end = 0.5 * np.pi
        # Parameters for epsilon-greedy exploration
        self.epsilon = 1
        self.delta = 0.0002
        # Periodically inspect the network optimal policy
        self.evaluation_mode = False
        # When greedy policy works save it
        self.snapshot_manager = NetworkGreedyPolicySnapshotManager()
        self.optimal_policy_loaded = False

        self.min_dist_to_goal = 1.0

        # debug flag
        self.debug = True

    # Function to check whether the agent has reached the end of an episode
    def has_finished_episode(self):
        if self.num_steps_taken % self.episode_length == 0:
            self.episodes += 1
            # Make sure you finish the evaluation
            self.evaluation_mode = False
            return True
        else:
            return False

    # Function to get the next action, using whatever method you like
    def get_next_action(self, state):
        # Some debug info
        if self.debug:
            if self.num_steps_taken % 500 == 0:
                print('Steps: {}, epsilon: {}, episode length: {}'.format(
                    self.num_steps_taken,
                    self.epsilon,
                    self.episode_length
                ))

        # Periodically evaluate the policy
        if self._should_evaluate_policy():
            self.evaluation_mode = True

        # Update the number of steps which the agent has taken
        self.num_steps_taken += 1

        # Evaluate the policy
        if self.evaluation_mode:
            return self.get_greedy_action_for_evaluation(state)

        if self.num_steps_taken > 2000 and self.episodes % 10 == 0:
            self._decrease_episode_length(delta=50)

        # Store the state; this will be used later, when storing the transition
        self.state = state
        # Here, the action is random, but you can change this
        action = self.e_greedy_action()
        # Get the continuous action
        cartesian_action = self._action_to_cartesian(action)
        # Store the action; this will be used later, when storing the transition
        self.action = action
        return cartesian_action

    # Function to set the next state and distance, which resulted from applying action self.action at state self.state
    def set_next_state_and_distance(self, next_state, distance_to_goal):
        if self.min_dist_to_goal > distance_to_goal:
            self.min_dist_to_goal = distance_to_goal
            print(distance_to_goal)

        # Don't train the network when evaluating it
        if self.evaluation_mode:
            if distance_to_goal < 0.03:
                steps = self.num_steps_taken % self.episode_length
                steps_taken = steps if steps != 0 else self.episode_length
                if steps_taken < self.snapshot_manager.get_min_steps_to_goal():
                    weights = self.dqn.q_network.state_dict()
                    self.snapshot_manager.preserve_weights(
                        num_steps=steps_taken,
                        weights=weights
                    )
                    # if self.debug:
                    print('Greedy policy works! Reached goal in {} steps.'.format(
                        steps_taken
                    ))
            return

        # Convert the distance to a reward
        reward = self.calculate_reward(next_state, distance_to_goal)
        # Create a transition
        transition = (self.state, self.action, reward, next_state)

        # Add transition to the buffer
        self.dqn.replay_buffer.add(transition)
        if self.dqn.replay_buffer.is_big_enough():
            self.dqn.train_q_network()
            self.epsilon = max(self.epsilon-self.delta, 0.15)

        # Update target network every 50th step
        if self.num_steps_taken % 100 == 0:
            self.dqn.update_target_network()

    def calculate_reward(self, next_state, distance_to_goal):
        reward = 1 - distance_to_goal
        if distance_to_goal <= 0.2:
            reward *= 3
        elif distance_to_goal <= 0.3:
            reward *= 2
        elif distance_to_goal <= 0.5:
            reward *= 1.5

        if not np.any(self.state - next_state):
            reward = -0.5
        return reward

    # Function to get the greedy action for a particular state
    def get_greedy_action(self, state):
        self._load_snapshot_state()
        best_action = self.dqn.find_best_action(state, 10, 5)
        return self._action_to_cartesian(best_action)

    # Function to get the greedy action for a particular state when evaluating stuff
    def get_greedy_action_for_evaluation(self, state):
        best_action = self.dqn.find_best_action(state, 10, 5)
        return self._action_to_cartesian(best_action)

    def random_action(self):
        return np.random.uniform(
            low=self.action_space_start,
            high=self.action_space_end
        )

    def e_greedy_action(self):
        best_action = self.dqn.find_best_action(self.state)
        prob = np.random.uniform(low=0.0, high=1.0)
        if prob < self.epsilon:
            return self.random_action()
        else:
            return best_action

    def _action_to_cartesian(self, action):
        radians = action
        x = np.cos(radians)
        if x < 0:
            print('error')
        y = np.sin(radians)
        return np.array([x, y], dtype=np.float32) * self.step_size

    def _cartesian_to_action(self, cartesian):
        return np.arctan2(cartesian[1], cartesian[0])

    def _decrease_episode_length(self, delta=50):
        if self.episode_length > 100:
            self.episode_length -= delta

    def _should_evaluate_policy(self):
        return all([
            self.episode_length == 100,
            self.num_steps_taken > 2500,
            self.episodes % 5 == 0
        ])

    def _load_snapshot_state(self):
        if not self.optimal_policy_loaded and self.snapshot_manager.stores_snapshot():
            optimal_weights = self.snapshot_manager.get_optimal_weights()
            self.dqn.q_network.load_state_dict(optimal_weights)
            self.optimal_policy_loaded = True


# The Network class inherits the torch.nn.Module class, which represents a neural network.
class Network(torch.nn.Module):

    # The class initialisation function. This takes as arguments the dimension of the network's input (i.e. the dimension of the state), and the dimension of the network's output (i.e. the dimension of the action).
    def __init__(self, input_dimension, output_dimension):
        # Call the initialisation function of the parent class.
        super(Network, self).__init__()
        # Define the network layers. This example network has two hidden layers, each with 100 units.
        self.layer_1 = torch.nn.Linear(
            in_features=input_dimension, out_features=100)
        self.layer_2 = torch.nn.Linear(in_features=100, out_features=100)
        # self.layer_3 = torch.nn.Linear(in_features=100, out_features=100)
        self.output_layer = torch.nn.Linear(
            in_features=100, out_features=output_dimension)

    # Function which sends some input data through the network and returns the network's output. In this example, a ReLU activation function is used for both hidden layers, but the output layer has no activation function (it is just a linear layer).
    def forward(self, input):
        layer_1_output = torch.nn.functional.relu(self.layer_1(input))
        layer_2_output = torch.nn.functional.relu(self.layer_2(layer_1_output))
        # layer_3_output = torch.nn.functional.relu(self.layer_3(layer_2_output))
        output = self.output_layer(layer_2_output)
        return output


import time
# The DQN class determines how to train the above neural network.


class DQN:

    # The class initialisation function.
    def __init__(self):
        # Create a Q-network, which predicts the q-value for a particular state.
        self.q_network = Network(input_dimension=3, output_dimension=1)
        # Define the optimiser which is used when updating the Q-network. The learning rate determines how big each gradient step is during backpropagation.
        self.optimiser = torch.optim.Adam(
            self.q_network.parameters(), lr=0.005)
        # Replay buffer
        self.replay_buffer = ReplayBuffer()
        # Target network
        self.target_q_network = Network(input_dimension=3, output_dimension=1)
        # Discount factor
        self.discount_factor = 0.9
        # Thread pool to speed up the loss calculation
        self.pool = Pool(4)

    def __getstate__(self):
        self_dict = self.__dict__.copy()
        del self_dict['pool']
        return self_dict

    # Function that is called whenever we want to train the Q-network. Each call to this function takes in a transition tuple containing the data we use to update the Q-network.
    def train_q_network(self):
        # Set all the gradients stored in the optimiser to zero.
        self.optimiser.zero_grad()

        # batch = self.replay_buffer.last_entry()
        batch = self.replay_buffer.random_sample()

        # Calculate the loss for this transition.
        loss = self._calculate_long_run_loss(batch)

        # Compute the gradients based on this loss, i.e. the gradients of the loss with respect to the Q-network parameters.
        loss.backward()
        # Take one gradient step to update the Q-network.
        self.optimiser.step()
        # Return the loss as a scalar
        return loss.item()

    def _calculate_long_run_loss(self, batch):
        s_a, r, s_p = batch
        prediction_tensor = self.q_network.forward(torch.tensor(s_a))

        # actions = []
        # for s in s_p:
        #     actions.append(self.find_best_action(s))

        actions = self.pool.map(self.find_best_action, s_p)
        actions = np.array(actions).reshape(-1, 1)
        s_p_a = np.hstack((s_p, actions)).astype(np.float32)

        # bellman equation
        state_prime_tensor = self.target_q_network.forward(
            torch.tensor(s_p_a)).detach()

        expected_value = r + self.discount_factor * state_prime_tensor.data.numpy()
        return torch.nn.MSELoss()(torch.tensor(expected_value), prediction_tensor)

    def update_target_network(self):
        weights = self.q_network.state_dict()
        self.target_q_network.load_state_dict(weights)

    # Estimate the best action for this state
    def find_best_action(self, state, n=5, k=3):
        low = -0.5 * np.pi
        high = 0.5 * np.pi
        sample = np.linspace(
            start=-0.5 * np.pi,
            stop=0.5 * np.pi,
            num=n
        )
        for _ in range(3):
            s_a = np.hstack(
                (np.repeat(state.reshape(1, -1), n, axis=0).reshape(n, 2),
                 sample.reshape(-1, 1))
            ).astype(np.float32)
            vals = self.q_network.forward(
                torch.tensor(s_a)
            ).detach().numpy().squeeze()
            best_idx = vals.argsort()[-k:][::-1]
            actions = sample[best_idx]
            mean, std = np.mean(actions), np.std(actions)
            sample = np.random.normal(
                loc=mean,
                scale=std,
                size=n
            )
        return max(min(mean, high), low)


class ReplayBuffer:

    def __init__(self):
        self.buffer = collections.deque(maxlen=20000)
        self.sample_size = 20

    def size(self):
        return len(self.buffer)

    def is_big_enough(self):
        return self.size() >= self.sample_size

    def add(self, transition):
        self.buffer.appendleft(transition)

    def random_sample(self):
        buffer_size = self.size()
        sample_idx = np.random.choice(
            np.arange(buffer_size), size=self.sample_size, replace=False)

        state_actions = []
        rewards = []
        states_prime = []

        for idx in sample_idx:
            s, a, r, s_p = self.buffer[idx]
            s_a = np.array([s[0], s[1], a], dtype=np.float32)
            state_actions.append(s_a)
            rewards.append(r)
            states_prime.append(s_p)

        state_actions = np.array(state_actions, dtype=np.float32)
        rewards = np.array(rewards, dtype=np.float64).reshape(-1, 1)
        states_prime = np.array(states_prime, dtype=np.float32)

        return state_actions, rewards, states_prime


# Takes care to save the weights when network reaches goal with minimal amount of steps
class NetworkGreedyPolicySnapshotManager:
    def __init__(self):
        self.min_steps_to_goal = 100
        self.weights = None
        self.has_snapshot = False

    def preserve_weights(self, num_steps, weights):
        if num_steps < self.min_steps_to_goal:
            self.min_steps_to_goal = num_steps
            self.weights = weights

    def get_optimal_weights(self):
        return self.weights

    def get_min_steps_to_goal(self):
        return self.min_steps_to_goal

    def stores_snapshot(self):
        return self.has_snapshot
