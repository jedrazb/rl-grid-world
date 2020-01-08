############################################################################
############################################################################
# THIS IS THE ONLY FILE YOU SHOULD EDIT
#
#
# Agent must always have these five functions:
#     __init__(self)
#     has_finished_episode(self)
#     get_next_action(self, state)
#     set_next_state_and_distance(self, next_state, distance_to_goal)
#     get_greedy_action(self, state)
#
#
# You may add any other functions as you wish
############################################################################
############################################################################

import collections

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
        self.action_space = np.array([0, 1, 2, 3])
        self.step_size = 0.02
        self.continuous_actions = self.step_size * np.array([
            [0, 1],
            [1, 0],
            [0, -1],
            [-1, 0]
        ], dtype=np.float32)
        self.epsilon = 1
        self.delta = 0.000015
        # Periodically inspect the network optimal policy
        self.evaluation_mode = False
        # When greedy policy works save it
        self.snapshot_manager = NetworkGreedyPolicySnapshotManager()
        self.optimal_policy_loaded = False

        # debug flag
        self.debug = False
        self.debug_optimal_policy = True

    # Function to check whether the agent has reached the end of an episode
    def has_finished_episode(self):
        if self.num_steps_taken % self.episode_length == 0:
            self.episodes += 1
            # Make sure you finish the evaluation
            self.evaluation_mode = False
            # Make sure that the model will be training
            self.dqn.q_network.train()
            return True
        else:
            return False

    # Function to get the next action, using whatever method you like
    def get_next_action(self, state):
        # Some debug info
        if self.debug:
            if self.num_steps_taken % 1000 == 0:
                print('Steps: {}, epsilon: {}, episode length: {}'.format(
                    self.num_steps_taken,
                    self.epsilon,
                    self.episode_length
                ))

        # Periodically evaluate the policy
        if self._should_evaluate_policy():
            self.evaluation_mode = True
            self.dqn.q_network.eval()

        # Update the number of steps which the agent has taken
        self.num_steps_taken += 1

        # Evaluate the policy
        if self.evaluation_mode:
            return self.get_greedy_action_for_evaluation(state)

        # Decrease the episode length so that it converges to 100
        if self.num_steps_taken > 10000 and self.episodes % 10 == 0 and self.num_steps_taken % self.episode_length == 1:
            self.decrease_episode_length(delta=25)

        # Store the state; this will be used later, when storing the transition
        self.state = state

        # Make action
        action = self.e_greedy_action()

        # Get the continuous action
        continuous_action = self._discrete_action_to_continuous(action)
        # Store the action; this will be used later, when storing the transition
        self.action = action
        return continuous_action

    # Function to set the next state and distance, which resulted from applying action self.action at state self.state
    def set_next_state_and_distance(self, next_state, distance_to_goal):
        # Don't train the network when evaluating it
        if self.evaluation_mode:
            # If there is no solution reaching the goal, preserve the one which minimises
            # the end distance to goal
            if self.snapshot_manager.min_steps_to_goal == 1000:
                self.snapshot_manager.preserve_weights_minimising_distance_if_didnt_reach_goal(
                    distance=distance_to_goal,
                    weights=self.dqn.q_network.state_dict()
                )

            # otherwise consider only the solutions which reach the goal
            if distance_to_goal < 0.03:
                steps = self.num_steps_taken % self.episode_length
                steps_taken = steps if steps != 0 else self.episode_length
                steps_taken -= 1
                if steps_taken < self.snapshot_manager.get_min_steps_to_goal():
                    weights = self.dqn.q_network.state_dict()
                    self.snapshot_manager.preserve_weights(
                        num_steps=steps_taken,
                        weights=weights
                    )
                    if self.debug_optimal_policy:
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
        if self.num_steps_taken % 50 == 0:
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
            reward /= 1.5
        return reward

    # Function to get the greedy action for a particular state
    def get_greedy_action(self, state):
        self._load_snapshot_state()

        action_rewards = self.dqn.q_network.forward(
            torch.tensor(state)
        ).detach().numpy()
        discrete_action = np.argmax(action_rewards)
        return self._discrete_action_to_continuous(discrete_action)

    # Function to get the greedy action for a particular state when evaluating stuff
    def get_greedy_action_for_evaluation(self, state):
        action_rewards = self.dqn.q_network.forward(
            torch.tensor(state)
        ).detach().numpy()
        discrete_action = np.argmax(action_rewards)
        return self._discrete_action_to_continuous(discrete_action)

    def random_action(self):
        return self.action_space[np.random.randint(low=0, high=3)]

    def e_greedy_action(self):
        action_rewards = self.dqn.q_network.forward(
            torch.tensor(self.state)
        ).detach().numpy()
        prob = np.random.uniform(low=0.0, high=1.0)
        if prob < self.epsilon:
            return self.random_action()
        else:
            return np.argmax(action_rewards)

    def _discrete_action_to_continuous(self, discrete_action):
        return self.continuous_actions[discrete_action]

    def decrease_episode_length(self, delta=50):
        if self.episode_length > 100:
            self.episode_length -= delta

    def _should_evaluate_policy(self):
        return all([
            self.episode_length == 100,
            self.num_steps_taken > 15000,
            self.episodes % 10 == 0,
            self.num_steps_taken % self.episode_length == 0
        ])

    def _load_snapshot_state(self):
        if not self.optimal_policy_loaded and self.snapshot_manager.stores_snapshot():
            optimal_weights = self.snapshot_manager.get_optimal_weights()
            self.dqn.q_network.load_state_dict(optimal_weights)
            self.optimal_policy_loaded = True
            self.dqn.q_network.eval()


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
        self.output_layer = torch.nn.Linear(
            in_features=100, out_features=output_dimension)

    # Function which sends some input data through the network and returns the network's output. In this example, a ReLU activation function is used for both hidden layers, but the output layer has no activation function (it is just a linear layer).
    def forward(self, input):
        layer_1_output = torch.nn.functional.relu(self.layer_1(input))
        layer_2_output = torch.nn.functional.relu(self.layer_2(layer_1_output))
        output = self.output_layer(layer_2_output)
        return output


# The DQN class determines how to train the above neural network.
class DQN:

    # The class initialisation function.
    def __init__(self):
        # Create a Q-network, which predicts the q-value for a particular state.
        self.q_network = Network(input_dimension=2, output_dimension=4)
        # Define the optimiser which is used when updating the Q-network. The learning rate determines how big each gradient step is during backpropagation.
        self.optimiser = torch.optim.Adam(
            self.q_network.parameters(), lr=0.005)
        # Replay buffer
        self.replay_buffer = ReplayBuffer()
        # Target network
        self.target_q_network = Network(input_dimension=2, output_dimension=4)
        # Discount factor
        self.discount_factor = 0.9

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
        s, a, r, s_p, idx = batch
        predicted_rewards = self.q_network.forward(torch.tensor(s))
        prediction_tensor = torch.gather(predicted_rewards, 1, torch.tensor(a))

        # bellman equation
        predicted_rewards_prime = self.target_q_network.forward(
            torch.tensor(s_p)).detach()
        max_actions = np.argmax(
            predicted_rewards_prime.detach().numpy(), axis=1).reshape(-1, 1)
        state_prime_tensor = torch.gather(
            predicted_rewards_prime, 1, torch.tensor(max_actions)).detach()

        expected_value = r + self.discount_factor * state_prime_tensor.data.numpy()
        idx_to_update = idx[(expected_value > np.mean(
            expected_value, axis=0) + np.std(expected_value, axis=0)).squeeze()]

        self.replay_buffer.update_weights(idx_to_update)

        return torch.nn.MSELoss()(torch.tensor(expected_value), prediction_tensor)

    def update_target_network(self):
        weights = self.q_network.state_dict()
        self.target_q_network.load_state_dict(weights)


class ReplayBuffer:

    def __init__(self):
        self.buffer = collections.deque(maxlen=10000)
        self.sample_size = 200
        self.p = collections.deque(maxlen=10000)
        self.min_p = 0.05

    def size(self):
        return len(self.buffer)

    def is_big_enough(self):
        return self.size() >= self.sample_size

    def add(self, transition):
        self.buffer.appendleft(transition)
        self.p.appendleft(self.min_p)

    def update_weights(self, idx):
        for i in idx:
            self.p[i] = self.min_p * 2

    def random_sample(self):
        buffer_size = self.size()
        prob = np.array(self.p)
        prob = prob / np.sum(prob)
        sample_idx = np.random.choice(
            np.arange(buffer_size), size=self.sample_size, replace=False, p=prob)

        states = []
        actions = []
        rewards = []
        states_prime = []

        for idx in sample_idx:
            s, a, r, s_p = self.buffer[idx]
            states.append(s)
            actions.append(a)
            rewards.append(r)
            states_prime.append(s_p)

        states = np.array(states, dtype=np.float32)
        rewards = np.array(rewards, dtype=np.float64).reshape(-1, 1)
        actions = np.array(actions, dtype=np.int64).reshape(-1, 1)
        states_prime = np.array(states_prime, dtype=np.float32)

        return states, actions, rewards, states_prime, sample_idx


# Takes care to save the weights when network reaches goal with minimal amount of steps
class NetworkGreedyPolicySnapshotManager:
    def __init__(self):
        self.min_steps_to_goal = 1000  # magic number, assume 100 or less steps in testing
        self.min_distance_to_goal = 1
        self.weights = None

    def preserve_weights(self, num_steps, weights):
        self.min_steps_to_goal = num_steps
        self.weights = weights

    def preserve_weights_minimising_distance_if_didnt_reach_goal(self, distance, weights):
        if distance < self.min_distance_to_goal and self.min_steps_to_goal == 1000:
            self.min_distance_to_goal = distance
            self.weights = weights

    def get_optimal_weights(self):
        return self.weights

    def get_min_steps_to_goal(self):
        return self.min_steps_to_goal

    def stores_snapshot(self):
        return self.weights is not None
