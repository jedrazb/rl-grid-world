import numpy as np
import cv2


# The Environment class defines the "world" within which the agent is acting
class Environment:

    # Function to initialise an Environment object
    def __init__(self,  magnification):
        # Set the magnification factor of the display
        self.magnification = magnification
        # Set the width and height of the environment
        self.width = 1.0
        self.height = 1.0
        # Create an image which will be used to display the environment
        self.image = np.zeros([int(self.magnification * self.height),
                               int(self.magnification * self.width), 3], dtype=np.uint8)
        # Define the space of the environment
        self.init_state = None
        self.free_blocks = None
        self.goal_state = None
        self._define_environment_space()

    # Define environment space
    def _define_environment_space(self):
        # Set the initial state of the agent
        init_state_x = 0.05
        init_state_y = np.random.uniform(0.05, 0.95)
        self.init_state = np.array(
            [init_state_x, init_state_y], dtype=np.float32)
        # Create an empty list of free blocks
        self.free_blocks = []
        # Create the first free space block
        block_bottom = init_state_y - np.random.uniform(0.1, 0.2)
        block_top = init_state_y + np.random.uniform(0.1, 0.2)
        block_left = 0.02
        block_right = block_left + np.random.uniform(0.03, 0.1)
        top_left = (block_left, block_top)
        bottom_right = (block_right, block_bottom)
        block = (top_left, bottom_right)
        self.free_blocks.append(block)
        prev_top = top_left[1]
        prev_bottom = bottom_right[1]
        prev_right = bottom_right[0]
        # Whilst the latest free space block has not reached 0.2 from the right-hand edge of the environment, continue adding free space blocks
        while prev_right < 0.8:
            is_within_boundary = False
            while not is_within_boundary:
                block_height = np.random.uniform(0.05, 0.4)
                block_bottom_max = prev_top - 0.05
                block_bottom_min = prev_bottom - (block_height - 0.05)
                block_bottom_mid = 0.5 * (block_bottom_min + block_bottom_max)
                block_bottom_half_range = block_bottom_max - block_bottom_mid
                r1 = np.random.uniform(-block_bottom_half_range,
                                       block_bottom_half_range)
                r2 = np.random.uniform(-block_bottom_half_range,
                                       block_bottom_half_range)
                if np.fabs(r1) > np.fabs(r2):
                    block_bottom = block_bottom_mid + r1
                else:
                    block_bottom = block_bottom_mid + r2
                block_top = block_bottom + block_height
                block_left = prev_right
                block_width = np.random.uniform(0.03, 0.1)
                block_right = block_left + block_width
                top_left = (block_left, block_top)
                bottom_right = (block_right, block_bottom)
                if block_bottom < 0 or block_top > 1 or block_left < 0 or block_right > 1:
                    is_within_boundary = False
                else:
                    is_within_boundary = True
            block = (top_left, bottom_right)
            self.free_blocks.append(block)
            prev_top = block_top
            prev_bottom = block_bottom
            prev_right = block_right
        # Add the final free space block
        block_height = np.random.uniform(0.05, 0.15)
        block_bottom_max = prev_top - 0.05
        block_bottom_min = prev_bottom - (block_height - 0.05)
        block_bottom = np.random.uniform(block_bottom_min, block_bottom_max)
        block_top = block_bottom + block_height
        block_left = prev_right
        block_right = 0.98
        top_left = (block_left, block_top)
        bottom_right = (block_right, block_bottom)
        block = (top_left, bottom_right)
        self.free_blocks.append(block)
        # Set the goal state
        self.goal_state = np.array([0.95, np.random.uniform(
            block_bottom + 0.01, block_top - 0.01)], dtype=np.float32)

    # Function to reset the environment, which is done at the start of each episode
    def reset(self):
        return self.init_state

    # Function to execute an agent's step within this environment, returning the next state and the distance to the goal
    def step(self, state, action):
        # If the action is greater than the maximum action, then the agent stays still
        if np.linalg.norm(action) > 0.02:
            next_state = state
        else:
            # Determine what the new state would be if the agent could move there
            next_state = state + action
            # If this state is outside the environment's perimeters, then the agent stays still
            if next_state[0] < 0.0 or next_state[0] > 1.0 or next_state[1] < 0.0 or next_state[1] > 1.0:
                next_state = state
            # If this state is inside the walls, then the agent stays still
            is_agent_in_free_space = False
            for block in self.free_blocks:
                if block[0][0] < next_state[0] < block[1][0] and block[1][1] < next_state[1] < block[0][1]:
                    is_agent_in_free_space = True
                    break
            if not is_agent_in_free_space:
                next_state = state
        # Compute the distance to the goal
        distance_to_goal = np.linalg.norm(next_state - self.goal_state)
        # Return the next state and the distance to the goal
        return next_state, distance_to_goal

    # Function to draw the environment and display it on the screen, if required
    def show(self, agent_state):
        # Create a grey image, representing the environment walls
        self.image.fill(100)
        # Draw all the free blocks, representing the free space
        for block in self.free_blocks:
            top_left = (int(self.magnification *
                            block[0][0]), int(self.magnification * (1 - block[0][1])))
            bottom_right = (int(
                self.magnification * block[1][0]), int(self.magnification * (1 - block[1][1])))
            cv2.rectangle(self.image, top_left, bottom_right,
                          (0, 0, 0), thickness=cv2.FILLED)
        # Draw the agent
        agent_centre = (int(agent_state[0] * self.magnification),
                        int((1 - agent_state[1]) * self.magnification))
        agent_radius = int(0.01 * self.magnification)
        agent_colour = (0, 0, 255)
        cv2.circle(self.image, agent_centre,
                   agent_radius, agent_colour, cv2.FILLED)
        # Draw the goal
        goal_centre = (int(self.goal_state[0] * self.magnification),
                       int((1 - self.goal_state[1]) * self.magnification))
        goal_radius = int(0.01 * self.magnification)
        goal_colour = (0, 255, 0)
        cv2.circle(self.image, goal_centre,
                   goal_radius, goal_colour, cv2.FILLED)
        # Show the image
        cv2.imshow("Environment", self.image)
        # This line is necessary to give time for the image to be rendered on the screen
        cv2.waitKey(1)
