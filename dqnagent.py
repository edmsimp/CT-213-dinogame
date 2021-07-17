import random
import numpy as np
from collections import deque
from tensorflow.keras import models, layers, optimizers, activations, losses
from rl.agents import DQNAgent
from rl.memory import SequentialMemory
from rl.policy import LinearAnnealedPolicy, EpsGreedyQPolicy


class Agent:
    """
    Represents a Deep Q-Networks (DQN) agent.
    """
    def __init__(self, state_size, action_size):
        """
        Creates a Deep Q-Networks (DQN) agent.

        :param state_size: number of dimensions of the feature vector of the state.
        :type state_size: int.
        :param action_size: number of actions.
        :type action_size: int.
        :param gamma: discount factor.
        :type gamma: float.
        :param epsilon: epsilon used in epsilon-greedy policy.
        :type epsilon: float.
        :param epsilon_min: minimum epsilon used in epsilon-greedy policy.
        :type epsilon_min: float.
        :param epsilon_decay: decay of epsilon per episode.
        :type epsilon_decay: float.
        :param learning_rate: learning rate of the action-value neural network.
        :type learning_rate: float.
        :param buffer_size: size of the experience replay buffer.
        :type buffer_size: int.
        """
        self.state_size = state_size
        self.height, self.width, self.channels = self.state_size    
        self.action_size = action_size
        self.model = self.make_model(self.height, self.width, self.channels, self.action_size)

    def make_model(self, height, width, channels, actions):
        """
        Makes the action-value neural network model using Keras.

        :return: action-value neural network.
        :rtype: Keras' model.
        """
        # Todo: Uncomment the lines below CHECK
        model = models.Sequential()
        # Todo: implement Keras' model CHECK      
        model.add(layers.Convolution2D(32, (8,8), strides=(4,4), activation=activations.relu, input_shape=(3, height, width, channels), padding="same"))
        model.add(layers.Convolution2D(64, (4,4), strides=(2,2), activation=activations.relu, padding="same"))
        model.add(layers.Convolution2D(64, (3,3), strides=(1,1), activation=activations.relu, padding="same"))
        model.add(layers.Flatten())
        model.add(layers.Dense(512, activation='relu'))
        model.add(layers.Dense(256, activation='relu'))
        model.add(layers.Dense(actions, activation='linear'))        
        model.summary()
        return model

    def build_agent(self, model, actions):
        policy = LinearAnnealedPolicy(EpsGreedyQPolicy(), attr='eps', value_max=1., value_min=.1, value_test=.2, nb_steps=10000)
        memory = SequentialMemory(limit=1000, window_length=3)
        dqn = DQNAgent(model=model, memory=memory, policy=policy,
                  enable_dueling_network=True, dueling_type='avg', 
                   nb_actions=actions, nb_steps_warmup=1000
                  )
        return dqn

    def act(self, state):
        """
        Chooses an action using an epsilon-greedy policy.

        :param state: current state.
        :type state: NumPy array with dimension (1, 2).
        :return: chosen action.
        :rtype: int.
        """
        # Todo: implement epsilon-greey action selection. CHECK

        action_values = self.model.predict(state)
        greedy_action = np.argmax(action_values)
        m = np.random.rand()
        if m >= self.epsilon:
            return greedy_action
        else:
            num_actions = action_values.shape[1]
            random_action = np.random.randint(num_actions)
            return random_action
