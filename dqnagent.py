import random
import numpy as np
from collections import deque
from tensorflow.keras import models, layers, optimizers, activations, losses, Model
import tensorflow as tf
from tensorflow.keras.layers import Conv2D,Flatten,Lambda,Dense,concatenate,Add




class Agent:
    """
    Represents a Deep Q-Networks (DQN) agent.
    """
    def __init__(self, state_size, action_size, gamma=0.99, epsilon=0.1, epsilon_min=0.0001, epsilon_decay=0.99, learning_rate=1e-4, buffer_size=50000):
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
        self.replay_buffer = deque(maxlen=buffer_size)  # giving a maximum length makes this buffer forget old memories
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.learning_rate = learning_rate
        self.model = self.make_model()
        

    def make_model(self):
        """
        Makes the action-value neural network model using Keras.
        :return: action-value neural network.
        :rtype: Keras' model.
        """

        """
        DQN - SIMPLE MODEL NEURAL NETWORK - UN/COMMENT TO ENABLE/DISABLE DQN METHOD
        """ 
        # ----------------------------------------------------------------------------------------------------------
        # model = models.Sequential()
        # model.add(layers.Convolution2D(32, (8,8), strides=(4,4), activation=activations.relu, 
        #     input_shape=(self.height, self.width, self.channels), padding="same"))
        # model.add(layers.Convolution2D(64, (4,4), strides=(2,2), activation=activations.relu, padding="same"))
        # model.add(layers.Convolution2D(64, (3,3), strides=(1,1), activation=activations.relu, padding="same"))
        # model.add(layers.Flatten())
        # model.add(layers.Dense(512, activation='relu'))
        # model.add(layers.Dense(256, activation='relu'))
        # model.add(layers.Dense(self.action_size, activation='linear'))
        # model.compile(loss=losses.mse, optimizer=optimizers.Adam(lr=self.learning_rate))  
        # model.summary()
        # ----------------------------------------------------------------------------------------------------------

        """
        DQN DUELING - DQN DUELING MODEL NEURAL NETWORK UN/COMMENT TO ENABLE/DISABLE DQN METHOD
        """      
        # ----------------------------------------------------------------------------------------------------------
        ## Create Neural Network Convolucional
        input_image = layers.Input(shape=(self.height, self.width, self.channels))

        nnConv = Conv2D(32, (8,8), strides=(4,4), activation='relu')(input_image)
        nnConv = Conv2D(64, (4,4), strides=(2,2), activation='relu')(nnConv)
        nnConv = Conv2D(64, (3,3), strides=(1,1), activation='relu')(nnConv)
        nnConv = Flatten()(nnConv)
        nnConv = Dense(512,activation='relu')(nnConv)
        nnConv = Dense(256,activation='relu')(nnConv)
        ## Create Layer for state value
        StateValue = Dense(1)(nnConv)
        StateValue = Lambda(lambda s: tf.expand_dims(s[:, 0], -1),
                       output_shape=(self.action_size,))(StateValue)

        ## Create Layer for Advantage
        Advantage = Dense(self.action_size)(nnConv)
        Advantage = Lambda(lambda a: a[:, :] - tf.math.reduce_mean(a[:, :], axis=1,keepdims=True),
                           output_shape=(self.action_size,))(Advantage)

        ## Concatenate the two layers into one
        qValue = Add()([StateValue, Advantage])
        #qValue = concatenate([StateValue,Advantage], name='sum')

        model = Model(inputs=input_image, outputs=qValue)
        model.compile(loss=losses.mse, optimizer=optimizers.Adam(lr=self.learning_rate)) 
        # ----------------------------------------------------------------------------------------------------------
        return model

    def act(self, state):
        """
        Chooses an action using an epsilon-greedy policy.
        :param state: current state.
        :type state: NumPy array with dimension (1, 2).
        :return: chosen action.
        :rtype: int.
        """
        
        action_values = self.model.predict(state)
        greedy_action = np.argmax(action_values)
        m = np.random.rand()
        if m >= self.epsilon:
            return greedy_action
        else:
            num_actions = action_values.shape[1]
            random_action = np.random.randint(num_actions)
            return random_action

    def append_experience(self, state, action, reward, next_state, done):
        """
        Appends a new experience to the replay buffer (and forget an old one if the buffer is full).
        :param state: state.
        :type state: NumPy array with dimension (1, 2).
        :param action: action.
        :type action: int.
        :param reward: reward.
        :type reward: float.
        :param next_state: next state.
        :type next_state: NumPy array with dimension (1, 2).
        :param done: if the simulation is over after this experience.
        :type done: bool.
        """
        self.replay_buffer.append((state, action, reward, next_state, done))

    def replay(self, batch_size):
        """
        Learns from memorized experience.
        :param batch_size: size of the minibatch taken from the replay buffer.
        :type batch_size: int.
        :return: loss computed during the neural network training.
        :rtype: float.
        """
        minibatch = random.sample(self.replay_buffer, batch_size)
        states, targets = [], []
        for state, action, reward, next_state, done in minibatch:
            target = self.model.predict(state)
            if not done:
                target[0][action] = reward + self.gamma * np.max(self.model.predict(next_state)[0])
            else:
                target[0][action] = reward
            # Filtering out states and targets for training
            states.append(state[0])
            targets.append(target[0])
        history = self.model.fit(np.array(states), np.array(targets), epochs=1, verbose=0)
        # Keeping track of loss
        loss = history.history['loss'][0]
        return loss

    def load(self, name):
        """
        Loads the neural network's weights from disk.
        :param name: model's name.
        :type name: str.
        """
        
        self.model.load_weights(name)

    def save(self, name):
        """
        Saves the neural network's weights to disk.
        :param name: model's name.
        :type name: str.
        """
        self.model.save_weights(name)

    def update_epsilon(self):
        """
        Updates the epsilon used for epsilon-greedy action selection.
        """
        self.epsilon *= self.epsilon_decay
        if self.epsilon < self.epsilon_min:
            self.epsilon = self.epsilon_min