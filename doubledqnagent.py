import random
import numpy as np
from tensorflow.keras import models, layers, optimizers, activations, losses, Model
import tensorflow as tf
from tensorflow.keras.layers import Conv2D,Flatten,Lambda,Dense,concatenate,Add

class ReplayBuffer(object):
    """
    Represents a Replay Buffer.
    """
    def __init__(self, max_size, input_shape, n_actions, discrete=False):
        """
        Creates a Replay Buffer.

        :param max_size: max number of memories that can be remembered.
        :type max_size: int.
        :param input_shape: state size.
        :type input_shape: tuple.
        :param n_actions: number of actions.
        :type n_actions: int.
        :param discrete: tells if the model is discrete or not.
        :type epsilon: boolean.
        """
        self.mem_size = max_size
        self.mem_cntr = 0
        self.discrete = discrete
        self.state_memory = np.zeros((self.mem_size, input_shape[0], input_shape[1], input_shape[2]))
        self.next_state_memory = np.zeros((self.mem_size, input_shape[0], input_shape[1], input_shape[2]))
        dtype = np.int8 if self.discrete else np.float32
        self.action_memory = np.zeros((self.mem_size, n_actions), dtype=dtype)
        self.reward_memory = np.zeros(self.mem_size)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.float32)

    def store_transition(self, state, action, reward, next_state, done):
        """
        Stores a memory.

        :param state: state to be stored.
        :type state: 4 dimensional NumPy array.
        :param action: action to be stored.
        :type action: int.
        :param reward: reward to be stored.
        :type reward: int.
        :param next_state: next state to be stored.
        :type next_state: 4 dimensional NumPy array.
        :param done: done to be stored.
        :type done: int(0 or 1).
        """
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.next_state_memory[index] = next_state
        if self.discrete:
            actions = np.zeros(self.action_memory.shape[1])
            actions[action] = 1.0
            self.action_memory[index] = actions
        else:
            self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.terminal_memory[index] = 1 - int(done)
        self.mem_cntr = self.mem_cntr + 1

    def sample_buffer(self, batch_size):
        """
        Selects memories randomly.

        :param batch_size: number of memories to be selected.
        :type batch-size: int.
        """
        max_mem = min(self.mem_cntr, self.mem_size)
        batch = np.random.choice(max_mem, batch_size)
        states = self.state_memory[batch]
        next_states = self.next_state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        terminal = self.terminal_memory[batch]

        return states, actions, rewards, next_states, terminal

class DoubleAgent:
    """
    Represents a Double Deep Q-Networks (DDQN) agent.
    """
    def __init__(self, state_size, action_size, batch_size, gamma=0.99, epsilon=0.1, epsilon_min=0.0001, epsilon_decay=0.99, learning_rate=1e-4, mem_size=50000, replace_target=100):
        """
        Creates a Double Deep Q-Networks (DDQN) agent.

        :param state_size: number of dimensions of the feature vector of the state.
        :type state_size: int.
        :param action_size: number of actions.
        :type action_size: int.
        :param batch_size: batch size.
        :type batch_size: int.
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
        :param replace_target: how many memories we should analize until updating the DDQN weights.
        :type replace_target: int.
        """
        self.state_size = state_size
        self.height, self.width, self.channels = self.state_size    
        self.action_size = action_size
        self.action_space = [i for i in range(self.action_size)]
        self.batch_size = batch_size
        self.replace_target = replace_target
        self.memory = ReplayBuffer(mem_size, state_size, action_size, True)
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.learning_rate = learning_rate
        self.q_eval = self.make_model()
        self.q_target = self.make_model()

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
        action_values = self.q_eval.predict(state)
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
        self.memory.store_transition(state, action, reward, next_state, done)

    def replay(self, batch_size):
        """
        Learns from memorized experience.

        :param batch_size: size of the minibatch taken from the replay buffer.
        :type batch_size: int.
        :return: loss computed during the neural network training.
        :rtype: float.
        """       
        state, action, reward, next_state, done = self.memory.sample_buffer(self.batch_size)
        action_values = np.array(self.action_space, dtype=np.int8)
        action_indices = np.dot(action, action_values)
        q_next = self.q_target.predict(next_state)
        q_eval = self.q_eval.predict(next_state)
        q_pred = self.q_eval.predict(state)
        max_actions = np.argmax(q_eval, axis=1)
        q_target = q_pred
        batch_index = np.arange(batch_size, dtype=np.int32)
        q_target[batch_index, action_indices] = reward + self.gamma * q_next[batch_index, max_actions.astype(int)]*done
        history = self.q_eval.fit(state, q_target, verbose=0)
        if self.memory.mem_cntr % self.replace_target == 0:
            self.updateDQN()
        # Keeping track of loss
        loss = history.history['loss'][0]

        return loss

    def updateDQN(self):
        """
        Updates the DDQN using both neural networks.

        """
        self.q_target.set_weights(self.q_eval.get_weights())

    def load(self, name):
        """
        Loads the neural network's weights from disk.

        :param name: model's name.
        :type name: str.
        """
        self.q_target.load_weights(name)

    def save(self, name):
        """
        Saves the neural network's weights to disk.

        :param name: model's name.
        :type name: str.
        """
        self.q_target.save_weights(name)

    def update_epsilon(self):
        """
        Updates the epsilon used for epsilon-greedy action selection.

        """
        self.epsilon *= self.epsilon_decay
        if self.epsilon < self.epsilon_min:
            self.epsilon = self.epsilon_min

