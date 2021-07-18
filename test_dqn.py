from dqnagent import Agent
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Convolution2D
from tensorflow.keras.optimizers import Adam
import gym
import os
import enviromentDino
from rl.memory import SequentialMemory
from rl.policy import LinearAnnealedPolicy, EpsGreedyQPolicy


#Comment this line to enable training using your GPU
#os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

env = enviromentDino.Dino()

space = env.observation_space.shape
actions = env.action_space.n

agent = Agent(space, actions)
model = agent.model

dqn = agent.build_agent(model, actions)
dqn.compile(Adam(lr=2e-5,))
if os.path.exists('SavedWeights/1m/dqn_weights.h5f'):
    print('Loading weights from previous learning session.')
    dqn.load_weights('SavedWeights/1m/dqn_weights.h5f')
else:
    print('No weights found from previous learning session.')
dqn.fit(env, nb_steps=10000, visualize=False, verbose=2)

scores = dqn.test(env, nb_episodes=10, visualize=True)
print(np.mean(scores.history['episode_reward']))

dqn.save_weights('SavedWeights/10k-Fast/dqn_weights.h5f')

del model, dqn

