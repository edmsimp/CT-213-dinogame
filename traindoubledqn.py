import os
import gym
import enviromentDino
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from doubledqnagent import DoubleAgent
from dqnagent import Agent
import tensorflow as tf

NUM_EPISODES = 30000  # Number of episodes used for training
RENDER = False  # If the Dino Game environment should be rendered
fig_format = 'png'  # Format used for saving matplotlib's figures
# fig_format = 'eps'
# fig_format = 'svg'

# Comment this line to enable training using your GPU
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

tf.compat.v1.disable_eager_execution()

# Initiating the Dino Game environment
env = enviromentDino.Dino()
state_size = env.observation_space.shape
action_size = env.action_space.n

batch_size = 32  # batch size used for the experience replay

# DDQN Agent
fig_name = 'ddqn_training'
agent = DoubleAgent(state_size, action_size, batch_size)

# Checking if weights from previous learning session exists
if os.path.exists("dino_game.h5"):
    print('Loading weights from previous learning session.')
    agent.load("dino_game.h5")
else:
    print('No weights found from previous learning session.')
done = False
return_history = []

for episodes in range(1, NUM_EPISODES + 1):
    # Reset the environment
    state = env.reset()
    state = np.expand_dims(state, axis=0)
    # Cumulative reward is the return since the beginning of the episode
    cumulative_reward = 0.0
    for time in range(1, 5000):
        if RENDER:
            env.render()  # Render the environment for visualization
        # Select action
        action = agent.act(state)
        # Take action, observe reward and new state
        next_state, reward, done, _ = env.step(action)
        # Reshaping to keep compatibility with Keras
        next_state = np.expand_dims(next_state, axis=0)
        # Appending this experience to the experience replay buffer
        agent.append_experience(state, action, reward, next_state, done)
        state = next_state
        # Accumulate reward
        cumulative_reward = enviromentDino.game.get_score()

        if done:
            print("episode: {}/{}, time: {}, score: {}, epsilon: {:.3}"
                  .format(episodes, NUM_EPISODES, time, cumulative_reward, agent.epsilon))
            break
        # We only update the policy if we already have enough experience in memory
        if agent.memory.mem_cntr > 2 * batch_size:
            loss = agent.replay(batch_size)
    return_history.append(cumulative_reward)
    agent.update_epsilon()
    # Every 10 episodes, update the plot for training monitoring
    if episodes % 20 == 0:
        fig=plt.figure()
        plt.plot(return_history, 'b')
        plt.xlabel('Episode')
        plt.ylabel('Return')
       #     plt.show(block=False)
        plt.pause(0.1)
        plt.savefig(fig_name + fig_format, fig_format=fig_format)
        # Saving the model to disk
        agent.save("dino_game.h5")
plt.pause(1.0)
