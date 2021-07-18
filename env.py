import gym

import enviromentDino


env = enviromentDino.Dino()
height, width, channels = env.observation_space.shape


for i in range(1000):
   env.render()
   env.step(env.action_space.sample()) # take a random action
env.close()