import gym

import enviromentDino


env = enviromentDino.Dino()
env.reset()
for i in range(1000):
    env.render()
    env.step(env.action_space.sample()) # take a random action
env.close()