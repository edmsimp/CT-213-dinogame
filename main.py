import gym

import enviromentDino


env = enviromentDino.Dino()
env.reset()
for i in range(1000):
    score = 0
    done = False
    env.render()
    state,reward,done, info = env.step(env.action_space.sample()) # take a random action
   
env.close()