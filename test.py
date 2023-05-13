from make_env import make_env
import numpy as np
import math
env = make_env('simple_spread', True)
obs = env.reset()

print(env.step(np.array([1,1])))
print(env.step(np.array([1,1])))
print(env.step(np.array([1,1])))
print(env.step(np.array([1,1])))







