import numpy as np 
import gym

from gym import Env, spaces

# Environment class:

class TRACKenv(Env):
  
  def __init__(self):
      super(TRACKenv, self).__init__()
      self.observation_shape = # Ex: (600, 800, 3)
      self.observation_space = # Ex: spaces.Box(low = np.zeros(self.observation_shape), 
                               #             high = np.ones(self.observation_shape),
                               #             dtype = np.float16)

      self.action_space =      # Ex: gym.spaces.Box(low = 0.,high = 6.,shape = (1,))
      
      # Can define other parameters of environment here
      
  # Reinitialize the environment
  def reset(self):
    
    return 
  
  # Agent takes a step in the environment, returns reward and whether the episode is finished
  def step(self, action):
    
    # Flag that marks the termination of an episode, to be triggered as desired
    done = False
    
    reward = 
    
    # if _____ :
    #   done = True
    
    return reward, done
  
  # Can optionally add a function to render the environment
  
  # def render(self)
