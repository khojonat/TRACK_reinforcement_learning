import sys
sys.path.insert(0, 'src')  # Change this to location of PyTrack src
from runTrack import *
from config import Config
from utils import *
from hdf5Track import *

import numpy as np
import gym

from gym import Env, spaces

# Environment class:

class TRACKenv(Env):

    def __init__(self):
        super(TRACKenv, self).__init__()
        self.observation_shape = (6,)# Let's do simple thing first with the observation being the number of particles left.
        self.observation_space = spaces.Box(low = -8.0, high = 8.0, shape =(6,) , dtype = np.float32)
        # Ex: spaces.Box(low = np.zeros(self.observation_shape),
                               #             high = np.ones(self.observation_shape),
                               #             dtype = np.float16)

        self.action_space = spaces.Box(low = -8.0, high = 8.0, shape =(6,) , dtype = np.float32)
        self.reward_range = (0,10000)
        # Ex: gym.spaces.Box(low = 0.,high = 6.,shape = (1,))

        # Can define other parameters of environment here
        self.observation = np.array([0,0,0,0,0,0], dtype = np.float32)  # represents the 6 initial voltages
        self.trials = 0  # trials counter

        self.cf = Config()
        # remove test folder
        rmfolder(cf.CHILD_TRACK,'test')
        copy2folder(cf.PARENT_TRACK,cf.CHILD_TRACK,'test')  # create folder
        self.sim_folder = str(cf.CHILD_TRACK)+'/test'

    def run(self, run_config, sim_folder):
        """
        Custom run file to run one TRACK configurate and extract the required information
        Note: the type of beam element must be a constant.
        1. copy defult configurate of TRACK.dat and sclinac.dat
        2. update TRACK.dat and sclinac.dat file and save inputs
        3. run TRACK and save distributions at corresponding locations.

        Inputs:
        run_config - dirction which contains important information about the track file, sclinac file, and locations
        sim_folder - location of simualation folder
        hdf5_file - location of hdf5 file
        trial - trial number

        Outputs:
        df - info about the particle distribtuion
        beam - info about the beam like emmitance and particle loss
        """
        cur_track = self.cf.TRACK_DAT
        cur_sclinac = self.cf.SCLINAC
        cur_sclinac0 = self.cf.SCLINAC0
        track_file = f"{sim_folder}/track.dat"
        sclinac_file = f"{sim_folder}/sclinac.dat"

        track_settings = run_config['track']  # given the track setting we want to update, this should update the track settings
        for i in track_settings:  # just use a dictionary so it's easy to configure.
            cur_track[i] = track_settings[i]

        # updating sclinac will be harder so we'll have to use indexes [element][argument][value]
        sclinac_setting = run_config['sclinac']
        for i in sclinac_setting:
            cur_sclinac[i[0]][i[1]] = i[2]

        make_track(track_file, cur_track)  # update track.dat file
        make_sclinac(sclinac_file, cur_sclinac)  # update sclinac.dat file
        df = get_distro(sim_folder, cur_sclinac, cur_track, -1)
        beam = get_beamout(sim_folder)
        return df, beam

  # Reinitialize the environment
    def reset(self, seed=None, return_info=False, options=None):
        # Don't really need this since the environment resets every time already.
        # Unless we want to start off at a certain voltage setting that is
        # remove test folder
        rmfolder(self.cf.CHILD_TRACK,'test')

        copy2folder(self.cf.PARENT_TRACK,self.cf.CHILD_TRACK,'test')  # create folder
        self.sim_folder = str(self.cf.CHILD_TRACK)+'/test'
        self.observation = ((np.random.rand(6,)-.5)*8).astype('float32')
        if return_info:
            info = {}
            return self.observation, info
        return self.observation

  # Agent takes a step in the environment, returns reward and whether the episode is finished
    def step(self, action):
        V = self.observation + action
        for i in range(len(V)):  # hard code bounds
            if np.abs(V[i]) > 8:
                V[i] = np.sign(V[i])*8
        run_config = {'track': {},
             'sclinac': [[4,2,V[0]],[5,2,V[1]],[7,2,V[2]],[8,2,V[3]],[10,2,V[4]],[11,2,V[5]]],
             'inputs':{'05_eq3d':{'voltage':V[0]},  # first index in name, second index is value
                       '06_eq3d':{'voltage':V[1]},
                      '08_eq3d':{'voltage':V[2]},
                      '09_eq3d':{'voltage':V[3]},
                      '11_eq3d':{'voltage':V[4]},
                      '12_eq3d':{'voltage':V[5]}}
             }
        dist, beam = self.run(run_config, self.sim_folder)

        new_obs = V
        self.observation = new_obs

        # Flag that marks the termination of an episode, to be triggered as desired
        # episode finished when loss reaches min or after a certain number of trials.
        done = False
        self.trials +=1

        reward = beam['#of_part_left'].values[-1]  # higher rewards the more particles that are left.

        if reward > 10000*.95:  # done if reward is greater than a number
            done = True
            self.reset()
        if self.trials > 99:  # if trials over 100, also end
            done = True
            self.reset()

        info = {}

        return new_obs, reward, done, info

  # Can optionally add a function to render the environment

  # def render(self)
