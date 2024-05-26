"""Test script for single agent problems.

This scripts runs the best model found by one of the executions of `singleagent.py`

Example
-------
To run the script, type in a terminal:

    $ python test_singleagent.py --exp ./results/save-<env>-<algo>-<obs>-<act>-<time_date>

"""
import os
import time
from datetime import datetime
import argparse
import re
import numpy as np
import gym
import torch
import matplotlib.pyplot  as plt
from stable_baselines3.common.env_checker import check_env
from stable_baselines3 import TD3
from stable_baselines3.td3 import MlpPolicy as td3ddpgMlpPolicy
from stable_baselines3.td3 import CnnPolicy as td3ddpgCnnPolicy
from stable_baselines3.common.evaluation import evaluate_policy

import math
import csv

import sys
sys.path.append('')

from gym_pybullet_drones.utils.utils import sync
from gym_pybullet_drones.utils.Logger import Logger
from gym_pybullet_drones.envs.single_agent_rl.BaseSingleAgentAviary import ActionType, ObservationType
from gym_pybullet_drones.envs.BaseAviary import DroneModel, Physics, BaseAviary
from LargeSpace import LargeSpaceEnv


if __name__ == "__main__":

    #### Load the model from file ##############################
  
    path = os.path.dirname(os.path.abspath(__file__)) + '/results/TD3_Large_Space/best/best_model.zip'
    
    model = TD3.load(path) 


    #### Parameters to recreate the environment ################
    train_env = SmallSpaceEnv(drone_model=DroneModel.CF2P,
                 initial_xyzs=None,
                 initial_rpys=None,
                 physics=Physics.PYB,
                 freq=50,
                 aggregate_phy_steps=1,
                 gui=True,
                 record=False, 
                 obs=ObservationType.MOKH,
                 act=ActionType.RPM,
                 bounding_box = [6,6,3],
                 drone_target = [0,0,1],
                 first_call = True)


    logger = Logger(logging_freq_hz=int(train_env.SIM_FREQ/train_env.AGGR_PHY_STEPS),
                     num_drones=1
                     )
    obs = train_env.reset()
    for i in range(100*int(train_env.AGGR_PHY_STEPS/train_env.TIMESTEP)): # Up to 6''
        action, _states = model.predict(obs,
                                        deterministic=True # OPTIONAL 'deterministic=False'
                                        )
        obs, reward, done, info = train_env.step(action)
        train_env.render()
        if done:
            train_env.reset()
    train_env.close()
