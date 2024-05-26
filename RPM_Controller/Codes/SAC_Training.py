import os
import time
from datetime import datetime
import argparse
import subprocess
import numpy as np
import gym
import torch
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import SubprocVecEnv, VecTransposeImage
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3 import SAC 
from stable_baselines3 import TD3
from stable_baselines3 import DDPG
from stable_baselines3.td3 import MlpPolicy as td3ddpgMlpPolicy
from stable_baselines3.td3 import CnnPolicy as td3ddpgCnnPolicy
from stable_baselines3.a2c import MlpPolicy as a2cppoMlpPolicy
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback, StopTrainingOnRewardThreshold
from stable_baselines3.common.noise import ActionNoise
import sys
sys.path.append('/home/lab/mahran/gym-pybullet-drones-1.0.0')
from gym_pybullet_drones.envs.BaseAviary import Physics, DroneModel
from gym_pybullet_drones.envs.single_agent_rl.BaseSingleAgentAviary import ActionType, ObservationType
from ControlRLMokh2 import ControlRL2

if __name__ == "__main__":
    # Save directory
    save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results', 'mokh_sac_2_5')
    os.makedirs(save_dir, exist_ok=True)
    train_env = ControlRL2(drone_model=DroneModel.CF2X,
                 initial_xyzs=None,
                 initial_rpys=None,
                 physics=Physics.PYB,
                 freq=50,
                 aggregate_phy_steps=1,
                 gui=False,
                 record=False, 
                 obs=ObservationType.MOKH,
                 act=ActionType.RPM,
                 bounding_box = [6,6,3],
                 drone_target = [0,0,1],
                 first_call = True)

    # TD3 Model
    offpolicy_kwargs = dict(activation_fn=torch.nn.LeakyReLU,
                            net_arch=[400, 300]
                            ) # or None # or dict(net_arch=dict(qf=[256, 128, 64, 32], pi=[256, 128, 64, 32]))
    
    model = SAC(
        "MlpPolicy",  # You need to specify the policy class, e.g., "MlpPolicy"
        train_env,
        learning_rate=7e-4,
        batch_size=256,
        action_noise= NormalActionNoise(mean=0,sigma=0.2),
        gamma=0.99,
        buffer_size=2000000,
        tensorboard_log=os.path.join(save_dir, 'tb/'),
        device="cpu",
        verbose=1,
        policy_kwargs=offpolicy_kwargs
    )

    # Train the model
    checkpoint_callback = CheckpointCallback(save_freq=1000, save_path=os.path.join(save_dir, 'logs/'), name_prefix='rl_model')
    
    eval_callback = EvalCallback(train_env, best_model_save_path=os.path.join(save_dir, 'best/'),
                             log_path=os.path.join(save_dir, 'best/'), 
                             eval_freq=502,
                             deterministic=True, render=False)
    callback_list = [checkpoint_callback, eval_callback]
    model.learn(
        total_timesteps=int(3.671e6),
        callback=callback_list,
        log_interval=10
    )

    # Save the model
    model.save(os.path.join(save_dir, 'success_model.zip'))
    print("Model saved at:", os.path.join(save_dir, 'success_model.zip'))
