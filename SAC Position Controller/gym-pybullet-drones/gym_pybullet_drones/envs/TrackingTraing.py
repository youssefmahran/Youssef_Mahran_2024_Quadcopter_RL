import os
import numpy as np
import torch
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3 import SAC
from stable_baselines3.sac import MlpPolicy
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback

import sys
sys.path.append('/home/youssefmahran2/Desktop/Youssef_Mahran_2024_Quadcopter_RL/gym-pybullet-drones')

from gym_pybullet_drones.envs.BaseAviary import DroneModel, Physics
from gym_pybullet_drones.envs.BaseRLAviary import ActionType, ObservationType

from TrackingEnv import TrackingEnv

if __name__ == "__main__":

    # Save directory initialization
    save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results', 'Tracking_SAC')
    os.makedirs(save_dir, exist_ok=True)

    # Initialize training environment
    train_env = TrackingEnv(drone_model=DroneModel.CF2X,
                 num_drones=1,
                 initial_xyzs=None,
                 initial_rpys=None,
                 physics=Physics.PYB,
                 pyb_freq= 50,
                 ctrl_freq= 50,
                 gui=False,
                 record=False,
                 obs=ObservationType.POS,
                 act=ActionType.URP
                 )

    # Initialize network architechture
    offpolicy_kwargs = dict(activation_fn=torch.nn.LeakyReLU,
                            net_arch=[400, 300]
                            )
    
    # Initialize exploration noise
    action_noise = NormalActionNoise(mean=np.zeros(train_env.action_space.shape), sigma=0.2 * np.ones(train_env.action_space.shape))

    # Load the best stabilization agent and set the environment
    model_path = "results/Stabilization/best/best_model.zip"
    model = SAC.load(model_path)
    model.set_env(train_env)
    model.tensorboard_log = os.path.join(save_dir, 'tb/')

    # Save the model each 1000 step
    checkpoint_callback = CheckpointCallback(save_freq=1000, save_path=os.path.join(save_dir, 'logs/'), name_prefix='rl_model')
    
    # Evaluate the model each episode and save the best model
    eval_callback = EvalCallback(train_env, best_model_save_path=os.path.join(save_dir, 'best/'),
                             log_path=os.path.join(save_dir, 'best/'), 
                             eval_freq=500,
                             deterministic=True, render=False)
    
    # Start the training
    callback_list = [checkpoint_callback, eval_callback]
    model.learn(
        total_timesteps=int(20e6),
        callback=callback_list,
        log_interval=10
    )

    # Save the model
    model.save(os.path.join(save_dir, 'success_model.zip'))
    print("Model saved at:", os.path.join(save_dir, 'success_model.zip'))
