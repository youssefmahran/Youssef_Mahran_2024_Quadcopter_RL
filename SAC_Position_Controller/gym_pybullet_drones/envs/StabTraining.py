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

from StabilizationEnv import StabilizationEnv

if __name__ == "__main__":

    # Save directory initialization
    save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results', 'Stabilization_SAC')
    os.makedirs(save_dir, exist_ok=True)

    # Initialize training environment
    train_env = StabilizationEnv(drone_model=DroneModel.CF2X,
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
    

    # Initialize model
    model = SAC(
        "MlpPolicy",
        train_env,
        learning_rate=7e-4,
        buffer_size=1000000,
        batch_size=256,
        gamma=0.99,
        tensorboard_log=os.path.join(save_dir, 'tb/'),
        device="cuda" if torch.cuda.is_available() else "cpu",
        verbose=1,
        policy_kwargs=offpolicy_kwargs
    )

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
