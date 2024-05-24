import gymnasium as gym
from stable_baselines3 import SAC

import sys
sys.path.append('')

from gym_pybullet_drones.envs.BaseAviary import DroneModel, Physics,  ActionType, ObservationType

from gym_pybullet_drones.envs.HelixEnv import TrajectoryEnv

if __name__ == "__main__":

    # Initialize the environment
    train_env = TrajectoryEnv(drone_model=DroneModel.CF2X,
                 num_drones=1,
                 initial_xyzs=None,
                 initial_rpys=None,
                 physics=Physics.PYB,
                 pyb_freq= 50,
                 ctrl_freq= 50,
                 gui=True,
                 record=False,
                 obs=ObservationType.POS,
                 act=ActionType.URP
                 )

# Load the trained agent
model_path = "results/Tracking/best/best_model.zip"
model = SAC.load(model_path)

# Reset the environment
obs, info = train_env.reset()
flag = True
while flag:
    action, _ = model.predict(obs, deterministic=True)
    
    # Step through the environment with the predicted action
    step_result = train_env.step(action)

    
    # Unpack the step result
    obs, reward, terminated, truncated, info = step_result
    
    # Print out information
    print(f"Observation: {obs}")
    print(f"Reward: {reward}")
    print(f"Terminated: {terminated}")
    print(f"Truncated: {truncated}")
    print(f"Info: {info}")

    # If terminated, reset the environment
    if terminated or truncated:
        obs, info = train_env.reset()
        flag = False
# Close the environment
train_env.close()
