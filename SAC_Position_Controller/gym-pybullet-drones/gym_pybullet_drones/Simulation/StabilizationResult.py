from stable_baselines3 import SAC
import time

import sys
sys.path.append('')

from gym_pybullet_drones.envs.BaseAviary import DroneModel, Physics
from gym_pybullet_drones.envs.BaseRLAviary import ActionType, ObservationType
from gym_pybullet_drones.envs.StabilizationEnv import StabilizationEnv


# Instantiate the environment
train_env = StabilizationEnv(drone_model=DroneModel.CF2X,
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

# Load your trained RL model
model_path = "results/Tracking/best/best_model.zip"
model = SAC.load(model_path)

# Reset the environment
obs, info = train_env.reset()

# Test for 10 episodess
for i in range(100*int(train_env.CTRL_FREQ/train_env.PYB_STEPS_PER_CTRL)):
    action, _ = model.predict(obs, deterministic=True)
    
    # Step through the environment with the predicted action
    step_result = train_env.step(action)
    time.sleep(0.004)
    
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
# Close the environment
train_env.close()
