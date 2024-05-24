import gymnasium as gym
from stable_baselines3 import SAC

import sys
sys.path.append('/home/youssefmahran2/Desktop/Youssef_Mahran_2024_Quadcopter_RL/gym-pybullet-drones')

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

#Logging Arrays
start = time.time()
time_data = []
x_position_data = []
y_position_data = []
z_position_data = []
r_position_data = []
p_position_data = []
yaw_position_data = []


while flag:
    action, _ = model.predict(obs, deterministic=True)
    
    # Step through the environment with the predicted action
    step_result = train_env.step(action)
    train_env._getDroneStateVector(0)
    state = train_env._getDroneStateVector(0)
    #Get Current States
    x = state[0]
    y = state[1]
    z = state[2]
    r = state[7]
    p = state[8]
    yaw = state[9]
    
    #Add states to logging Arrays
    time_data.append(train_env.step_counter/train_env.CTRL_FREQ)
    x_position_data.append(x)
    y_position_data.append(y)
    z_position_data.append(z)
    r_position_data.append(r)
    p_position_data.append(p)
    yaw_position_data.append(yaw)
    
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

#Save the data into a CSV file
csv_file = "trajectory.csv"
with open(csv_file, mode='w', newline='') as file:
    writer = csv.writer(file)
    for t, x, y, z, r, p, yaw in zip(time_data, x_position_data, y_position_data, z_position_data, r_position_data, p_position_data, yaw_position_data):
        writer.writerow([t, x, y, z, r, p, yaw])
    print("CSV Success")
