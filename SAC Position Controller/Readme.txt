############### REQUIREMENTS ###############
- Download Python 3
	- Python 3.8 is recommended
- Download Anaconda

############### INSTALLATION ###############
- Open a new terminal
- Create a new virtual environment: "conda create -n drones python=3.8"
- Activate the virtual environment: "conda activate drones"
- Download the following libraries:
	- numpy: "pip install numpy"
	- matplotlib: "pip install matplotlib"
	- pybullet: "pip install pybullet"
	- gym: "pip install gymnasium"
	- Pillow: "pip install pillow"
	- Cycler: "pip install cycler"
	- Stable Baselines3: "pip install stable-baselines3[extra]"

############### USE ###############

- To run the stabilization simulation:
	- open Stabilization.py
	- open a new terminal
	- type "$ conda activate drones"
	- type "$ cd ~/gym-pybullet-drones/gym_pybullet_drones/envs"
	- type "$ python3 StabilizationResult.py"

- To run the position tracking simulation:
	- open a new terminal
	- type "$ conda activate drones"
	- type "$ cd ~/gym-pybullet-drones/gym_pybullet_drones/envs"
	- type "$ python3 TrackingResult.py"

- To run the trajectory tracking simulation:
	- open a new terminal
	- type "$ conda activate drones"
	- type "$ cd ~/gym-pybullet-drones/gym_pybullet_drones/envs"
	- type "$ python3 InfinitySymbol.py" or "$python3 Helix.py"
