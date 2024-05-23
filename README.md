# Soft Actor-Critic Position Controller
## REQUIREMENTS 
- Download Python >=3.8
- Download Anaconda
## INSTALLATION
- Open a new terminal
- Create a new virtual environment
```
 conda create -n drones python=3.8
```
- Activate the virtual environment
```
conda activate myenv
```
- Download the following libraries:
	- numpy
	- matplotlib
	- pybullet
	- gym
	- Pillow
	- Cycler
	- Stable Baselines3
	- PyTorch
```
pip install numpy matplotlib pybullet gymnasium pillow cycler stable-baselines3[extra] torch torchvision torchaudio 
```
## USE
Open `Helix.py`, `InfinitySymbol.py`, `StabilizationResult.py` and `TrackingResult.py` found in `'Youssef_Mahran_2024_Quadcopter_RL/gym-pybullet-drones/gym_pybullet_drones/Simulation'` and add the path of "gym_pybullet_drones" folder to line 5. 
For example:
```
   sys.path.append('/home/youssefmahran2/Desktop/Youssef_Mahran_2024_Quadcopter_RL/gym-pybullet-drones/')
```
- To run the stabilization simulation:
	- Open a new terminal
	- Run the file using the created virtual environment as the python interpreter using:
```
cd /the/path/to/Youssef_Mahran_2024_Quadcopter_RL/gym-pybullet-drones/gym_pybullet_drones/Simulation/
conda activate drones
python3 StabilizationResult.py
```

- To run the position tracking simulation:
	- Open a new terminal
	- Run the file using the created virtual environment as the python interpreter using:
```
cd /the/path/to/Youssef_Mahran_2024_Quadcopter_RL/gym-pybullet-drones/gym_pybullet_drones/Simulation/
conda activate drones
python3 TrackingResult.py
```

- To run the trajectory tracking simulation:
	- Open a new terminal
	- Run the file using the created virtual environment as the python interpreter using:
```
cd /the/path/to/Youssef_Mahran_2024_Quadcopter_RL/gym-pybullet-drones/gym_pybullet_drones/Simulation/
conda activate drones
python3 InfinitySymbol.py
```
or
```
cd /the/path/to/Youssef_Mahran_2024_Quadcopter_RL/gym-pybullet-drones/gym_pybullet_drones/Simulation/
conda activate drones
python3 Helix.py
```
