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
conda activate drones
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
Open [`Helix.py`](gym_pybullet_drones/Simulation/Helix.py), [`InfinitySymbol.py`](gym_pybullet_drones/Simulation/InfinitySymbol.py), [`StabilizationResult.py`](gym_pybullet_drones/Simulation/StabilizationResult.py) and [`TrackingResult.py`](gym_pybullet_drones/Simulation/TrackingResult.py) found in [`Simulaton`](gym_pybullet_drones/Simulation) folder and add the path of "gym_pybullet_drones" folder to line 5. 
For example:
```
   sys.path.append('/path/to/Youssef_Mahran_2024_Quadcopter_RL/SAC_Position_Controller/')
```
- To run the stabilization simulation:
	- Open a new terminal
	- Run the file using the created virtual environment as the python interpreter using:
```
cd /path/to/Youssef_Mahran_2024_Quadcopter_RL/SAC_Position_Controller/gym_pybullet_drones/Simulation/
conda activate drones
python3 StabilizationResult.py
```

- To run the position tracking simulation:
	- Open a new terminal
	- Run the file using the created virtual environment as the python interpreter using:
```
cd /path/to/Youssef_Mahran_2024_Quadcopter_RL/SAC_Position_Controller/gym_pybullet_drones/Simulation/
conda activate drones
python3 TrackingResult.py
```

- To run the trajectory tracking simulation:
	- Open a new terminal
	- Run the file using the created virtual environment as the python interpreter using:
```
cd /path/to/Youssef_Mahran_2024_Quadcopter_RL/SAC_Position_Controller/gym_pybullet_drones/Simulation/
conda activate drones
python3 InfinitySymbol.py
```
or
```
cd /path/to/Youssef_Mahran_2024_Quadcopter_RL/SAC_Position_Controller/gym_pybullet_drones/Simulation/
conda activate drones
python3 Helix.py
```
