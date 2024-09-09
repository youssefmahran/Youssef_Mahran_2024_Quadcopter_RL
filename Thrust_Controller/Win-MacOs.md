# Soft Actor-Critic Thrust Vector Controller
## REQUIREMENTS 
- Download and install [Anaconda](https://www.anaconda.com/download/success)
## INSTALLATION
- Open a new Anaconda Prompt
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
Open [`Helix.py`](Simulation/Helix.py), [`InfinitySymbol.py`](Simulation/InfinitySymbol.py), [`StabilizationResult.py`](Simulation/StabilizationResult.py) and [`TrackingResult.py`](Simulation/TrackingResult.py) found in [`Simulaton`](Simulation) folder and add the path of "gym_pybullet_drones" folder to line 5. 
For example:
```
   sys.path.append('/path/to/Thrust_Controller/')
```
- To run the stabilization simulation:
	- Open a new Anaconda Prompt
	- Run the file using the created virtual environment as the python interpreter using:
```
cd /path/to/SAC_Position_Controller/Simulation/
conda activate drones
python3 StabilizationResult.py
```

- To run the position tracking simulation:
	- Open a new Anaconda Prompt
	- Run the file using the created virtual environment as the python interpreter using:
```
cd /path/to/Thrust_Controller/Simulation/
conda activate drones
python3 TrackingResult.py
```

- To run the trajectory tracking simulation:
	- Open a new Anaconda Prompt
	- Run the file using the created virtual environment as the python interpreter using:
```
cd /path/to/Thrust_Controller/Simulation/
conda activate drones
python3 InfinitySymbol.py
```
or
```
cd /path/to/Thrust_Controller/Simulation/
conda activate drones
python3 Helix.py
```
