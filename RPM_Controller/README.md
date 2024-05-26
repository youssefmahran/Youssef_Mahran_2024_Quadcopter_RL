# Low-level RPM Controller
## REQUIREMENTS 
- Download Python 3.6
- Download Anaconda
## INSTALLATION
- Open a new terminal
- Create a new virtual environment
```
 conda create -n baselines python=3.6
```
- Activate the virtual environment
```
conda activate baselines
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
conda install -c anaconda numpy pillow
conda install -c conda-forge matplotlib pybullet gym cycler stable-baselines3
conda install pytorch torchvision torchaudio cpuonly -c pytorch
```
## USE
Open [`TD3_Small_Space_Result.py`](Codes/TD3_Small_Space_Result.py), [`TD3_Large_Space_Result.py`](Codes/TD3_Large_Space_Result.py), [`SAC_Small_Space_Result.py`](Codes/SAC_Small_Space_Result.py) and [`SAC_Large_Space_Result.py`](Codes/SAC_Large_Space_Result.py) found in [`Codes`](Codes/) folder and add the path of "gym_pybullet_drones" folder to line 5. 
For example:
```
   sys.path.append('/path/to/Youssef_Mahran_2024_Quadcopter_RL/RPM_Controller/gym-pybullet-drones-1.0.0')
```
- To run the TD3 Small Space simulation:
	- Open a new terminal
	- Run the file using the created virtual environment as the python interpreter using:
```
cd /path/to/Youssef_Mahran_2024_Quadcopter_RL/RPM_Controller/Codes/
conda activate baselines
python3 TD3_Small_Space_Result.py
```

- To run the TD3 Large Space simulation:
	- Open a new terminal
	- Run the file using the created virtual environment as the python interpreter using:
```
cd /path/to/Youssef_Mahran_2024_Quadcopter_RL/RPM_Controller/Codes/
conda activate baselines
python3 TD3_Large_Space_Result.py
```

- To run the SAC Small Space simulation:
	- Open a new terminal
	- Run the file using the created virtual environment as the python interpreter using:
```
cd /path/to/Youssef_Mahran_2024_Quadcopter_RL/RPM_Controller/Codes/
conda activate baselines
python3 SAC_Small_Space_Result.py
```

- To run the SAC Large Space simulation:
```
cd /path/to/Youssef_Mahran_2024_Quadcopter_RL/RPM_Controller/Codes/
conda activate baselines
python3 SAC_Large_Space_Result.py
```
