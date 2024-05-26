from os import stat, truncate
import numpy as np
from gym import spaces
import pybullet as p 

from gym_pybullet_drones.envs.BaseAviary import DroneModel, Physics, BaseAviary
from gym_pybullet_drones.envs.single_agent_rl.BaseSingleAgentAviary import ActionType, ObservationType, BaseSingleAgentAviary
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl

class LargeSpaceEnv(BaseSingleAgentAviary):

    def __init__(self,
                 drone_model: DroneModel=DroneModel.CF2X,
                 initial_xyzs=None,
                 initial_rpys=None,
                 physics: Physics=Physics.PYB,
                 freq: int=50,
                 aggregate_phy_steps: int=1,
                 gui=True,
                 record=False, 
                 obs: ObservationType=ObservationType.MOKH,
                 act: ActionType=ActionType.RPM,
                 bounding_box = [6,6,3],
                 drone_target = [0,0,1],
                 first_call = True):

        super().__init__(drone_model=drone_model,
                         initial_xyzs=initial_xyzs,
                         initial_rpys=initial_rpys,
                         physics=physics,
                         freq=freq,
                         aggregate_phy_steps=aggregate_phy_steps,
                         gui=gui,
                         record=record,
                         obs=obs,
                         act=act
                         )        

        self.drone_target = drone_target
        self.bounding_box = bounding_box
        self.first_call = first_call
        self.last_last_clipped_action = self.last_clipped_action[0]

    def _computeReward(self):
        if self.first_call :
            p.loadURDF("sphere2red_nocol.urdf",
                   self.drone_target,
                   p.getQuaternionFromEuler([0, 0, 0]),
                   useFixedBase=1,
                   globalScaling=0.05,
                   physicsClientId=self.CLIENT
                   )
            self.first_call = False
        a = 7
        dx,dy,dz = self._posDiff()
        e_k = np.sqrt(dx**2 + dy**2 + dz**2)
        er1 = 1/(a*e_k)
        exp = (-0.5) * ((e_k/0.5)**2)
        den = np.sqrt(2*np.pi*(0.5**2))
        er2 = (a/den)*np.exp(exp)
        reward = er1 + er2
        return reward

    def _computeDone(self):
        if self.step_counter/self.SIM_FREQ > 10: #or self._boundingboxCollision() or self._isUpright():
            self.first_call = True
            return True
        else:
            return False

    def _boundingboxCollision(self):
        xbox = self.bounding_box[0]/2
        ybox = self.bounding_box[1]/2
        zbox = self.bounding_box[2]
        state = self._getDroneStateVector(0)
        xdrone = state[0]
        ydrone = state[1]
        zdrone = state[2]
        if(xdrone<=xbox-0.1 and xdrone>=-xbox+0.1 and ydrone<=ybox-0.1 and ydrone>=-ybox+0.1 and zdrone<=zbox-0.1 and zdrone>0.1):
            return False
        else:
            return True

    def _isUpright(self):
        state = self._getDroneStateVector(0)
        roll = state[7]
        pitch = state[8]
        yaw = state[9]
        if np.abs(roll) > np.pi/2 or np.abs(pitch) > np.pi/2 or np.abs(yaw) > np.pi/2:
            return True
        else:
            return False

    def _posDiff(self):
        state = self._getDroneStateVector(0)
        xdiff = state[0] - self.drone_target[0]
        ydiff = state[1] - self.drone_target[1]
        zdiff = state[2] - self.drone_target[2]
        return xdiff,ydiff,zdiff

    def _computeObs(self):
        if self.OBS_TYPE == ObservationType.MOKH: 
            obs = self._clipAndNormalizeState(self._getDroneStateVector(0))
            differences = self._posDiff()
            return np.hstack([obs[7:10], obs[10:13], obs[13:16],differences]).reshape(12,)
        else:
            print("[ERROR] in BaseSingleAgentAviary._computeObs()") 

    def _clipAndNormalizeState(self,state):
        MAX_LIN_VEL_XY = 3 
        MAX_LIN_VEL_Z = 1

        MAX_XY = MAX_LIN_VEL_XY*self.EPISODE_LEN_SEC
        MAX_Z = MAX_LIN_VEL_Z*self.EPISODE_LEN_SEC

        MAX_PITCH_ROLL = np.pi

        clipped_pos_xy = np.clip(state[0:2], -MAX_XY, MAX_XY)
        clipped_pos_z = np.clip(state[2], 0, MAX_Z)
        clipped_rp = np.clip(state[7:9], -MAX_PITCH_ROLL, MAX_PITCH_ROLL)
        clipped_vel_xy = np.clip(state[10:12], -MAX_LIN_VEL_XY, MAX_LIN_VEL_XY)
        clipped_vel_z = np.clip(state[12], -MAX_LIN_VEL_Z, MAX_LIN_VEL_Z)

        if self.GUI:
            self._clipAndNormalizeStateWarning(state,
                                               clipped_pos_xy,
                                               clipped_pos_z,
                                               clipped_rp,
                                               clipped_vel_xy,
                                               clipped_vel_z
                                               )

        normalized_pos_xy = clipped_pos_xy / MAX_XY
        normalized_pos_z = clipped_pos_z / MAX_Z
        normalized_rp = clipped_rp / MAX_PITCH_ROLL
        normalized_y = state[9] / np.pi
        normalized_vel_xy = clipped_vel_xy / MAX_LIN_VEL_XY
        normalized_vel_z = clipped_vel_z / MAX_LIN_VEL_XY
        normalized_ang_vel = state[13:16]/np.linalg.norm(state[13:16]) if np.linalg.norm(state[13:16]) != 0 else state[13:16]

        norm_and_clipped = np.hstack([normalized_pos_xy,
                                      normalized_pos_z,
                                      state[3:7],
                                      normalized_rp,
                                      normalized_y,
                                      normalized_vel_xy,
                                      normalized_vel_z,
                                      normalized_ang_vel,
                                      state[16:20]
                                      ]).reshape(20,)

        return norm_and_clipped

    def _clipAndNormalizeStateWarning(self,
                                      state,
                                      clipped_pos_xy,
                                      clipped_pos_z,
                                      clipped_rp,
                                      clipped_vel_xy,
                                      clipped_vel_z,
                                      ):
        """ _TODO:
            Revision of the code to see if it fits our purpose
        """


        """Debugging printouts associated to `_clipAndNormalizeState`.

        Print a warning if values in a state vector is out of the clipping range.
        
        """

    def _computeInfo(self):
        """Computes the current info dict(s).

        Unused.

        Returns
        -------
        dict[str, int]
            Dummy value.

        """
        return {"answer": 42} 

    def reset(self):
        p.resetSimulation(physicsClientId=self.CLIENT)
        self.INIT_XYZS= np.array([[np.random.uniform(-2.5,2.5) , np.random.uniform(-2.5,2.5) ,np.random.uniform(0.2,2.5)]])
        self._housekeeping()
        self._updateAndStoreKinematicInformation()
        self._startVideoRecording()
        return self._computeObs()
