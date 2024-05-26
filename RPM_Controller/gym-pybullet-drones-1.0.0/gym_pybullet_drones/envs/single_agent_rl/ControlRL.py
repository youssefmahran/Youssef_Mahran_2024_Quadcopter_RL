import numpy as np
from gym import spaces
import pybullet as p 

import sys
sys.path.append('C:/Users/eta1/Desktop/Mazen/Bachelor/gym-pybullet-drones-1.0.0')

from gym_pybullet_drones.envs.BaseAviary import DroneModel, Physics, BaseAviary
from gym_pybullet_drones.envs.single_agent_rl.BaseSingleAgentAviary import ActionType, ObservationType, BaseSingleAgentAviary


class ControlRL(BaseSingleAgentAviary):


    def __init__(self,
                 drone_model: DroneModel=DroneModel.CF2X,
                 initial_xyzs=None,
                 initial_rpys=None,
                 physics: Physics=Physics.PYB,
                 freq: int=240,
                 aggregate_phy_steps: int=1,
                 gui=False,
                 record=False, 
                 obs: ObservationType=ObservationType.QUAT,
                 act: ActionType=ActionType.RPM,
                 bounding_box = [3,3,2]):

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


        # topViewMatrix = p.computeViewMatrix(cameraEyePosition=[0,0,4.6],cameraTargetPosition=[0,0,0],cameraUpVector=[0,1,0])
        # topProjectionMatrix = p.computeProjectionMatrixFOV(fov=45,aspect=1,nearVal=0.1,farVal=4.7)
        # width,height,rgbImg,depthImg,segImg = p.getCameraImage(width=200, height=200,viewMatrix=topViewMatrix, projectionMatrix=topProjectionMatrix,renderer=p.ER_BULLET_HARDWARE_OPENGL, physicsClientId=self.CLIENT)

        self.bounding_box = bounding_box
        #self._boundingboxCollision(self)



    def _computeReward(self):
        """ TODO: 
            Find a suitable reward function
            Add negative reward for collision
        """
        # xbox = self.bounding_box[0]/2
        # ybox = self.bounding_box[1]/2
        # zbox = self.bounding_box[2]
        state = self._getDroneStateVector(0)
        # xdrone = state[0]
        # ydrone = state[1]
        # zdrone = state[2]
        positionerror = -1 * np.linalg.norm(np.array([0,0,1])-state[0:3])
        if(self._boundingboxCollision()):
            return -1000000
        else:
            return 100 * np.exp(positionerror)



    def _computeDone(self):
        """ TODO:
            Add parameters in class to define bounding box and create a check for end conditions
        """
        if self.step_counter/self.SIM_FREQ > 10 or self._boundingboxCollision():
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
        #self.boxID = p.createCollisionShape(shapeType = GEOM_BOX,halfExtents = self.bounding_box)


    def _computeObs(self):
        """Returns the current observation of the environment.

        Returns
        -------
        ndarray
            A Box() of shape (H,W,4) or (12,) depending on the observation type.

        """
        # state = self._getDroneStateVector(0)
        # # roll -> phi || pitch -> theta || yaw -> gamma
        # roll = state[7]
        # pitch = state[8]
        # yaw = state[9]
        # rot00 = np.cos()

        if self.OBS_TYPE == ObservationType.QUAT: 
            obs = self._clipAndNormalizeState(self._getDroneStateVector(0))
            ############################################################
            #### OBS OF SIZE 20 (WITH QUATERNION AND RPMS)
            # return obs
            ############################################################
            #### OBS SPACE OF SIZE 16
            return np.hstack([obs[0:3],obs[3:7], obs[7:10], obs[10:13], obs[13:16]]).reshape(16,)
            ############################################################
        else:
            print("[ERROR] in BaseSingleAgentAviary._computeObs()") 



    def _clipAndNormalizeState(self,
                               state
                               ):


        """ TODO:
            Revision of the code to see if it fits our purpose
        """

        
        """Normalizes a drone's state to the [-1,1] range.

        Parameters
        ----------
        state : ndarray
            (20,)-shaped array of floats containing the non-normalized state of a single drone.

        Returns
        -------
        ndarray
            (20,)-shaped array of floats containing the normalized state of a single drone.

        """
        MAX_LIN_VEL_XY = 3 
        MAX_LIN_VEL_Z = 1

        MAX_XY = MAX_LIN_VEL_XY*self.EPISODE_LEN_SEC
        MAX_Z = MAX_LIN_VEL_Z*self.EPISODE_LEN_SEC

        MAX_PITCH_ROLL = np.pi # Full range

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
        normalized_y = state[9] / np.pi # No reason to clip
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
        # if not(clipped_pos_xy == np.array(state[0:2])).all():
        #     print("[WARNING] it", self.step_counter, "in HoverAviary._clipAndNormalizeState(), clipped xy position [{:.2f} {:.2f}]".format(state[0], state[1]))
        # if not(clipped_pos_z == np.array(state[2])).all():
        #     print("[WARNING] it", self.step_counter, "in HoverAviary._clipAndNormalizeState(), clipped z position [{:.2f}]".format(state[2]))
        # if not(clipped_rp == np.array(state[7:9])).all():
        #     print("[WARNING] it", self.step_counter, "in HoverAviary._clipAndNormalizeState(), clipped roll/pitch [{:.2f} {:.2f}]".format(state[7], state[8]))
        # if not(clipped_vel_xy == np.array(state[10:12])).all():
        #     print("[WARNING] it", self.step_counter, "in HoverAviary._clipAndNormalizeState(), clipped xy velocity [{:.2f} {:.2f}]".format(state[10], state[11]))
        # if not(clipped_vel_z == np.array(state[12])).all():
        #     print("[WARNING] it", self.step_counter, "in HoverAviary._clipAndNormalizeState(), clipped z velocity [{:.2f}]".format(state[12]))


    def _computeInfo(self):
        """Computes the current info dict(s).

        Unused.

        Returns
        -------
        dict[str, int]
            Dummy value.

        """
        return {"answer": 42} #### Calculated by the Deep Thought supercomputer in 7.5M years

    def reset(self):
        """Resets the environment.

        Returns
        -------
        ndarray | dict[..]
            The initial observation, check the specific implementation of `_computeObs()`
            in each subclass for its format.

        """
        p.resetSimulation(physicsClientId=self.CLIENT)
        #### Housekeeping ##########################################
        # p.createVisualShape(shapeType=p.GEOM_SPHERE,radius=0.1,visualFramePosition=[1,1,1],rgbaColor=[0,0,1,0], physicsClientId=self.CLIENT)
        # p.loadURDF("sphere2.urdf",
        #            [1, 1, 1],
        #            p.getQuaternionFromEuler([0,0,0]),
        #            useFixedBase=1,
        #            globalScaling=0.1,
        #            physicsClientId=self.CLIENT)
        self.INIT_XYZS= np.array([[np.random.uniform(-1.3,1.3) , np.random.uniform(-1.3,1.3) , np.random.uniform(0.2,1.8)]])
        #self.INIT_XYZS=np.array([[1,1,1]])
        self._housekeeping()
        #### Update and store the drones kinematic information #####
        self._updateAndStoreKinematicInformation()
        #### Start video recording #################################
        self._startVideoRecording()
        #### Return the initial observation ########################
        return self._computeObs()
