from __future__ import annotations

import torch
torch.cuda.empty_cache()
torch.cuda.ipc_collect()
import os
import math
import random


import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg,RigidObjectCfg,RigidObject
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg, ViewerCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils import configclass
from isaaclab.sensors import CameraCfg,Camera
from isaaclab.actuators import ImplicitActuatorCfg 

from cfg.robotcfg import SPOT_CFG
from controller.spot_operational_space import OperationSpaceController
import torch.nn.functional as F

from .utils import set_goal_position, set_goal_orientation, quat2mat, mat2euler, euler2mat, mat2quat, axisangle2quat
import numpy as np
import math


@configclass
class SpotCurtainEnvCfg(DirectRLEnvCfg):
    # env
    decimation = 5
    episode_length_s = 6000.0
    action_space = 12
    observation_space = 144
    camera = True

    viewer = ViewerCfg(eye=(-2, 0.0, 2.0), lookat=(3.0, 4.0, 0.0))

    # simulation
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 200,
        render_interval=decimation,
        use_fabric= True
        )
    root = os.getcwd()
    # robot need to change
    robot_cfg: ArticulationCfg = SPOT_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    robot_cfg.spawn.usd_path = root + '/asset/spot/spot_wrist.usd'
    robot_cfg.init_state.pos = (-0.4, 1.55, 0.4)
    robot_cfg.spawn.activate_contact_sensors = False

    camera_cfg: CameraCfg = CameraCfg(
        prim_path="/World/envs/env_.*/Robot/spot/body/center_camera/center_camera",
        update_period=0.1,
        height=480,
        width=640,
        data_types=["rgb"], #, "depth"],
        spawn= None
    )
    wrist_camera_cfg: CameraCfg = CameraCfg(
        prim_path="/World/envs/env_.*/Robot/spot/arm_wr1/wrist_camera",
        update_period=0.1,
        height=480,
        width=640,
        data_types=["rgb"], #, "depth"],
        spawn= None
    )

    bird_camera_cfg: CameraCfg = CameraCfg(
        prim_path="/World/envs/env_.*/bridgeData/bird_camera/bird_camera/bird_camera",
        update_period=0.1,
        height=480,
        width=640,
        data_types=["rgb"],
        spawn=None
    )
    

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4, env_spacing=4.0, replicate_physics=False)



class SpotCurtainEnv( DirectRLEnv):
    cfg: SpotCurtainEnvCfg
    def __init__(self,
                 cfg: SpotCurtainEnvCfg,
                 render_mode: str | None = None, **kwargs):

        super().__init__(cfg, render_mode, **kwargs)
        print('=================Spot Curtain Env====================')
        self.num_actions = cfg.action_space
        self.dt = self.cfg.sim.dt * self.cfg.decimation
        self.flag = torch.zeros((self.num_envs), device=self.sim.device)
        self._ee_name = 'arm0_link_ee'

        self.num_actions = cfg.action_space
        self.robot_dof_targets = torch.zeros(
            (self.num_envs, self.robot.num_joints), dtype=torch.float, device=self.sim.device
        )
        self.robot_dof_pos = torch.zeros(
            (self.num_envs, self.robot.num_joints), device=self.sim.device)

        self.dt = self.cfg.sim.dt * self.cfg.decimation
        self.flag = torch.zeros((self.num_envs), device=self.sim.device)

        self.ee_idx = self.robot.find_bodies("arm_ee")[0][0]
        self._ee_name = 'arm0_link_ee'
        self.controller.init_ctrl(self._ee_name,self.robot.body_names,self.robot.joint_names)
        self.disable_vla_mode()

    def enable_vla_mode(self):
        """Enable VLA command transformation"""
        self.vla_mode = True

    def disable_vla_mode(self):
        """Disable VLA command transformation"""
        self.vla_mode = False

    def get_robot_ee_state(self):
        """Get the end-effector state: position, orientation, gripper state"""
        robot_ee_pos = self.arm_ee_pos_w[0].cpu().numpy()
        robot_ee_quat = self.arm_ee_quat_w[0].cpu().numpy()
        
        # Assuming gripper state is represented by the last joint position
        gripper_joint_name = 'arm0_f1x'  # Example gripper joint name
        if gripper_joint_name in self.robot.joint_names:
            gripper_idx = self.robot.joint_names.index(gripper_joint_name)
            robot_gripper = self.robot_dof_pos[0, gripper_idx].cpu().numpy()
        else:
            robot_gripper = None  # Gripper joint not found
        
        return robot_ee_pos, robot_ee_quat, robot_gripper
    
    def _setup_scene(self,) :
        from isaaclab.sim.spawners.from_files import spawn_from_usd
        
        self.robot = Articulation(self.cfg.robot_cfg)     
        
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())

        # Spawn cabinet directly 
        for env_idx in range(self.scene.cfg.num_envs):
            #TODO: Place all objects for BridgeData Tests here also add physics to the object
            bridgeData_prim_path = f"/World/envs/env_{env_idx}/bridgeData"
            root = os.getcwd()
            #table
            table_path=root+'/asset/objects/small_wooden_table.usd'
            table_cfg = sim_utils.UsdFileCfg(usd_path=table_path)
            table_cfg.scale = (0.0002, 0.0002, 0.0002)
            spawn_from_usd(
                prim_path=bridgeData_prim_path+"/table",
                cfg=table_cfg,
                translation=(0.93, 1.41, -0.152),
                orientation=(0.7, 0.7, 0, 0),  # x, y, z, w
                
            )

            

            positions_list = self.generate_points()
            random.shuffle(positions_list)

            #TODO: spawn the banana at a random position on the table

            #banana
            banana_path=root+'/asset/objects/banana.usd'
            banana_cfg = sim_utils.UsdFileCfg(usd_path=banana_path)
            banana_cfg.scale = (0.002, 0.002, 0.002)
            spawn_from_usd(
                prim_path=bridgeData_prim_path+"/banana",
                cfg=banana_cfg,
                translation=positions_list.pop(0),
                orientation=(0, 0, 0, 0),  # x, y, z, w
            )

            if(random.random()>0.3):
                #scissor

                scissor_path=root+'/asset/objects/scissors.usd'
                scissor_cfg = sim_utils.UsdFileCfg(usd_path=scissor_path)
                scissor_cfg.scale = (0.005, 0.005, 0.005)
                spawn_from_usd(
                    prim_path=bridgeData_prim_path+"/scissor",
                    cfg=scissor_cfg,
                    translation=positions_list.pop(0),
                    orientation=(0, 0, 0, 0),  # x, y, z, w
                )

            if(random.random()>0.8):
                pos = positions_list.pop(0)
                #can
                can_path=root+'/asset/objects/soda_can.usd'
                can_cfg = sim_utils.UsdFileCfg(usd_path=can_path)
                can_cfg.scale = (0.0005, 0.0005, 0.0005)
                spawn_from_usd(
                    prim_path=bridgeData_prim_path+"/soda_can",
                    cfg=can_cfg,
                    translation=(pos[0], pos[1], 0.323),
                    orientation=(0, 0, 0, 0),  # x, y, z, w
                )

            if(random.random()>0.4):
                #brush can be used instead ot banana or as clutter
                brush_path=root+'/asset/objects/paint_brush.usd'
                brush_cfg = sim_utils.UsdFileCfg(usd_path=brush_path)
                brush_cfg.scale = (0.005, 0.005, 0.005)
                spawn_from_usd(
                    prim_path=bridgeData_prim_path+"/brush",
                    cfg=brush_cfg,
                    translation=positions_list.pop(0),
                    orientation=(0, 0, 0, 0),  # x, y, z, w
                )
            
            if(random.random()>0.2):
                pos = positions_list.pop(0)
                #pot
                pot_path=root+'/asset/objects/Pot.usd'
                pot_cfg = sim_utils.UsdFileCfg(usd_path=pot_path)
                pot_cfg.scale = (0.07, 0.07, 0.07)
                spawn_from_usd(
                    prim_path=bridgeData_prim_path+"/pot",
                    cfg=pot_cfg,
                    translation=(pos[0]+0.3, pos[1], 0.33),
                    orientation=(0.7, 0.7, 0, 0),  # x, y, z, w
                )
            

            #bird_camera
            bird_camera_path=root+'/asset/objects/bird_camera.usd'
            bird_camera_cfg = sim_utils.UsdFileCfg(usd_path=bird_camera_path)
            spawn_from_usd(
                prim_path=bridgeData_prim_path+"/bird_camera",
                cfg=bird_camera_cfg,
                translation=(1.5, 1.3, 1.6),
                orientation=(0.66, 0.24, 0.24, 0.66),  # x, y, z, w
            )

           
        self._camera = Camera(self.cfg.bird_camera_cfg)

        # clone, filter, and replicate
        self.scene.clone_environments(copy_from_source=False)
        self.scene.filter_collisions(global_prim_paths=[])
        # add articultion to scene
        print(f'robot object check: {self.robot}')
        self.scene.articulations["robot"] = self.robot
        self.scene.sensors["camera"] = self._camera

        try:
            self.cabinet = Articulation("/World/envs/env_.*/Cabinet")
            self.scene.articulations["cabinet"] = self.cabinet
            print("Cabinet added as controllable articulation")
        except Exception as e:
            print(f"Cabinet added as static object: {e}")
            self.cabinet = None

        self.controller = OperationSpaceController(num_robot=self.num_envs,
                                                   device=self.device,
                                                   )

        # Add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0)
        light_cfg.func("/World/Light", light_cfg)

    # Generate a list of 8 (x, y, z) points within the specified limits,
    # ensuring that no two points are closer than 0.15 units apart.
    def generate_points(self, num_points=8, x_range=(0.5, 1.2), y_range=(1.0, 1.8), z_value=0.313, min_dist=0.15):
        points = []
        attempts = 0
        max_attempts = 1000
        while len(points) < num_points and attempts < max_attempts:
            x = random.uniform(*x_range)
            y = random.uniform(*y_range)
            z = z_value
            candidate = (x, y, z)
            # Check distance to all existing points
            if all(math.hypot(x - px, y - py) >= min_dist for px, py, _ in points):
                points.append(candidate)
            attempts += 1
        if len(points) < num_points:
            raise RuntimeError("Could not generate enough non-overlapping points.")
        return points

    def _reset_idx(self, env_ids):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        super()._reset_idx(env_ids)
        num_indices = len(env_ids)

        limit = self.robot.data.joint_pos_limits[:, :, :]
        pos = torch.clamp(self.robot.data.default_joint_pos[env_ids], limit[:, :, 0], limit[:, :, 1])

        dof_pos = torch.zeros((num_indices, self.robot.num_joints), device=self.sim.device)
        dof_vel = torch.zeros((num_indices, self.robot.num_joints), device=self.sim.device)
        dof_pos[:, :] = pos
        default_root_state = self.robot.data.default_root_state[env_ids]
        default_root_state[:, :3] += self.scene.env_origins[env_ids]
        self.robot_dof_targets[env_ids, :] = pos
        self.robot_dof_pos[env_ids, :] = pos

        self.robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self.robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self.robot.write_joint_state_to_sim(dof_pos, dof_vel, None, env_ids)

        if hasattr(self, 'cabinet') and self.cabinet is not None:
            # Reset cabinet position (RigidObject reset)
            cabinet_default_state = self.cabinet.data.default_root_state[env_ids].clone()
            cabinet_default_state[:, :3] += self.scene.env_origins[env_ids]
            self.cabinet.write_root_pose_to_sim(cabinet_default_state[:, :7], env_ids)
            self.cabinet.write_root_velocity_to_sim(cabinet_default_state[:, 7:], env_ids)
    
    ################################################# translation functions 

    def calculate_arm_goal(self, arm_actions):
        """
        Calculate arm goal positions and orientations using OSC-style goal setting
        
        Args:
            arm_actions: [num_envs, 9] tensor containing [base_x, base_y, base_yaw, arm_x, arm_y, arm_z, arm_rx, arm_ry, arm_rz]
        
        Returns:
            goal_pos: [num_envs, 3] target positions
            goal_ori: [num_envs, 3, 3] target orientation matrices
        """
        
        pos_delta = arm_actions[:, 3:6]  # [arm_x, arm_y, arm_z] - relative position delta
        rot_delta = arm_actions[:, 6:9]  # [arm_rx, arm_ry, arm_rz] - relative rotation delta
        rot_delta = rot_delta[0].cpu().numpy() #only one env

        current_pos = self.arm_ee_pos_w  # [num_envs, 3]
        current_quat = self.arm_ee_quat_w[0].cpu().numpy()  # [num_envs, 4] - (w, x, y, z)

        current_rot_mat = quat2mat(current_quat)  # [num_envs, 3, 3]
        
        # Set goal position using OSC utility
        goal_pos = set_goal_position(
            delta=pos_delta,
            current_position=current_pos,
            position_limit=None #TODO set position limits
        )
        
        # Set goal orientation using OSC utility
        # Check if rotation delta is non-zero (similar to OSC's approach)
        bools = [0.0 if math.isclose(elem, 0.0) else 1.0 for elem in rot_delta]
        
        if sum(bools) > 0.0:
            # There's a valid rotation command
            goal_ori = set_goal_orientation(
                delta=rot_delta,  # axis-angle representation
                current_orientation=current_rot_mat,
                orientation_limit=None #TODO set orientation limits
            )
        else:
            # No rotation command, keep current orientation
            goal_ori = current_rot_mat.copy()

        goal_ori_euler = mat2quat(goal_ori)
        
        # Convert back to PyTorch tensors
        goal_pos_tensor = torch.tensor(
            goal_pos, 
            device=current_pos.device, 
            dtype=torch.float32 
        ).unsqueeze(0)  # [1, 3]
    
        goal_ori_tensor = torch.tensor(
            goal_ori_euler, 
            device=current_pos.device, 
            dtype=torch.float32 
        ).unsqueeze(0) # [1, 4]

        return goal_pos_tensor, goal_ori_tensor

    ################################################# cyclic functions 

    def _pre_physics_step(self, actions):
        #actions = [
        #    base_x,    base_y,    base_yaw,     # Base movement (3D)
        #    arm_x,     arm_y,     arm_z,        # Arm position delta (3D)  
        #    arm_rx,    arm_ry,    arm_rz,       # Arm rotation delta (3D)
        #    gripper_open, wrist_rot, wrist_pitch # Gripper/wrist commands (3D)
        #]
        
        robot_pos_w = self.robot.data.root_pos_w
        robot_quat_w = self.robot.data.root_quat_w
        
        try:
            self.camera_pos_w = self._camera.data.pos_w         # [num_envs, 3] - world position
            self.camera_quat_w = self._camera.data.quat_w_world # [num_envs, 4] - (w,x,y,z) world quaternion
        except AttributeError as e:
            print(f"⚠️ Camera pose not available: {e}")
            # Fallback: use robot pose
            self.camera_pos_w = robot_pos_w
            self.camera_quat_w = robot_quat_w

        body_state_w = self.robot.data.body_link_state_w
        self.arm_ee_pos_w = body_state_w[:, self.ee_idx, 0:3]
        self.arm_ee_quat_w = body_state_w[:, self.ee_idx, 3:7]

        lin_vel = self.robot.data.root_lin_vel_b
        ang_vel = self.robot.data.root_ang_vel_b
        gravity_b = self.robot.data.projected_gravity_b
        current_joint_pos = self.robot.data.joint_pos
        current_joint_vel = self.robot.data.joint_vel
        body_state_w = self.robot.data.body_state_w

        if(self.vla_mode):
            actions = actions.clone()
            actions[:, 3:6] = actions[:, 3:6] * 0.2 
            #TODO:scale the action commands properly with maximum output and min input thresholds
            transformed_pos_cmd, transformed_ori_cmd = self.calculate_arm_goal(actions)

            #print(transformed_ori_cmd)#tensor([[ 0.9919, -0.0675, -0.0304,  0.1030]], device='cuda:0')
            #print("type:", type(transformed_ori_cmd))#type: <class 'torch.Tensor'>
            
            

            action,index,success = self.controller.compute(lin_vel, ang_vel,  gravity_b,
                                                    current_joint_pos, current_joint_vel,
                                                    body_state_w,
                                                    actions[:,:3],
                                                    transformed_pos_cmd,
                                                    transformed_ori_cmd) #arm 

        else:
            arm_pos = None
            arm_ori = None
            if actions[:,3:].any()!=0:
                arm_pos = actions[:,3:6]  # arm position (3) + arm rotation (3)
                axis_angle = actions[0:,6:9].cpu().numpy()
                quat_numpy = axisangle2quat(axis_angle)  # Convert Euler to Quaternion
                
                
                arm_ori = torch.tensor(
                    [quat_numpy],  # Wrap in list to create [1, 4] batch dimension
                    device=actions.device, 
                    dtype=torch.float32
                )
                #print(arm_ori)#[0. 0. 0. 1.]
                #print("type:", type(arm_ori))#type: <class 'numpy.ndarray'>

            action,index,success = self.controller.compute(lin_vel, ang_vel,  gravity_b,
                                                    current_joint_pos, current_joint_vel,
                                                    body_state_w,
                                                    actions[:,:3], #base
                                                    arm_pos,
                                                    arm_ori)
        
        
        #TODO:
        #if not all(success):
            # this means IK failed because of unreachable target position. so use arm position command to move robot base instead.

        gripper_comd = None
        #self.robot_dof_targets[:, index] = action
        for joint_action, joint_indices in zip(action, index):
            self.robot_dof_targets[:, joint_indices] = joint_action
            #print("after Kinematic solver, what joints are addressed?")
            #print (f"Joint indices: {joint_indices}") # Joint indices: [4, 9, 14, 15, 16, 17, 18]
            joint_names = [self.robot.joint_names[i] for i in joint_indices]
            #print (f"Joint names: {joint_names}") # Joint names: ['arm0_sh0', 'arm0_sh1', 'arm0_el0', 'arm0_el1', 'arm0_wr0', 'arm0_wr1', 'arm0_f1x']
            #print (f"Joint actions: {joint_action}")   # Joint actions: tensor([[-0.2315, -1.9298,  2.9268,  0.5250, -0.0306, -0.8258, -1.0711]], device='cuda:0')
            if 'arm0_f1x' in joint_names:
                f1x_pos = joint_names.index('arm0_f1x')
                #gripper_comd = joint_action[:, f1x_pos:f1x_pos+1].clone()  # Extract the value at the arm0_f1x position

        if gripper_comd is not None and torch.any(gripper_comd != 0):            
            # Find joint indices for wrist and gripper
            joint_names = self.robot.joint_names
            f1x_idx = joint_names.index('arm0_f1x') if 'arm0_f1x' in joint_names else None
            
            # Apply gripper commands directly
            if f1x_idx is not None:
                current_f1x = current_joint_pos[:, f1x_idx]
                
                # Apply incremental movement
                new_f1x = current_f1x + torch.deg2rad(gripper_comd)
                
                # Clamp to joint limits: -90° to 0°
                self.robot_dof_targets[:, f1x_idx] = torch.clamp(
                    new_f1x, 
                    torch.deg2rad(torch.tensor(-90.0, device=self.sim.device)), 
                    torch.deg2rad(torch.tensor(0.0, device=self.sim.device))
                )

        limit = self.robot.data.joint_pos_limits[:, :, :]
        self.robot_dof_targets = torch.clamp(self.robot_dof_targets, limit[:, :, 0], limit[:, :, 1])

    def _pre_physics_step_original(self, actions):

        self.actions = actions.clone().to(self.sim.device)
        lin_vel = self.robot.data.root_lin_vel_b
        ang_vel = self.robot.data.root_ang_vel_b
        gravity_b = self.robot.data.projected_gravity_b
        current_joint_pos = self.robot.data.joint_pos
        current_joint_vel = self.robot.data.joint_vel
        body_state_w = self.robot.data.body_state_w
        arm_comd =None
        if self.actions[:,3:].any()!=0:
            arm_comd = self.actions[:,3:]

        # do not set the base_pose if arm related to body frame
        action,index,_ = self.controller.compute(lin_vel, ang_vel,  gravity_b,
                                                  current_joint_pos, current_joint_vel,
                                                  body_state_w,  # ,y,x
                                                  self.actions[:,:3],
                                                  arm_comd) #, success
        self.robot_dof_targets[:, index] = action
        limit = self.robot.data.joint_limits[:, :, :]
        self.robot_dof_targets = torch.clamp(self.robot_dof_targets, limit[:, :, 0], limit[:, :, 1])


    def _apply_action(self):
        self.robot.set_joint_position_target(self.robot_dof_targets) # 10 times


    def _get_image_obs(self):
        camera_data = {}
        process = False
        for data_type in self.cfg.bird_camera_cfg.data_types:
            if data_type == "rgb":
                tem_data = self._camera.data.output[data_type].to(torch.uint8)
                if process:
                    encode_feature = self.encoder.extract_dino_features(tem_data)
                    camera_data['dino_feature'] = encode_feature
            elif data_type == "depth":
                tem_data = self._camera.data.output[data_type]
                tem_data[tem_data == float("inf")] = 0
            else:
                tem_data = self._camera.data.output[data_type]

            camera_data[data_type] = tem_data
    
        return camera_data

    def _get_observations(self, ):
        if self.cfg.camera:
            camera_data = self._get_image_obs()
            return camera_data


    def _get_states(self):
        return None

    def compute_rewards(self) -> None:

        rewards = torch.zeros((self.num_envs, 1), device=self.sim.device)

        return rewards

    # know if it finish or not
    def _get_dones(self) -> None:
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        resets = torch.zeros_like(time_out, device=self.sim.device)
        return  time_out,resets

    def _get_rewards(self) -> dict[str, torch.Tensor]:
        total_reward = self.compute_rewards()
        return total_reward
