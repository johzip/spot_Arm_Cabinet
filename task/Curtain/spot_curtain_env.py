from __future__ import annotations

import torch
import os
import math


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
    robot_cfg.init_state.pos = (-0.2, 1.55, 0.4)
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
        self.actions = torch.zeros((self.num_envs, self.num_actions), device=self.sim.device)

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

    def _setup_scene(self,) :
        from isaaclab.sim.spawners.from_files import spawn_from_usd
        
        self.robot = Articulation(self.cfg.robot_cfg)        
        self._camera = Camera(self.cfg.wrist_camera_cfg)
        
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

            #brush can be used instead ot banana or as clutter
            #brush_path=root+'/asset/objects/paint_brush.usd'
            #brush_cfg = sim_utils.UsdFileCfg(usd_path=brush_path)
            #brush_cfg.scale = (0.005, 0.005, 0.005)
            #spawn_from_usd(
            #    prim_path=bridgeData_prim_path+"/brush",
            #    cfg=brush_cfg,
            #    translation=(0.55, 1.17, 0.313),
            #    orientation=(0, 0, 0, 0),  # x, y, z, w
            #)

            #banana
            banana_path=root+'/asset/objects/banana.usd'
            banana_cfg = sim_utils.UsdFileCfg(usd_path=banana_path)
            banana_cfg.scale = (0.002, 0.002, 0.002)
            spawn_from_usd(
                prim_path=bridgeData_prim_path+"/banana",
                cfg=banana_cfg,
                translation=(0.67, 1.08, 0.313),
                orientation=(0, 0, 0, 0),  # x, y, z, w
            )

            #pot
            pot_path=root+'/asset/objects/Pot.usd'
            pot_cfg = sim_utils.UsdFileCfg(usd_path=pot_path)
            pot_cfg.scale = (0.07, 0.07, 0.07)
            spawn_from_usd(
                prim_path=bridgeData_prim_path+"/pot",
                cfg=pot_cfg,
                translation=(0.98, 1.5, 0.33),
                orientation=(0.7, 0.7, 0, 0),  # x, y, z, w
            )

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

    def frameRotation_matrix(self, quat):
        """
        Convert quaternion (w, x, y, z) to 3x3 rotation matrix
        
        Args:
            quat: torch.Tensor of shape [num_envs, 4] - quaternion in (w, x, y, z) format
            
        Returns:
            R: torch.Tensor of shape [num_envs, 3, 3] - rotation matrix
        """
        # Normalize quaternion to ensure unit quaternion
        quat = F.normalize(quat, dim=-1)
        
        # Extract quaternion components (w, x, y, z)
        w, x, y, z = quat[..., 0], quat[..., 1], quat[..., 2], quat[..., 3]
        
        # Compute rotation matrix elements
        # Using standard quaternion to rotation matrix formula
        xx, yy, zz = x*x, y*y, z*z
        xy, xz, yz = x*y, x*z, y*z
        wx, wy, wz = w*x, w*y, w*z
        
        # Build rotation matrix [num_envs, 3, 3]
        R = torch.stack([
            torch.stack([1-2*(yy+zz), 2*(xy-wz), 2*(xz+wy)], dim=-1),
            torch.stack([2*(xy+wz), 1-2*(xx+zz), 2*(yz-wx)], dim=-1),
            torch.stack([2*(xz-wy), 2*(yz+wx), 1-2*(xx+yy)], dim=-1)
        ], dim=-2)
        
        return R
    
    def apply_vla_command_with_transformation(self, actions):
        """
        Transform VLA command from camera frame to gripper frame using homogeneous transformations
        
        Args:
            actions: [num_envs, 12] - robot actions containing VLA command in format:
            [base_x, base_y, base_yaw, arm_x, arm_y, arm_z, arm_rx, arm_ry, arm_rz, gripper_open, wrist_rot, wrist_pitch]
        
        Returns:
            transformed_arm_command: [num_envs, 6] - [x, y, z, rx, ry, rz] in gripper frame
            gripper_command: [num_envs, 3] - [gripper, wrist_rot, wrist_pitch]
        """
        
        # Extract VLA command from actions (target position/rotation in camera frame)
        base_pos = actions[:, 0:3]     # [base_x, base_y, base_yaw] - not used for VLA
        arm_pos = actions[:, 3:6]      # [arm_x, arm_y, arm_z] - target position in camera frame (Xc, Yc, Zc)
        arm_rot = actions[:, 6:9]      # [arm_rx, arm_ry, arm_rz] - target rotation in camera frame
        gripper_cmd = actions[:, 9:12] # [gripper_open, wrist_rot, wrist_pitch]

        # VLA command represents target position in camera frame
        target_pos_camera = arm_pos.float()  # [num_envs, 3] - (Xc, Yc, Zc)
        target_rot_camera = arm_rot.float()  # [num_envs, 3] - rotation in camera frame
        
        # Get current poses (both are in world frame with (w, x, y, z) quaternion format)
        camera_pos_w = self.camera_pos_w.float()    # [num_envs, 3] - tcw (camera position in world)
        camera_quat_w = self.camera_quat_w.float()  # [num_envs, 4] - (w, x, y, z) camera orientation in world
        gripper_pos_w = self.arm_ee_pos_w.float()   # [num_envs, 3] - tgw (gripper position in world)
        gripper_quat_w = self.arm_ee_quat_w.float() # [num_envs, 4] - (w, x, y, z) gripper orientation in world
        
        # Step 1: Build transformation matrices
        # Rcw - camera to world rotation matrix
        R_cw = self.frameRotation_matrix(camera_quat_w)    # [num_envs, 3, 3]
        
        # Rgw - gripper to world rotation matrix  
        R_gw = self.frameRotation_matrix(gripper_quat_w)   # [num_envs, 3, 3]
        
        # Step 2: Transform target position from camera frame to world frame
        # Homogeneous transformation: (Xw, Yw, Zw, 1) = Hcw * (Xc, Yc, Zc, 1)
        # Which simplifies to: (Xw, Yw, Zw) = Rcw * (Xc, Yc, Zc) + tcw
        target_pos_world = torch.bmm(R_cw, target_pos_camera.unsqueeze(-1)).squeeze(-1) + camera_pos_w
        
        # Step 3: Transform target position from world frame to gripper frame
        # Homogeneous transformation: (Xg, Yg, Zg, 1) = Hwg * (Xw, Yw, Zw, 1)
        # Where Hwg = [Rwg, twg; 0, 1] and Rwg = Rgw^T (inverse rotation)
        R_wg = R_gw.transpose(-2, -1)  # World to gripper rotation (Rwg = Rgw^T)
        
        # Apply transformation: (Xg, Yg, Zg) = Rwg * (Xw - tgw)
        #target_pos_gripper = torch.bmm(R_wg, (target_pos_world - gripper_pos_w).unsqueeze(-1)).squeeze(-1)
        
        # Step 4: Transform target rotation from camera frame to gripper frame
        # For rotation vectors, apply same rotational transformations
        # Camera frame to world frame
        target_rot_world = torch.bmm(R_cw, target_rot_camera.unsqueeze(-1)).squeeze(-1)
        
        # World frame to gripper frame
        target_rot_gripper = torch.bmm(R_wg, target_rot_world.unsqueeze(-1)).squeeze(-1)
        
        # Step 5: Convert target position in gripper frame to arm command (delta)
        # The target position in gripper frame represents where we want to move relative to current gripper
        # Apply scaling factors since VLA commands are often very small
        position_scale = 1  # Scale VLA position commands (adjust based on your VLA model)
        rotation_scale = 1  # Scale VLA rotation commands (adjust based on your VLA model)
        
        arm_position_delta = target_pos_world * position_scale
        arm_rotation_delta = target_rot_gripper * rotation_scale
        
        # Step 6: Create output commands
        # Combine position and rotation deltas for arm command
        transformed_arm_command = torch.cat([arm_position_delta, arm_rotation_delta], dim=1)  # [num_envs, 6]
        
        # Use original gripper commands (no transformation needed)
        gripper_command = gripper_cmd  # [num_envs, 3] - [gripper_open, wrist_rot, wrist_pitch]
        
        return transformed_arm_command, gripper_command

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

        if(self.vla_mode):
            transformed_arm_cmd, transformed_gripper_cmd = self.apply_vla_command_with_transformation(actions)
            # Use transformed commands instead of manual control
            
            actions = torch.cat([actions[:, :3], transformed_arm_cmd, transformed_gripper_cmd], dim=1)


        self.actions = actions.clone().to(self.sim.device)
        lin_vel = self.robot.data.root_lin_vel_b
        ang_vel = self.robot.data.root_ang_vel_b
        gravity_b = self.robot.data.projected_gravity_b
        current_joint_pos = self.robot.data.joint_pos
        current_joint_vel = self.robot.data.joint_vel
        body_state_w = self.robot.data.body_state_w
        arm_comd = None
        gripper_comd = None
        if self.actions[:,3:].any()!=0:
            arm_comd = self.actions[:,3:9]  # arm position (3) + arm rotation (3)
            gripper_comd = self.actions[:,9:]  # gripper_open + wrist_rot + wrist_pitch

        # do not set the base_pose if arm related to body frame
        action,index,success = self.controller.compute(lin_vel, ang_vel,  gravity_b,
                                                  current_joint_pos, current_joint_vel,
                                                  body_state_w,
                                                  self.actions[:,:3],
                                                  arm_comd) #arm 
        
        #self.robot_dof_targets[:, index] = action
        for joint_action, joint_indices in zip(action, index):
            self.robot_dof_targets[:, joint_indices] = joint_action
    

        if gripper_comd is not None and torch.any(gripper_comd != 0):            
            # Find joint indices for wrist and gripper
            joint_names = self.robot.joint_names
            wr0_idx = joint_names.index('arm0_wr0') if 'arm0_wr0' in joint_names else None
            wr1_idx = joint_names.index('arm0_wr1') if 'arm0_wr1' in joint_names else None  
            f1x_idx = joint_names.index('arm0_f1x') if 'arm0_f1x' in joint_names else None
            
            #TODO: remove sensitivety scaling HERE
            # Apply gripper commands directly
            if f1x_idx is not None:
                gripper_step = gripper_comd[:, 0] * 5.0  # Increased to 5° per step
                current_f1x = current_joint_pos[:, f1x_idx]
                
                # Apply incremental movement
                new_f1x = current_f1x + torch.deg2rad(gripper_step)
                
                # Clamp to joint limits: -90° to 0°
                self.robot_dof_targets[:, f1x_idx] = torch.clamp(
                    new_f1x, 
                    torch.deg2rad(torch.tensor(-90.0, device=self.sim.device)), 
                    torch.deg2rad(torch.tensor(0.0, device=self.sim.device))
                )
                
            if wr0_idx is not None and gripper_comd.shape[1] > 1:
                wrist_rotation_step = gripper_comd[:, 1] * 3.0  # Tripled from 1.0 to 3.0
                current_wr0 = current_joint_pos[:, wr0_idx]
                
                # Apply incremental movement
                new_wr0 = current_wr0 + torch.deg2rad(wrist_rotation_step)
                
                # Clamp to joint limits: -105° to 105°
                self.robot_dof_targets[:, wr0_idx] = torch.clamp(
                    new_wr0,
                    torch.deg2rad(torch.tensor(-105.0, device=self.sim.device)),
                    torch.deg2rad(torch.tensor(105.0, device=self.sim.device))
                )
                
            if wr1_idx is not None and gripper_comd.shape[1] > 2:
                wrist_pitch_step = gripper_comd[:, 2] * 3.0  # Reduced step size
                current_wr1 = current_joint_pos[:, wr1_idx]
                
                # Apply incremental movement
                new_wr1 = current_wr1 + torch.deg2rad(wrist_pitch_step)
                
                # Clamp to joint limits: -165° to 165°
                self.robot_dof_targets[:, wr1_idx] = torch.clamp(
                    new_wr1,
                    torch.deg2rad(torch.tensor(-165.0, device=self.sim.device)),
                    torch.deg2rad(torch.tensor(165.0, device=self.sim.device))
                )

        limit = self.robot.data.joint_pos_limits[:, :, :]
        self.robot_dof_targets = torch.clamp(self.robot_dof_targets, limit[:, :, 0], limit[:, :, 1])

    def _apply_action(self):
        self.robot.set_joint_position_target(self.robot_dof_targets) # 10 times


    def _get_image_obs(self):
        camera_data = {}
        process = False
        for data_type in self.cfg.wrist_camera_cfg.data_types:
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
