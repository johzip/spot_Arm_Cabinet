# Copyright (c) 2021-2023, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#
from operator import index
from typing import Optional, Tuple
import os
import carb
import numpy as np
import isaaclab.utils.string as string_utils
import torch
from isaacsim.core.utils.extensions import enable_extension


enable_extension("omni.isaac.motion_generation")
#from omni.isaac.motion_generation import LulaKinematicsSolver
from controller.spot_kinematics_solver import ArticulationKinematicsSolver
from controller.spot_loco_solver import LocomotionController
class OperationSpaceController:
    """ Operation-space controller.
    """
    def __init__(
            self,
            num_robot,
            device,
            #end_effector_frame_name
    ):
        """Initialize operation-space controller.

        Args:
            num_envs: The number of robots to control.
            end_effector_frame_name: The name of end effector
            device: The device to use for computations.

        Raises:
            ValueError: When invalid control command is provided.
        """

        self.arm_ctrl = ArticulationKinematicsSolver()
        self.base_ctrl = LocomotionController(num_envs=num_robot,device=device)
        self.num_robot = num_robot
        self.device = device
        return

    def init_ctrl(self,ee_names,body_names,joint_names):
        self.joint_names = joint_names
        if isinstance(ee_names, str):
            ee_names = [ee_names] * self.num_robot
        self.arm_ctrl.intial_multi_solver(ee_names)
        self.ee_idx = string_utils.resolve_matching_names("arm_ee", body_names, )[0][0]#self.robot.find_bodies("arm_ee")[0][0] #arm_fngr
        self.base_idx = string_utils.resolve_matching_names("body", body_names, )[0][0] #self.robot.find_bodies('body')[0][0]  # arm_sh0
        base_body_name = ['fl_hx', 'fr_hx', 'hl_hx', 'hr_hx', 'fl_hy', 'fr_hy', 'hl_hy', 'hr_hy', 'fl_kn', 'fr_kn',
                     'hl_kn', 'hr_kn']
        self.base_idxs = string_utils.resolve_matching_names(base_body_name, joint_names, True)[0] #[self.robot.find_joints(name)[0][0] for name in body_name]
        self.arm_idxs = string_utils.resolve_matching_names('arm0_.*', joint_names, )[0]#self.robot.find_joints('arm0_.*')[0]
        

    def compute(self,
                root_lin_vel_b,
                root_ang_vel_b,
                gravity_b,
                current_joint_pos,
                current_joint_vel,
                body_state_w,
                base_command,
                arm_pos_command = None,
                arm_ori_command = None):

        joint_actions = []
        joint_indices = []
        success = [False] * self.num_robot
        
        if base_command is not None:
            current_base_joints_pos = current_joint_pos[:,self.base_idxs]
            current_base_joints_vel = current_joint_vel[:, self.base_idxs]
            base_joint_act = self.base_ctrl.compute_action(root_lin_vel_b, root_ang_vel_b,  gravity_b,
                                                             current_base_joints_pos, current_base_joints_vel,
                                                             base_command)
            joint_actions.append(base_joint_act)
            joint_indices.append(self.base_idxs)
            success = [True] * self.num_robot
        
        if arm_pos_command is not None or arm_ori_command is not None:
            current_arm_joints_pos = current_joint_pos[:, self.arm_idxs]
            ee_pose_w = body_state_w[:, self.ee_idx, 0:7].cpu().numpy()
            robot_base_pose = body_state_w[:, self.base_idx, 0:7].cpu().numpy()

            self.arm_ctrl.set_robot_base_pose(robot_base_pose[:,:3], robot_base_pose[:,3:])
            current_arm_joints_pos = current_arm_joints_pos.cpu().numpy()
        
            target_pos = arm_pos_command.cpu().numpy() if arm_pos_command is not None else None
            target_ori = arm_ori_command.cpu().numpy() if arm_ori_command is not None else None
            
            arm_joint_act, arm_success = self.arm_ctrl.compute_inverse_kinematics(
                warm_start = current_arm_joints_pos,
                target_position = target_pos,
                target_orientation = target_ori,
            )
            
            arm_joint_act = torch.from_numpy(arm_joint_act).to(self.device, torch.float32)
            joint_actions.append(arm_joint_act)
            joint_indices.append(self.arm_idxs)
            success = arm_success

        return joint_actions, joint_indices, success

