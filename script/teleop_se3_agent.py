# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to run a keyboard teleoperation with Isaac Lab manipulation environments with OpenVLA assistance."""
"""Launch Isaac Sim Simulator first."""

import argparse
import sys
import os
import numpy as np



#from omni.isaac.lab.app import AppLauncher
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="OpenVLA-enhanced keyboard teleoperation.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument("--teleop_device", type=str, default="keyboard", help="Device for interacting with environment")
parser.add_argument("--task", type=str, default='MoDe-Spot-Curtain-v0', help="Name of the task.")
parser.add_argument("--sensitivity", type=float, default=1.0, help="Sensitivity factor.")
parser.add_argument("--enable_openvla", action="store_true", default=False, help="Enable OpenVLA assistance")
parser.add_argument("--openvla_prompt", type=str, default="What action should the robot take to find and open the top drawer?", help="OpenVLA prompt")


# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)

# parse the arguments
args_cli = parser.parse_args()
args_cli.enable_cameras = True
# launch omniverse app
app_launcher = AppLauncher(headless=args_cli.headless, enable_cameras=args_cli.enable_cameras)
simulation_app = app_launcher.app

"""Rest everything follows."""


import gymnasium as gym
import torch
import omni.log

from task.Curtain import SpotCurtainEnv 
from controller import se3_keyboard, spot_operational_space, spot_loco_solver, spot_kinematics_solver
from controller.se3_keyboard import MMKeyboard
from script.openVLAAssistant import OpenVLAAssistant
from isaaclab_tasks.utils import parse_env_cfg
import time
import cv2


def main():
    """Running keyboard teleoperation with Isaac Lab manipulation environment using OpenVLA to controll the robot."""
    if args_cli.enable_openvla:
        openvla_assistant = OpenVLAAssistant(enabled=args_cli.enable_openvla)

    # parse configuration
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )

    # create environment
    env = gym.make(args_cli.task, cfg=env_cfg,render_mode="rgb_array" if args_cli.enable_cameras else None)

    # create controller
    #lower()converts all uppercase characters in a string to their lowercase equivalents
    if args_cli.teleop_device.lower() == "keyboard":
        teleop_interface = MMKeyboard(
            arm_pos_sensitivity = args_cli.sensitivity * 0.05,
            arm_rot_sensitivity = args_cli.sensitivity * 0.8,
            base_com_sensitivity = args_cli.sensitivity*2,
        )
    else:
        raise ValueError(f"Invalid device interface '{args_cli.teleop_device}'. Supported: 'keyboard'.")
    
    # add teleoperation key for env reset
    teleop_interface.add_callback("L", env.reset)

    # print helper for keyboard
    print(teleop_interface)

    # reset environment
    env.reset()

    teleop_interface.reset()
    actions = torch.zeros(env.action_space.shape, dtype=torch.float32, device=args_cli.device)

    openvla_counter = 0
    suggested_action = None

    # simulate environment
    while simulation_app.is_running():
        with torch.inference_mode():
            
            #if torch.any(actions != 0.0):
            #    print(f"Action: {actions}")
            obs_dict = env.step(actions)[0] 
            obs = obs_dict["rgb"]

            #safeObsImageToFile(obs)

            arm_delta_pose, gripper_command, base_delta_com, finish_flag = teleop_interface.advance()

            if openvla_assistant.ready and obs is not None:
                openvla_counter += 1
                if openvla_counter % 100 == 0:  # Every 100 steps, get OpenVLA suggestion
                    print("🤖 Getting OpenVLA suggestion...")
                    suggested_action = openvla_assistant.get_action_suggestion(obs, args_cli.openvla_prompt)
                    if suggested_action is not None:
                        print(f"🤖 OpenVLA suggests: {suggested_action}")

            arm_delta_pose = torch.tensor(arm_delta_pose).to(torch.float).to(device=args_cli.device).reshape(args_cli.num_envs,-1)
            base_delta_com = torch.tensor(base_delta_com).to(torch.float).to(device=args_cli.device).reshape(args_cli.num_envs,-1)
            
            gripper_actions = torch.tensor(gripper_command).to(torch.float).to(device=args_cli.device).reshape(args_cli.num_envs, -1)
        
            
            #actions= torch.concat([base_delta_com, arm_delta_pose, gripper_actions], dim=1)
            
            if suggested_action is not None:
                # Convert OpenVLA action to robot format (you'll need to implement this conversion)
                # 7-DoF end-effector deltas of the form (x,             y,          z,              roll,       pitch,      yaw,      gripper )
                #Example suggested_action fromOpenVLA: [-0.00020879, -0.00042412,  0.00703386,  0.00049971, -0.00747924, -0.00167851,   0.    ]
                
                #TODO: if x,y,z is in arm reach then perform arm movement only (this is the neccesary part)
                #TODO: if x,y,z is out of arm reach then perform base movement instead of arm movement (advanced)
                #TODO: translate  roll, pitch, yaw, in wrist wr0 and wr1 movement OR implement self._delta_arm_rot = np.zeros(3)  # (roll, pitch, yaw) usage instead
                #TODO: translate gripper value into gripper commands

                openvla_x, openvla_y, openvla_z = suggested_action[0], suggested_action[1], suggested_action[2]
                openvla_roll, openvla_pitch, openvla_yaw = suggested_action[3], suggested_action[4], suggested_action[5]
                openvla_gripper = suggested_action[6]

                # Create robot action components
                # Base: always zero (no base movement)
                ai_base_delta = torch.zeros(args_cli.num_envs, 3, device=args_cli.device)
                
                # Arm: use OpenVLA's x, y, z for position + zero rotation (or manual rotation)
                ai_arm_delta = torch.tensor([
                    [openvla_x, openvla_y, openvla_z, 0.0, 0.0, 0.0]  # position from AI, rotation from manual
                ], device=args_cli.device).repeat(args_cli.num_envs, 1)
                
                # Gripper: use OpenVLA's roll, yaw, gripper (skip pitch)
                ai_gripper_actions = torch.tensor([
                    [openvla_gripper, openvla_roll, openvla_yaw]  # gripper, wrist_rot, wrist_pitch
                ], device=args_cli.device).repeat(args_cli.num_envs, 1)
                
                # Combine AI actions
                actions = torch.concat([ai_base_delta, ai_arm_delta, ai_gripper_actions], dim=1)
                print(f"🤖 Using AI control: base=[0,0,0], arm=[{openvla_x:.4f},{openvla_y:.4f},{openvla_z:.4f}], gripper=[{openvla_gripper:.4f},{openvla_roll:.4f},{openvla_yaw:.4f}]")
    
            else:
                # Manual control
                actions = torch.concat([base_delta_com, arm_delta_pose, gripper_actions], dim=1)

            
            #actions= torch.concat([base_delta_com, arm_delta_pose], dim=1)

            if finish_flag:
                env.close()
                simulation_app.close()


    # close the simulator
    env.close()

def safeObsImageToFile(obs):
    if obs is not None:
        timestamp = int(time.time() * 1000)
        filename = f"out/robot_camera_{timestamp}"
        os.makedirs("out", exist_ok=True)
        try:
            # Convert to numpy
            if obs.dim() == 4:
                img_np = obs[0].cpu().numpy()
            else:
                img_np = obs.cpu().numpy()
                    
            # Convert to uint8
            if img_np.dtype != np.uint8:
                if img_np.max() <= 1.0:
                    img_np = (img_np * 255).astype(np.uint8)
                else:
                    img_np = np.clip(img_np, 0, 255).astype(np.uint8)
                    
            # Convert RGB to BGR for OpenCV
            img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

            cv2.imwrite(f"{filename}.png", img_bgr)
            print(f"✅ Saved with OpenCV: {filename}.png")
                    
        except Exception as error:
            print(f"❌ save Image failed: {error}")


if __name__ == "__main__":

    main()
    # close sim app
    simulation_app.close()

