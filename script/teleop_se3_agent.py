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
import requests
from PIL import Image
import io


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
parser.add_argument("--openvla_prompt", type=str, default="Put Banana into pot", help="OpenVLA prompt")


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
from isaaclab_tasks.utils import parse_env_cfg
import time
import cv2



def send_to_openvla(image_np, prompt, server_url="http://localhost:8000/predict"):
    # Convert numpy image to PNG bytes
    pil_image = Image.fromarray(image_np).convert("RGB")
    img_bytes = io.BytesIO()
    pil_image.save(img_bytes, format="PNG")
    img_bytes.seek(0)
    files = {"image": ("image.png", img_bytes, "image/png")}
    data = {"prompt": prompt}
    response = requests.post(server_url, files=files, data=data)
    if response.ok:
        result = response.json()
        if "action" in result:
            return result["action"]
        else:
            print("❌ 'action' key not found in response:", result)
            return None
    else:
        print("❌ OpenVLA REST API error:", response.text)
        return None


def main():
    """Running keyboard teleoperation with Isaac Lab manipulation environment using OpenVLA to controll the robot."""
    if args_cli.enable_openvla:
        vlaMode = True
    else:
        vlaMode = False

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

            if vlaMode and obs is not None:

                env.unwrapped.enable_vla_mode()
                image_np = obs[0].cpu().numpy() if obs.dim() == 4 else obs.cpu().numpy()
                suggested_action = send_to_openvla(image_np, args_cli.openvla_prompt)
            else:
                env.unwrapped.disable_vla_mode()
                

            arm_delta_pose = torch.tensor(arm_delta_pose).to(torch.float).to(device=args_cli.device).reshape(args_cli.num_envs,-1)
            base_delta_com = torch.tensor(base_delta_com).to(torch.float).to(device=args_cli.device).reshape(args_cli.num_envs,-1)
            
            gripper_actions = torch.tensor(gripper_command).to(torch.float).to(device=args_cli.device).reshape(args_cli.num_envs, -1)
            
            
            if vlaMode and suggested_action is not None:
                
                #suggested_action = np.array(suggested_action) * 2.5
                openvla_x, openvla_y, openvla_z = suggested_action[0], suggested_action[1], suggested_action[2]
                openvla_roll, openvla_pitch, openvla_yaw = suggested_action[3], suggested_action[4], suggested_action[5]
                openvla_gripper = suggested_action[6]

                
                # Arm: use OpenVLA's x, y, z for position + zero rotation (or manual rotation)
                ai_arm_delta = torch.tensor([
                    [openvla_x, openvla_y, openvla_z, openvla_roll, openvla_pitch, openvla_yaw]  # position from AI, rotation from manual
                ], device=args_cli.device).repeat(args_cli.num_envs, 1)
                
                # Gripper: use OpenVLA's roll, yaw, gripper (skip pitch)
                ai_gripper_actions = torch.tensor([
                    [openvla_gripper, openvla_roll, openvla_yaw] 
                ], device=args_cli.device).repeat(args_cli.num_envs, 1)

                # Combine AI actions
                actions = torch.concat([base_delta_com, ai_arm_delta, ai_gripper_actions], dim=1)
                #print(f"suggested_action: {suggested_action}")
                print(f"🤖 Using AI control: base={base_delta_com[0].tolist()}, arm={ai_arm_delta[0].tolist()}, gripper={ai_gripper_actions[0].tolist()}")
            else:
                # Manual control
                actions = torch.concat([base_delta_com, arm_delta_pose, gripper_actions], dim=1)


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

