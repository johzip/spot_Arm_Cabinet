# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to run a keyboard teleoperation with Isaac Lab manipulation environments."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys
import os
import time
import uuid
import json
import imageio
import numpy as np
import atexit

#from omni.isaac.lab.app import AppLauncher
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


from script.dataset_Collector import DROIDStyleDatasetCollector
from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Keyboard teleoperation for DeMoMTasksuite.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument("--teleop_device", type=str, default="keyboard", help="Device for interacting with environment")
parser.add_argument("--task", type=str, default='MoDe-Spot-Curtain-v0', help="Name of the task.")
parser.add_argument("--sensitivity", type=float, default=1.0, help="Sensitivity factor.")
#parser.add_argument("--enable_cameras", type=float, default=1.0, help="Sensitivity factor.")

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
from isaaclab.sensors import save_images_to_file
#from omni.isaac.lab_tasks.utils import parse_env_cfg
#from omni.isaac.lab.sensors import save_images_to_file


# Generate a unique folder name for this execution
EXECUTION_ID = str(uuid.uuid4())[:8]
OUT_DIR = os.path.join("out", f"run_{EXECUTION_ID}")
os.makedirs(OUT_DIR, exist_ok=True)

def main():
    """Running keyboard teleoperation with Isaac Lab manipulation environment."""
     # Initialize dataset collector
    dataset_collector = DROIDStyleDatasetCollector(save_dir=OUT_DIR)

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
    
    # Add dataset collection callbacks
    dataset_collector.start_episode(
        language_instruction="Pick up object",
    )
    env.reset()
    
    
    teleop_interface.add_callback("L", dataset_collector.end_episode)
    print(teleop_interface)

    # reset environment
    env.reset()

    teleop_interface.reset()

    #actions = torch.zeros_like(env.actions)
    print(f"env.action_space.shape: {env.action_space.shape}")
    actions = torch.zeros(env.action_space.shape, dtype=torch.float32, device=args_cli.device)

    step_counter = 0

    # simulate environment
    while simulation_app.is_running():
        #TODO fix the pause logic currently I wish to wait for an event from teleop_interface
        #while not teleop_interface.event: #endless loop until event occurs
        #    time.sleep(0.01)
        #    try:
        #        simulation_app.update()
        #    except:
        #        pass

        # run everything in inference mode
        with torch.inference_mode():
            obs_dict = env.step(actions)[0]

            arm_delta_pose, gripper_command, base_delta_com, finish_flag = teleop_interface.advance()

            arm_delta_pose = torch.tensor(arm_delta_pose).to(torch.float).to(device=args_cli.device).reshape(args_cli.num_envs,-1)
            base_delta_com = torch.tensor(base_delta_com).to(torch.float).to(device=args_cli.device).reshape(args_cli.num_envs,-1)
            
            gripper_actions = torch.tensor(gripper_command).to(torch.float).to(device=args_cli.device).reshape(args_cli.num_envs, -1)
        
            
            actions= torch.concat([base_delta_com, arm_delta_pose, gripper_actions], dim=1)
            
            # Collect dataset step if episode is active
            if dataset_collector.current_episode is not None:
                # Get robot state from environment
                robot_state = {
                    "ee_pos": env.unwrapped.arm_ee_pos_w[0].cpu().numpy() if hasattr(env.unwrapped, 'arm_ee_pos_w') else None,
                    "ee_quat": env.unwrapped.arm_ee_quat_w[0].cpu().numpy() if hasattr(env.unwrapped, 'arm_ee_quat_w') else None,
                    "joint_pos": env.unwrapped.robot.data.joint_pos[0].cpu().numpy() if hasattr(env.unwrapped, 'robot') else None,
                    # TODO: Add more robot state fields as needed
                }
                
                # Check if this should be terminal step (e.g., if finish_flag or specific key pressed)
                is_terminal = finish_flag  # or some other condition
                
                dataset_collector.add_step(
                    obs_dict=obs_dict,
                    action=actions[0],  # Single environment action
                    robot_state=robot_state,
                    step_idx=step_counter,
                    is_terminal=is_terminal
                )
                
                step_counter += 1
                
                if is_terminal:
                    dataset_collector.end_episode()
                    step_counter = 0
                    print("🏁 Episode auto-ended due to finish flag")

            if finish_flag:
                env.close()
                simulation_app.close()


    # close the simulator
    env.close()



if __name__ == "__main__":

    main()
    # close sim app
    simulation_app.close()

