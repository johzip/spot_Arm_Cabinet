# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to run a keyboard teleoperation with Isaac Lab manipulation environments."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys
import os

#from omni.isaac.lab.app import AppLauncher
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

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



def main():
    """Running keyboard teleoperation with Isaac Lab manipulation environment."""
    # parse configuration
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )

    # create environment
    env = gym.make(args_cli.task, cfg=env_cfg,render_mode="rgb_array" if args_cli.enable_cameras else None)
    # check environment name (for reach , we don't allow the gripper)
    print(f" before Curtain? ")
    if "Curtain" in args_cli.task:
        omni.log.warn(
            f"The environment '{args_cli.task}' does not support gripper control. The device command will be ignored."
        )

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

    #actions = torch.zeros_like(env.actions)
    actions = torch.zeros(env.action_space.shape, dtype=torch.float32, device=args_cli.device)

    # simulate environment
    while simulation_app.is_running():
        # run everything in inference mode
        with torch.inference_mode():
            obs_dict = env.step(actions)[0]
            obs = obs_dict["rgb"]

            arm_delta_pose, gripper_command, base_delta_com,finish_flag = teleop_interface.advance()
            arm_delta_pose = torch.tensor(arm_delta_pose).to(torch.float).to(device=args_cli.device).reshape(args_cli.num_envs,-1)
            base_delta_com = torch.tensor(base_delta_com).to(torch.float).to(device=args_cli.device).reshape(args_cli.num_envs,-1)

            # pre-process actions
            actions= torch.concat([base_delta_com,arm_delta_pose], dim=1)

            if finish_flag:
                env.close()
                simulation_app.close()


    # close the simulator
    env.close()


if __name__ == "__main__":

    main()
    # close sim app
    simulation_app.close()

