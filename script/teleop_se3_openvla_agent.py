# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to run keyboard teleoperation with OpenVLA assistance."""

import argparse
import sys
import os
import torch
import numpy as np
from PIL import Image

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from isaaclab.app import AppLauncher

# Add argparse arguments
parser = argparse.ArgumentParser(description="OpenVLA-enhanced keyboard teleoperation.")
parser.add_argument("--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument("--teleop_device", type=str, default="keyboard", help="Device for interacting with environment")
parser.add_argument("--task", type=str, default='MoDe-Spot-Curtain-v0', help="Name of the task.")
parser.add_argument("--sensitivity", type=float, default=1.0, help="Sensitivity factor.")
parser.add_argument("--enable_openvla", action="store_true", default=False, help="Enable OpenVLA assistance")
parser.add_argument("--openvla_prompt", type=str, default="What action should the robot take to find and open the top drawer?", help="OpenVLA prompt")

# Append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)

# Parse arguments
args_cli = parser.parse_args()
args_cli.enable_cameras = True

# Launch omniverse app
app_launcher = AppLauncher(headless=args_cli.headless, enable_cameras=args_cli.enable_cameras)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import torch
import omni.log

from task.Curtain import SpotCurtainEnv 
from controller.se3_keyboard import MMKeyboard
from isaaclab_tasks.utils import parse_env_cfg
from isaaclab.sensors import save_images_to_file

# ✅ ADD: OpenVLA imports (adapted from Spot_In_Scene.py)
if args_cli.enable_openvla:
    try:
        from transformers import AutoModelForVision2Seq, AutoProcessor
        import transformers.utils.import_utils
        import transformers.modeling_utils
        
        # Disable flash attention warnings (from Spot_In_Scene.py)
        transformers.utils.import_utils._flash_attn_available = False
        transformers.modeling_utils.is_flash_attn_available = lambda: False
        
        OPENVLA_AVAILABLE = True
        print("OpenVLA imports successful")
    except ImportError as e:
        print(f"OpenVLA not available: {e}")
        OPENVLA_AVAILABLE = False
else:
    OPENVLA_AVAILABLE = False

class OpenVLAAssistant:
    """OpenVLA integration for Isaac Lab (adapted from Spot_In_Scene.py)"""
    
    def __init__(self, enabled=True):
        self.enabled = enabled and OPENVLA_AVAILABLE
        self.vla = None
        self.processor = None
        self.ready = False
        
        if self.enabled:
            self._load_model()
    
    def _load_model(self):
        """Load OpenVLA model (adapted from Spot_In_Scene.py)"""
        try:
            print("Loading OpenVLA model...")
            
            self.processor = AutoProcessor.from_pretrained("openvla/openvla-7b", trust_remote_code=True)
            
            self.vla = AutoModelForVision2Seq.from_pretrained(
                "openvla/openvla-7b", 
                attn_implementation="flash_attention_2",  
                torch_dtype=torch.bfloat16, 
                low_cpu_mem_usage=True, 
                trust_remote_code=True,
                device_map=None,
            )
            
            self.vla = self.vla.to("cuda:0")
            self.ready = True
            print("✅ OpenVLA loaded successfully")
            
        except Exception as e:
            print(f"Failed to load OpenVLA: {e}")
            self.enabled = False
            self.ready = False
    
    def get_action_suggestion(self, rgb_image, prompt):
        """Get action suggestion from OpenVLA"""
        if not self.ready:
            return None
            
        try:
            # Convert Isaac Lab image format to PIL (adapted format)
            if isinstance(rgb_image, torch.Tensor):
                # Isaac Lab typically returns [H, W, 3] tensors
                if rgb_image.dim() == 4:  # [B, H, W, 3]
                    rgb_image = rgb_image[0]  # Take first batch
                
                # Convert to numpy and ensure proper format
                image_np = rgb_image.cpu().numpy()
                
                # Ensure values are in 0-255 range
                if image_np.max() <= 1.0:
                    image_np = (image_np * 255).astype(np.uint8)
                else:
                    image_np = image_np.astype(np.uint8)
                
                pil_image = Image.fromarray(image_np)
            else:
                pil_image = rgb_image
            
            # Format prompt (adapted from Spot_In_Scene.py)
            formatted_prompt = f"In: {prompt}\nOut:"
            
            # Get prediction
            inputs = self.processor(formatted_prompt, pil_image).to("cuda:0", dtype=torch.bfloat16)
            action = self.vla.predict_action(**inputs, unnorm_key="bridge_orig", do_sample=False)
            
            return action
            
        except Exception as e:
            print(f"OpenVLA prediction error: {e}")
            return None

def main():
    """Enhanced teleoperation with OpenVLA assistance."""
    
    # Initialize OpenVLA
    openvla_assistant = OpenVLAAssistant(enabled=args_cli.enable_openvla)
    
    # Parse configuration
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )

    # Create environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.enable_cameras else None)
    
    print(f"Environment created: {args_cli.task}")
    if "Curtain" in args_cli.task:
        omni.log.warn(f"The environment '{args_cli.task}' does not support gripper control.")

    # Create keyboard controller
    if args_cli.teleop_device.lower() == "keyboard":
        teleop_interface = MMKeyboard(
            arm_pos_sensitivity=args_cli.sensitivity * 0.05,
            arm_rot_sensitivity=args_cli.sensitivity * 0.8,
            base_com_sensitivity=args_cli.sensitivity * 2,
        )
    else:
        raise ValueError(f"Invalid device interface '{args_cli.teleop_device}'. Supported: 'keyboard'.")
    
    # Add teleoperation callbacks
    teleop_interface.add_callback("L", env.reset)
    
    # ✅ ADD: OpenVLA callback
    if openvla_assistant.ready:
        def trigger_openvla():
            print("🤖 Requesting OpenVLA suggestion...")
            return True
        teleop_interface.add_callback("SPACE", trigger_openvla)  # Spacebar for OpenVLA
        print("OpenVLA ready - Press SPACE for AI suggestions")
    
    print(teleop_interface)

    # Reset environment
    env.reset()
    teleop_interface.reset()

    # Initialize actions
    try:
        actions = torch.zeros(env.action_space.shape, dtype=torch.float32, device=args_cli.device)
    except:
        actions = torch.zeros((args_cli.num_envs, 9), dtype=torch.float32, device=args_cli.device)

    # ✅ ADD: OpenVLA state tracking
    openvla_mode = False
    openvla_counter = 0
    suggested_action = None

    # Main simulation loop
    while simulation_app.is_running():
        with torch.inference_mode():
            # Step environment
            obs_dict = env.step(actions)[0]
            obs = obs_dict.get("rgb", None)

            # Get keyboard input
            arm_delta_pose, gripper_command, base_delta_com, finish_flag = teleop_interface.advance()
            
            # ✅ ADD: Check for OpenVLA trigger (adapted from Spot_In_Scene.py pattern)
            if openvla_assistant.ready and obs is not None:
                # Check if spacebar was pressed (you'll need to modify MMKeyboard to support this)
                # For now, we'll use a simple counter approach
                openvla_counter += 1
                
                if openvla_counter % 100 == 0:  # Every 100 steps, get OpenVLA suggestion
                    print("🤖 Getting OpenVLA suggestion...")
                    suggested_action = openvla_assistant.get_action_suggestion(obs, args_cli.openvla_prompt)
                    if suggested_action is not None:
                        print(f"🤖 OpenVLA suggests: {suggested_action}")
                        # You can process this suggestion and convert to robot actions
            
            # Convert teleop commands to actions
            arm_delta_pose = torch.tensor(arm_delta_pose).to(torch.float32).to(device=args_cli.device).reshape(args_cli.num_envs, -1)
            base_delta_com = torch.tensor(base_delta_com).to(torch.float32).to(device=args_cli.device).reshape(args_cli.num_envs, -1)

            # ✅ ADD: Blend manual and AI control
            if suggested_action is not None and openvla_mode:
                # Convert OpenVLA action to robot format (you'll need to implement this conversion)
                # For now, just use manual control
                actions = torch.concat([base_delta_com, arm_delta_pose], dim=1)
            else:
                # Manual control
                actions = torch.concat([base_delta_com, arm_delta_pose], dim=1)

            if finish_flag:
                break

    # Cleanup
    env.close()
    simulation_app.close()

if __name__ == "__main__":
    main()