import os
import json
import numpy as np
import torch
import tensorflow as tf
from typing import Dict, Any, List, Optional
from datetime import datetime
import uuid

class DROIDStyleDatasetCollector:
    def __init__(self, save_dir: str = "collected_data"):
        """
        Initialize dataset collector for DROID-style dataset
        
        Args:
            save_dir: Directory to save collected episodes
        """
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
        # Current episode data
        self.current_episode = None
        self.episode_steps = []
        self.episode_id = None
        self.recording_start_time = None
        
        print(f"📁 Dataset collector initialized. Saving to: {save_dir}")
    
    def start_episode(self, language_instruction: str, 
                     language_instruction_2: str = "", 
                     language_instruction_3: str = ""):
        """Start collecting a new episode"""
        print("🚀 Starting new episode collection...")
        
        self.episode_id = f"episode_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{str(uuid.uuid4())[:8]}"
        self.recording_start_time = datetime.now()
        self.episode_steps = []
        
        # Create episode folder
        self.episode_folder = os.path.join(self.save_dir, self.episode_id)
        os.makedirs(self.episode_folder, exist_ok=True)
        os.makedirs(os.path.join(self.episode_folder, "images"), exist_ok=True)
        
        self.current_episode = {
            "episode_metadata": {
                "recording_folderpath": self.episode_folder,
                "file_path": os.path.join(self.episode_folder, "episode_data.json"),
                "episode_id": self.episode_id,
                "start_time": self.recording_start_time.isoformat(),
            },
            "language_instructions": {
                "language_instruction": language_instruction,
                "language_instruction_2": language_instruction_2,
                "language_instruction_3": language_instruction_3,
            }
        }
        
        print(f"🎬 Started episode: {self.episode_id}")
        print(f"📝 Instruction: {language_instruction}")
    
    def add_step(self, 
                obs_dict: Dict[str, Any],
                action: torch.Tensor,
                robot_state: Dict[str, Any],
                step_idx: int,
                is_terminal: bool = False):
        """
        Add a step to the current episode
        
        Args:
            obs_dict: Environment observation dictionary (with 'rgb' key)
            action: Robot action tensor [12] - [base(3), arm_pos(3), arm_rot(3), gripper(3)]
            robot_state: Dict containing robot state info
            step_idx: Current step index
            is_terminal: Whether this is the last step
        """
        
        if self.current_episode is None:
            raise ValueError("No episode started. Call start_episode() first.")
        
        # Determine step flags
        is_first = step_idx == 0
        is_last = is_terminal
        
        image_paths = {}
    
        # Extract different camera views from obs_dict
        if "rgb" in obs_dict:
            # Main camera (treat as exterior_1)
            exterior_1_path = os.path.join(self.episode_folder, "images", f"step_{step_idx:06d}_exterior_1.png")
            self._save_image(obs_dict["rgb"], exterior_1_path)
            image_paths["exterior_1"] = exterior_1_path
        
        if "wrist_rgb" in obs_dict:
            # Wrist camera
            wrist_path = os.path.join(self.episode_folder, "images", f"step_{step_idx:06d}_wrist.png")
            self._save_image(obs_dict["wrist_rgb"], wrist_path)
            image_paths["wrist"] = wrist_path
        
        if "overhead_rgb" in obs_dict:
            # Overhead/exterior camera 2
            exterior_2_path = os.path.join(self.episode_folder, "images", f"step_{step_idx:06d}_exterior_2.png")
            self._save_image(obs_dict["overhead_rgb"], exterior_2_path)
            image_paths["exterior_2"] = exterior_2_path
        
        # Fallback: use main camera for missing views
        default_image = image_paths.get("exterior_1", "")
        
        # Create step data structure
        step_data = {
            "step_idx": step_idx,
            "is_first": is_first,
            "is_last": is_last,
            "is_terminal": is_terminal,
            
            # Language instructions (repeated for each step in DROID format)
            "language_instruction": self.current_episode["language_instructions"]["language_instruction"],
            "language_instruction_2": self.current_episode["language_instructions"]["language_instruction_2"], 
            "language_instruction_3": self.current_episode["language_instructions"]["language_instruction_3"],
            
            # Observation data
            "observation": {
                "gripper_position": self._extract_gripper_position(robot_state),
                "cartesian_position": self._extract_cartesian_position(robot_state),
                "joint_position": self._extract_joint_position(robot_state),
                "wrist_image_left": image_paths.get("wrist", default_image),
                "exterior_image_1_left": image_paths.get("exterior_1", default_image),
                "exterior_image_2_left": image_paths.get("exterior_2", default_image),
            },
                
            # Action data (what the robot should do)
            "action_dict": {
                "gripper_position": self._extract_action_gripper_position(action),
                "gripper_velocity": self._extract_action_gripper_velocity(action),
                "cartesian_position": self._extract_action_cartesian_position(action),
                "cartesian_velocity": self._extract_action_cartesian_velocity(action),
                "joint_position": self._extract_action_joint_position(action),
                "joint_velocity": self._extract_action_joint_velocity(action),
            },
            
            # Reward and discount
            "discount": 1.0,
            "reward": 1.0 if is_last else 0.0,  # Reward 1 on final step for demos
            "action": self._extract_raw_action(action),  # Raw action vector
            
            # Timestamp
            "timestamp": datetime.now().isoformat(),
        }
        
        self.episode_steps.append(step_data)
        
        if step_idx % 10 == 0:
            print(f"📊 Added step {step_idx} to episode {self.episode_id}")
    
    def end_episode(self):
        """End current episode and save to disk"""
        
        if self.current_episode is None:
            raise ValueError("No episode to end.")
        
        # Mark last step
        if self.episode_steps:
            self.episode_steps[-1]["is_last"] = True
            self.episode_steps[-1]["is_terminal"] = True
            self.episode_steps[-1]["reward"] = 1.0
        
        # Complete episode data
        episode_data = {
            "episode_metadata": self.current_episode["episode_metadata"],
            "steps": self.episode_steps,
            "episode_stats": {
                "total_steps": len(self.episode_steps),
                "duration_seconds": (datetime.now() - self.recording_start_time).total_seconds(),
                "end_time": datetime.now().isoformat(),
            }
        }
        
        # Save episode data to JSON
        episode_file = os.path.join(self.episode_folder, "episode_data.json")
        with open(episode_file, 'w') as f:
            json.dump(episode_data, f, indent=2, default=str)
        
        print(f"💾 Episode {self.episode_id} saved with {len(self.episode_steps)} steps")
        print(f"📁 Saved to: {episode_file}")
        
        # Reset for next episode
        self.current_episode = None
        self.episode_steps = []
        self.episode_id = None
    
    def _save_image(self, image_tensor: torch.Tensor, image_path: str):
        """Save image tensor to file"""
        try:
            if image_tensor.dim() == 4:
                img_np = image_tensor[0].cpu().numpy()
            else:
                img_np = image_tensor.cpu().numpy()
            
            # Convert to uint8
            if img_np.dtype != np.uint8:
                if img_np.max() <= 1.0:
                    img_np = (img_np * 255).astype(np.uint8)
                else:
                    img_np = np.clip(img_np, 0, 255).astype(np.uint8)
            
            # Ensure shape is (H, W, 3)
            if img_np.shape[0] in [1, 3] and img_np.shape[-1] != 3:
                img_np = np.transpose(img_np, (1, 2, 0))
            
            # Save image
            import cv2
            cv2.imwrite(image_path, cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR))
            
        except Exception as e:
            print(f"❌ Failed to save image {image_path}: {e}")
    
    # TODO: Implement these extraction methods based on your robot state format
    def _extract_gripper_position(self, robot_state: Dict) -> List[float]:
        """Extract gripper position from robot state"""
        # TODO: Extract actual gripper position from robot_state
        return [0.0]  # Placeholder
    
    def _extract_cartesian_position(self, robot_state: Dict) -> List[float]:
        """Extract 6D cartesian position [x, y, z, rx, ry, rz]"""
        # TODO: Extract from robot_state["ee_pos"] and robot_state["ee_quat"]
        return [0.0] * 6  # Placeholder
    
    def _extract_joint_position(self, robot_state: Dict) -> List[float]:
        """Extract joint positions"""
        # TODO: Extract from robot_state["joint_pos"]
        return [0.0] * 7  # Placeholder for 7 joints
    
    def _extract_action_gripper_position(self, action: torch.Tensor) -> List[float]:
        """Extract gripper position command from action"""
        # TODO: Extract from action[9:12] or wherever gripper commands are
        return [0.0]  # Placeholder
    
    def _extract_action_gripper_velocity(self, action: torch.Tensor) -> List[float]:
        """Extract gripper velocity command"""
        return [0.0]  # Placeholder
    
    def _extract_action_cartesian_position(self, action: torch.Tensor) -> List[float]:
        """Extract cartesian position command from action"""
        # TODO: Extract from action[3:9] (arm_pos + arm_rot)
        return [0.0] * 6  # Placeholder
    
    def _extract_action_cartesian_velocity(self, action: torch.Tensor) -> List[float]:
        """Extract cartesian velocity command"""
        return [0.0] * 6  # Placeholder
    
    def _extract_action_joint_position(self, action: torch.Tensor) -> List[float]:
        """Extract joint position command"""
        return [0.0] * 7  # Placeholder
    
    def _extract_action_joint_velocity(self, action: torch.Tensor) -> List[float]:
        """Extract joint velocity command"""
        return [0.0] * 7  # Placeholder
    
    def _extract_raw_action(self, action: torch.Tensor) -> List[float]:
        """Extract raw action vector"""
        return action.cpu().numpy().flatten().tolist()