from transformers import AutoModelForVision2Seq, AutoProcessor
import transformers.utils.import_utils
import transformers.modeling_utils

import torch
import numpy as np
from PIL import Image
import gc

import time
import cv2
import os

# Disable flash attention warnings (from Spot_In_Scene.py)
transformers.utils.import_utils._flash_attn_available = False
transformers.modeling_utils.is_flash_attn_available = lambda: False

class OpenVLAAssistant:
    """OpenVLA integration for Isaac Lab (adapted from Spot_In_Scene.py)"""
    
    def __init__(self, enabled=True):
        self.enabled = enabled
        self.vla = None
        self.processor = None
        self.ready = False
        
        if self.enabled:
            torch.cuda.empty_cache()
            self._load_model()
    
    def _load_model(self):
        """Load OpenVLA model (adapted from Spot_In_Scene.py)"""
        try:
            print("Loading OpenVLA model...")
            if torch.cuda.is_available():
                total_memory = torch.cuda.get_device_properties(0).total_memory
                allocated_memory = torch.cuda.memory_allocated(0)
                free_memory = total_memory - allocated_memory
                print(f"📊 GPU Memory: {free_memory/1e9:.1f}GB free / {total_memory/1e9:.1f}GB total")
                
                if free_memory < 4e9:  # Less than 4GB free
                    print("⚠️ Warning: Low GPU memory, OpenVLA may fail")
            
            self.processor = AutoProcessor.from_pretrained("openvla/openvla-7b", trust_remote_code=True)
            
            self.vla = AutoModelForVision2Seq.from_pretrained(
                "openvla/openvla-7b", 
                attn_implementation="flash_attention_2",  
                torch_dtype=torch.bfloat16, 
                low_cpu_mem_usage=True, 
                trust_remote_code=True,
                device_map="cuda:0",
            )
            
            allocated_after = torch.cuda.memory_allocated(0)
            print(f"📊 OpenVLA using {(allocated_after)/1e9:.1f}GB GPU memory")

            self.vla = self.vla.to("cuda:0")
            self.ready = True
            print("✅ OpenVLA loaded successfully")
            
            
        except Exception as e:
            print(f"Failed to load OpenVLA: {e}")
            self.enabled = False
            self.ready = False

            torch.cuda.empty_cache()
            gc.collect()
    
    def get_action_suggestion(self, rgb_image, prompt):
        """Get action suggestion from OpenVLA"""
        if not self.ready:
            return None
            
        try:
            #Isaac Lab always gives uint8
            if isinstance(rgb_image, torch.Tensor):
                # Handle Isaac Lab format: [num_envs, H, W, 3] or [H, W, 3]
                if rgb_image.dim() == 4:
                    image_np = rgb_image[0].cpu().numpy()  # Take first environment
                else:
                    image_np = rgb_image.cpu().numpy()
            elif isinstance(rgb_image, np.ndarray):
                image_np = rgb_image
            else:
                # Already PIL Image
                pil_image = rgb_image.convert("RGB")
                
            # Direct conversion (no scaling needed for uint8)
            if 'image_np' in locals():
                pil_image = Image.fromarray(image_np)
            
            # Always convert to RGB
            pil_image = pil_image.convert("RGB")
            #self.safePilImageToFile(pil_image, "openvla_input")

                
                # Format prompt
            formatted_prompt = f"In: {prompt}\nOut:"
            
            
            with torch.no_grad():
                inputs = self.processor(formatted_prompt, pil_image).to("cuda:0", dtype=torch.bfloat16)
                action = self.vla.predict_action(**inputs, unnorm_key="bridge_orig", do_sample=False)
            
            # Clean up inputs from GPU
            del inputs
            torch.cuda.empty_cache()
            
            # Get prediction
            #inputs = self.processor(formatted_prompt, pil_image).to("cuda:0", dtype=torch.bfloat16)
            #action = self.vla.predict_action(**inputs, unnorm_key="bridge_orig", do_sample=False)
            
            return action
            
        except Exception as e:
            print(f"OpenVLA prediction error: {e}")
            return None
        
    def safePilImageToFile(self, pil_image, prefix="openvla_input"):
        """Save PIL image to file for debugging OpenVLA input"""
        if pil_image is not None:
            timestamp = int(time.time() * 1000)
            filename = f"out/{prefix}_{timestamp}"
            os.makedirs("out", exist_ok=True)
            
            try:
                # Method 1: Direct PIL save (simplest)
                pil_image.save(f"{filename}.png")
                print(f"✅ Saved PIL image: {filename}.png")
                
                # Optional Method 2: Using OpenCV (for consistency with your existing method)
                # Convert PIL to numpy array
                img_np = np.array(pil_image)
                
                # Convert RGB to BGR for OpenCV (PIL uses RGB, OpenCV uses BGR)
                img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
                
                cv2.imwrite(f"{filename}_cv2.png", img_bgr)
                print(f"✅ Saved with OpenCV: {filename}_cv2.png")
                
            except Exception as error:
                print(f"❌ Save PIL image failed: {error}")