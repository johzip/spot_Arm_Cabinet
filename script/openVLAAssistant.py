from transformers import AutoModelForVision2Seq, AutoProcessor
import transformers.utils.import_utils
import transformers.modeling_utils

import torch
import numpy as np
from PIL import Image
import gc

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