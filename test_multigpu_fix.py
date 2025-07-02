#!/usr/bin/env python3
"""
Test script to verify multi-GPU setup works without hanging
"""

import torch
import sys
import os

# Add the current directory to the path so we can import from wgp
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_multi_gpu_setup():
    """Test the multi-GPU setup function"""
    print("Testing multi-GPU setup...")
    
    # Check if CUDA is available
    if not torch.cuda.is_available():
        print("CUDA not available, skipping test")
        return True
    
    num_gpus = torch.cuda.device_count()
    print(f"Found {num_gpus} GPUs")
    
    if num_gpus <= 1:
        print("Only 1 or 0 GPUs available, test passed")
        return True
    
    # Test the setup function
    try:
        from wgp import setup_multi_gpu_for_hunyuan
        
        # Mock args.multigpu to True
        import wgp
        wgp.args = type('Args', (), {'multigpu': True})()
        
        result = setup_multi_gpu_for_hunyuan()
        print(f"Setup result: {result}")
        
        if result == num_gpus:
            print("✅ Multi-GPU setup successful!")
            return True
        else:
            print(f"❌ Expected {num_gpus}, got {result}")
            return False
            
    except Exception as e:
        print(f"❌ Error during setup: {e}")
        return False

if __name__ == "__main__":
    success = test_multi_gpu_setup()
    sys.exit(0 if success else 1) 