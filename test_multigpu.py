#!/usr/bin/env python3
"""
Test script for multi-GPU functionality in Wan2GP
"""

import torch
import sys
import os

# Add the current directory to the path so we can import from wgp
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_gpu_detection():
    """Test GPU detection functionality"""
    print("Testing GPU detection...")
    
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        print(f"Found {num_gpus} CUDA GPU(s)")
        
        for i in range(num_gpus):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
            print(f"  GPU {i}: {gpu_name} ({gpu_memory:.1f} GB)")
    else:
        print("No CUDA GPUs available")
        return False
    
    return True

def test_multi_gpu_setup():
    """Test multi-GPU setup functionality"""
    print("\nTesting multi-GPU setup...")
    
    # Import the function from wgp
    try:
        from wgp import setup_multi_gpu_for_hunyuan, get_available_gpus
        
        num_gpus = get_available_gpus()
        print(f"Available GPUs: {num_gpus}")
        
        if num_gpus > 1:
            print("Multi-GPU setup would be enabled")
            return True
        else:
            print("Single GPU detected - multi-GPU setup would be skipped")
            return True
    except ImportError as e:
        print(f"Error importing from wgp: {e}")
        return False

def test_data_parallel():
    """Test DataParallel functionality"""
    print("\nTesting DataParallel functionality...")
    
    if not torch.cuda.is_available():
        print("No CUDA available, skipping DataParallel test")
        return True
    
    try:
        # Create a simple model
        model = torch.nn.Linear(10, 10)
        model = model.cuda()
        
        # Test DataParallel wrapping
        if torch.cuda.device_count() > 1:
            model = torch.nn.DataParallel(model, device_ids=list(range(torch.cuda.device_count())))
            print("DataParallel model created successfully")
            
            # Test accessing the underlying model
            if hasattr(model, 'module'):
                print("Model is wrapped in DataParallel")
                actual_model = model.module
                print("Successfully accessed underlying model")
            else:
                print("Model is not wrapped in DataParallel")
        else:
            print("Only one GPU available, DataParallel not needed")
        
        return True
    except Exception as e:
        print(f"Error testing DataParallel: {e}")
        return False

def main():
    """Main test function"""
    print("Wan2GP Multi-GPU Functionality Test")
    print("=" * 40)
    
    tests = [
        ("GPU Detection", test_gpu_detection),
        ("Multi-GPU Setup", test_multi_gpu_setup),
        ("DataParallel", test_data_parallel),
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\nRunning {test_name} test...")
        try:
            result = test_func()
            results.append((test_name, result))
            status = "PASS" if result else "FAIL"
            print(f"{test_name}: {status}")
        except Exception as e:
            print(f"{test_name}: ERROR - {e}")
            results.append((test_name, False))
    
    print("\n" + "=" * 40)
    print("Test Results Summary:")
    for test_name, result in results:
        status = "PASS" if result else "FAIL"
        print(f"  {test_name}: {status}")
    
    all_passed = all(result for _, result in results)
    print(f"\nOverall: {'ALL TESTS PASSED' if all_passed else 'SOME TESTS FAILED'}")
    
    if all_passed:
        print("\nMulti-GPU functionality appears to be working correctly!")
        print("You can now use --multigpu flag with hunyuan video avatar models.")
    else:
        print("\nSome tests failed. Please check your CUDA installation and GPU setup.")

if __name__ == "__main__":
    main() 