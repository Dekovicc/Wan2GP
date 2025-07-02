#!/usr/bin/env python3
"""
Test script to verify wgp.py --i2v --multigpu --share doesn't hang
"""

import sys
import os
import subprocess
import time
import signal

def test_wgp_multigpu():
    """Test that wgp.py with multigpu doesn't hang"""
    print("Testing wgp.py --i2v --multigpu --share...")
    
    # Start the process
    cmd = [sys.executable, "wgp.py", "--i2v", "--multigpu", "--share"]
    
    try:
        # Start the process
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        # Wait for a reasonable amount of time (30 seconds)
        timeout = 30
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            # Check if process is still running
            if process.poll() is not None:
                # Process finished
                stdout, stderr = process.communicate()
                print(f"Process finished with return code: {process.returncode}")
                print(f"STDOUT: {stdout}")
                print(f"STDERR: {stderr}")
                
                if process.returncode == 0:
                    print("✅ Process completed successfully!")
                    return True
                else:
                    print("❌ Process failed")
                    return False
            
            # Check if we see the expected output
            if process.stdout:
                line = process.stdout.readline()
                if line:
                    print(f"Output: {line.strip()}")
                    if "Setting up multi-GPU mode" in line and "Successfully verified access" in line:
                        print("✅ Multi-GPU setup completed successfully!")
                        # Kill the process since we've verified it works
                        process.terminate()
                        time.sleep(2)
                        if process.poll() is None:
                            process.kill()
                        return True
            
            time.sleep(0.1)
        
        # If we get here, the process is still running after timeout
        print(f"⚠️ Process still running after {timeout} seconds, terminating...")
        process.terminate()
        time.sleep(2)
        if process.poll() is None:
            process.kill()
        
        print("❌ Process hung or took too long")
        return False
        
    except Exception as e:
        print(f"❌ Error running process: {e}")
        return False

if __name__ == "__main__":
    success = test_wgp_multigpu()
    sys.exit(0 if success else 1) 