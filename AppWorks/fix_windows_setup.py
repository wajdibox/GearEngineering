#!/usr/bin/env python3
"""
Windows Setup Fix for AppWorks

Run this script to fix the matplotlib import issue and set up AppWorks properly on Windows.
"""

import subprocess
import sys
import os

def run_command(command, description):
    """Run a command and return success status"""
    try:
        print(f"Running: {description}...")
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✓ {description} completed successfully")
            return True
        else:
            print(f"✗ {description} failed: {result.stderr}")
            return False
    except Exception as e:
        print(f"✗ Error with {description}: {e}")
        return False

def main():
    print("AppWorks Windows Setup Fix")
    print("=" * 50)
    
    # Install missing packages
    packages = [
        "matplotlib",
        "numpy", 
        "scipy",
        "pandas",
        "scikit-learn"
    ]
    
    print("Installing required Python packages...")
    success = True
    
    for package in packages:
        cmd = f"{sys.executable} -m pip install {package}"
        if not run_command(cmd, f"Installing {package}"):
            success = False
    
    if success:
        print("\n✓ All packages installed successfully!")
        
        # Test AppWorks import
        print("\nTesting AppWorks import...")
        try:
            # Test basic imports
            import matplotlib
            import numpy
            import scipy
            import pandas
            import sklearn
            print("✓ All required packages can be imported")
            
            # Test AppWorks specifically  
            sys.path.insert(0, '.')
            from AppWorks.data_generation import sample_parameters
            params = sample_parameters(n_samples=1, random_state=42)
            print("✓ AppWorks package loads correctly")
            
            print("\n" + "=" * 50)
            print("SUCCESS! AppWorks is now ready to use.")
            print("\nYou can now run:")
            print("  python -m AppWorks.main")
            print("  python simple_example.py")
            
        except Exception as e:
            print(f"✗ Import test failed: {e}")
            print("\nPlease run the following command manually:")
            print("pip install matplotlib numpy scipy pandas scikit-learn")
    
    else:
        print("\n✗ Some packages failed to install.")
        print("Please install manually using:")
        print("pip install matplotlib numpy scipy pandas scikit-learn")

if __name__ == "__main__":
    main()
    input("\nPress Enter to close...")