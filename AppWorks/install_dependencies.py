#!/usr/bin/env python3
"""
AppWorks Dependency Installer for Windows

This script installs all required dependencies for AppWorks to work properly
on your Windows system.
"""

import subprocess
import sys
import os

def install_package(package_name):
    """Install a Python package using pip"""
    try:
        print(f"Installing {package_name}...")
        result = subprocess.run([sys.executable, "-m", "pip", "install", package_name], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✓ {package_name} installed successfully")
            return True
        else:
            print(f"✗ Failed to install {package_name}: {result.stderr}")
            return False
    except Exception as e:
        print(f"✗ Error installing {package_name}: {e}")
        return False

def main():
    """Install all required dependencies"""
    print("="*60)
    print("AppWorks Dependency Installer")
    print("="*60)
    print()
    
    # Required packages
    required_packages = [
        "matplotlib>=3.5.0",
        "numpy>=1.21.0", 
        "scipy>=1.7.0",
        "pandas>=1.3.0",
        "scikit-learn>=1.0.0"
    ]
    
    # Optional packages (install if possible)
    optional_packages = [
        "pyfftw>=0.12.0",
        "gplearn>=0.4.0"
    ]
    
    print("Installing required packages...")
    success_count = 0
    
    for package in required_packages:
        if install_package(package):
            success_count += 1
    
    print(f"\nRequired packages: {success_count}/{len(required_packages)} installed")
    
    print("\nInstalling optional packages...")
    optional_success = 0
    
    for package in optional_packages:
        if install_package(package):
            optional_success += 1
    
    print(f"Optional packages: {optional_success}/{len(optional_packages)} installed")
    
    print("\n" + "="*60)
    
    if success_count == len(required_packages):
        print("✓ All required dependencies installed successfully!")
        print("\nYou can now run:")
        print("  python -m AppWorks.main")
        print("  python simple_example.py")
    else:
        print("✗ Some required packages failed to install.")
        print("Please check the error messages above and install manually:")
        print("  pip install matplotlib numpy scipy pandas scikit-learn")
    
    print("\nPress Enter to continue...")
    input()

if __name__ == "__main__":
    main()