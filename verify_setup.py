"""
AnCore - Setup Verification Script
Checks if everything is properly configured before running the system
"""

import sys
import os
from pathlib import Path


def print_header(text):
    """Print a formatted header"""
    print("\n" + "="*70)
    print(f" {text}")
    print("="*70)


def print_status(item, status, message=""):
    """Print status with icon"""
    icon = "✓" if status else "✗"
    color = "\033[92m" if status else "\033[91m"
    reset = "\033[0m"
    status_text = "OK" if status else "FAILED"
    msg = f" - {message}" if message else ""
    print(f"{color}{icon} {item}: {status_text}{reset}{msg}")


def check_python_version():
    """Check Python version"""
    version = sys.version_info
    required = (3, 8)
    is_ok = version >= required
    version_str = f"{version.major}.{version.minor}.{version.micro}"
    required_str = f"{required[0]}.{required[1]}+"
    
    print_status(
        "Python Version", 
        is_ok,
        f"Current: {version_str}, Required: {required_str}"
    )
    return is_ok


def check_file_exists(filepath, description):
    """Check if a file exists"""
    exists = os.path.exists(filepath)
    print_status(description, exists, filepath)
    return exists


def check_package(package_name):
    """Check if a Python package is installed"""
    try:
        __import__(package_name)
        print_status(f"Package: {package_name}", True, "Installed")
        return True
    except ImportError:
        print_status(f"Package: {package_name}", False, "Not installed")
        return False


def check_packages():
    """Check all required packages"""
    packages = [
        'torch',
        'transformers',
        'pandas',
        'numpy',
        'sklearn',
        'matplotlib',
        'seaborn',
        'tqdm'
    ]
    
    results = []
    for package in packages:
        results.append(check_package(package))
    
    return all(results)


def check_cuda():
    """Check CUDA availability"""
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        if cuda_available:
            device_name = torch.cuda.get_device_name(0)
            print_status("CUDA GPU", True, f"Available: {device_name}")
        else:
            print_status("CUDA GPU", False, "Not available (will use CPU)")
        return cuda_available
    except:
        print_status("CUDA GPU", False, "Cannot check (torch not installed)")
        return False


def check_project_structure():
    """Check if all project files are present"""
    required_files = [
        'ancore_config.py',
        'ancore_dataset.py',
        'ancore_model.py',
        'ancore_trainer.py',
        'ancore_main.py',
        'requirements.txt',
        'README.md',
        'QUICKSTART.md'
    ]
    
    results = []
    for file in required_files:
        results.append(check_file_exists(file, f"File: {file}"))
    
    return all(results)


def check_data_files():
    """Check if data files are present"""
    data_files = [
        os.path.join('fakenews', 'full.csv')
    ]
    
    results = []
    for file in data_files:
        results.append(check_file_exists(file, f"Data: {file}"))
    
    return all(results)


def estimate_requirements():
    """Estimate system requirements"""
    print("\n" + "-"*70)
    print("ESTIMATED REQUIREMENTS:")
    print("-"*70)
    print("  Disk Space:")
    print("    - Model files: ~500 MB")
    print("    - Dependencies: ~2-3 GB")
    print("    - Output files: ~10-50 MB")
    print("  ")
    print("  Memory:")
    print("    - Minimum RAM: 8 GB")
    print("    - Recommended RAM: 16 GB")
    print("  ")
    print("  Training Time (approximate):")
    print("    - With GPU: 15-30 minutes")
    print("    - With CPU: 1-2 hours")


def print_next_steps(all_ok):
    """Print next steps based on verification results"""
    print("\n" + "="*70)
    if all_ok:
        print(" VERIFICATION COMPLETE - ALL CHECKS PASSED!")
    else:
        print(" VERIFICATION COMPLETE - SOME ISSUES FOUND")
    print("="*70)
    
    if all_ok:
        print("\n🎉 Your system is ready!")
        print("\n📝 Next steps:")
        print("  1. Run demo: python demo.py")
        print("  2. Explore data: python explore_data.py")
        print("  3. Train model: python ancore_main.py --mode train")
        print("  4. Test predictions: python ancore_main.py --mode interactive")
    else:
        print("\n⚠️  Please fix the issues above before proceeding.")
        print("\n🔧 Common fixes:")
        print("  - Install packages: pip install -r requirements.txt")
        print("  - Check file paths and names")
        print("  - Verify Python version (3.8+)")


def show_quick_commands():
    """Show quick reference commands"""
    print("\n" + "-"*70)
    print("QUICK REFERENCE COMMANDS:")
    print("-"*70)
    print("  Install dependencies:")
    print("    pip install -r requirements.txt")
    print("  ")
    print("  Run demo:")
    print("    python demo.py")
    print("  ")
    print("  Explore data:")
    print("    python explore_data.py")
    print("  ")
    print("  Train model:")
    print("    python ancore_main.py --mode train")
    print("  ")
    print("  Interactive mode:")
    print("    python ancore_main.py --mode interactive")
    print("  ")
    print("  Get help:")
    print("    python ancore_main.py --help")


def main():
    """Main verification function"""
    print_header("AnCore Setup Verification")
    print("This script checks if your system is ready to run AnCore.\n")
    
    checks = []
    
    # Check Python version
    print_header("Python Environment")
    checks.append(check_python_version())
    
    # Check project structure
    print_header("Project Files")
    checks.append(check_project_structure())
    
    # Check data files
    print_header("Data Files")
    checks.append(check_data_files())
    
    # Check packages
    print_header("Python Packages")
    checks.append(check_packages())
    
    # Check CUDA (optional)
    print_header("Hardware Acceleration (Optional)")
    check_cuda()  # Not required, so don't add to checks
    
    # Estimate requirements
    estimate_requirements()
    
    # Show quick commands
    show_quick_commands()
    
    # Print next steps
    all_ok = all(checks)
    print_next_steps(all_ok)
    
    print("\n" + "="*70)
    print(f" Overall Status: {'READY' if all_ok else 'NOT READY'}")
    print("="*70 + "\n")
    
    return 0 if all_ok else 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
