#!/usr/bin/env python
"""
Setup script for Advanced EDA on MIND Dataset.

This script:
1. Checks dependencies
2. Downloads NLTK resources
3. Creates necessary directories
4. Validates data paths
"""

import os
import sys
import subprocess
import pkg_resources
import platform

# Define paths
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
SCRIPTS_DIR = os.path.join(ROOT_DIR, 'scripts')
NOTEBOOK_DIR = os.path.join(ROOT_DIR, 'notebook')
STREAMLIT_DIR = os.path.join(ROOT_DIR, 'streamlit')
DATA_DIR = os.path.join(ROOT_DIR, 'data')
REPO_ROOT = os.path.join(ROOT_DIR, '..')

# MIND dataset paths
MIND_DIR = os.path.join(REPO_ROOT, 'MINDLarge')
MIND_TRAIN_DIR = os.path.join(MIND_DIR, 'MINDlarge_train')
MIND_DEV_DIR = os.path.join(MIND_DIR, 'MINDlarge_dev')
MIND_TEST_DIR = os.path.join(MIND_DIR, 'MINDlarge_test')

# Check if required directories exist
def check_directories():
    """Check if the required directories exist and create them if needed."""
    print("Checking directories...")
    
    dirs_to_check = [
        SCRIPTS_DIR,
        NOTEBOOK_DIR,
        STREAMLIT_DIR,
        DATA_DIR
    ]
    
    for directory in dirs_to_check:
        if not os.path.exists(directory):
            print(f"Creating directory: {directory}")
            os.makedirs(directory, exist_ok=True)
    
    print("Directory check completed.")

# Check if required packages are installed
def check_dependencies():
    """Check if required packages are installed."""
    print("Checking dependencies...")
    
    requirements_file = os.path.join(ROOT_DIR, 'requirements.txt')
    
    if not os.path.exists(requirements_file):
        print("Warning: requirements.txt not found. Cannot check dependencies.")
        return
    
    with open(requirements_file, 'r') as f:
        requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]
    
    # Check each requirement
    missing_packages = []
    for req in requirements:
        package_name = req.split('>=')[0].split('==')[0].strip()
        try:
            pkg_resources.get_distribution(package_name)
            print(f"✅ {package_name} is installed")
        except pkg_resources.DistributionNotFound:
            missing_packages.append(req)
            print(f"❌ {package_name} is NOT installed")
    
    # Suggest installing missing packages
    if missing_packages:
        print("\nSome required packages are missing. Install them with:")
        print(f"pip install {' '.join(missing_packages)}")
        
        # Ask if user wants to install them now
        answer = input("Do you want to install them now? (y/n): ")
        if answer.lower() == 'y':
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install"] + missing_packages)
                print("Packages installed successfully.")
            except subprocess.CalledProcessError as e:
                print(f"Error installing packages: {e}")
        else:
            print("Please install the missing packages manually.")
    else:
        print("All required packages are installed.")

# Download NLTK resources
def download_nltk_resources():
    """Download required NLTK resources."""
    print("Checking NLTK resources...")
    
    try:
        import nltk
        
        # Create NLTK data directory if it doesn't exist
        nltk_data_dir = os.path.join(os.path.expanduser('~'), 'nltk_data')
        os.makedirs(nltk_data_dir, exist_ok=True)
        
        # Resources to download
        resources = [
            'punkt',
            'stopwords',
            'wordnet',
            'vader_lexicon'
        ]
        
        # Download each resource
        for resource in resources:
            try:
                nltk.data.find(f'tokenizers/{resource}' if resource == 'punkt' else resource)
                print(f"✅ NLTK resource '{resource}' is already downloaded")
            except LookupError:
                print(f"Downloading NLTK resource: {resource}")
                nltk.download(resource, quiet=True)
                print(f"✅ NLTK resource '{resource}' downloaded successfully")
    
    except ImportError:
        print("❌ NLTK is not installed. Please install it with: pip install nltk")

# Check data files
def check_data_files():
    """Check if MIND dataset files exist."""
    print("Checking MIND dataset files...")
    
    # Check if MIND directories exist
    if not os.path.exists(MIND_DIR):
        print(f"❌ MIND dataset directory not found: {MIND_DIR}")
        print("Please make sure the MINDLarge directory is present in the repository root.")
        return
    
    # Check train/dev/test directories
    dirs_to_check = [
        (MIND_TRAIN_DIR, "Train"),
        (MIND_DEV_DIR, "Dev"),
        (MIND_TEST_DIR, "Test")
    ]
    
    all_found = True
    for directory, name in dirs_to_check:
        if os.path.exists(directory):
            print(f"✅ MIND{name} directory found: {directory}")
            
            # Check if required files exist
            for filename in ['news.tsv', 'behaviors.tsv', 'entity_embedding.vec', 'relation_embedding.vec']:
                filepath = os.path.join(directory, filename)
                if os.path.exists(filepath):
                    print(f"  ✅ {filename} found")
                else:
                    print(f"  ❌ {filename} not found")
                    all_found = False
        else:
            print(f"❌ MIND{name} directory not found: {directory}")
            all_found = False
    
    if all_found:
        print("All required MIND dataset files are present.")
    else:
        print("Some MIND dataset files are missing.")

# Main setup function
def main():
    """Run the setup process."""
    print("=" * 80)
    print("Advanced EDA for MIND Dataset - Setup")
    print("=" * 80)
    
    # Print system information
    print(f"Python version: {sys.version}")
    print(f"Platform: {platform.platform()}")
    print(f"Working directory: {os.getcwd()}")
    print()
    
    # Run checks
    check_directories()
    print()
    
    check_dependencies()
    print()
    
    download_nltk_resources()
    print()
    
    check_data_files()
    print()
    
    print("=" * 80)
    print("Setup completed.")
    print("=" * 80)
    print("\nTo run the Jupyter notebook:")
    print(f"cd {NOTEBOOK_DIR}")
    print("jupyter notebook advanced_eda.ipynb")
    print()
    print("To run the Streamlit app:")
    print(f"cd {STREAMLIT_DIR}")
    print("streamlit run app.py")
    print("=" * 80)

if __name__ == "__main__":
    main()