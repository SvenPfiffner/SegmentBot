# Set up the environment for this project

import os
import sys
import subprocess

venv_dir = "venv"

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def create_venv():
    print("Creating virtual environment...")
    os.system(f"python -m venv {venv_dir}")
    print(f"{bcolors.OKGREEN}Virtual environment created.{bcolors.ENDC}")

def install_packages():
    print("Installing required packages...")
    os.system(f"{venv_dir}/bin/pip install -r requirements.txt")
    print(f"{bcolors.OKGREEN}Packages installed.{bcolors.ENDC}")

def install_segment_anything():
    print("Installing Segment Anything...")
    os.system(f"{venv_dir}/bin/pip install 'git+https://github.com/facebookresearch/segment-anything.git'")
    print(f"{bcolors.OKGREEN}Segment Anything installed.{bcolors.ENDC}")

def load_models():

    print("Loading models...")
    if not os.path.exists("betabot/checkpoints"):
        os.makedirs("betabot/checkpoints")
    
    if not os.path.exists("betabot/checkpoints/sam_vit_h_4b8939.pth"):
        os.system("wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth -p betabot/checkpoints")
    
    if not os.path.exists("betabot/checkpoints/groundingdino_swint_ogc.pth"):
        os.system("wget https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth -p betabot/checkpoints")
    
    print(f"{bcolors.OKGREEN}Models loaded.{bcolors.ENDC}")

def main():
    try:
        if not os.path.exists(venv_dir):
            create_venv()
        
        install_packages()
        install_segment_anything()
        load_models()

        print(f"{bcolors.OKGREEN}Setup complete.{bcolors.ENDC}")

        return 0
    
    except Exception as e:
        print(f"{bcolors.FAIL}An error occurred during setup.{bcolors.ENDC}")
        print(bcolors.WARNING + e + bcolors.ENDC)
        return 1

    
if __name__ == "__main__":
    status = main()
    if status != 0:
        print(f"{bcolors.FAIL}Setup failed. Please consult the error message above.{bcolors.ENDC}")

    sys.exit(status)