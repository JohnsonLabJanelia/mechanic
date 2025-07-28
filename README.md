# mechanic

Adapted from [YOLOv8-TensorRT](https://github.com/triple-Mu/YOLOv8-TensorRT) and [Nividia TensorRT](https://developer.nvidia.com/tensorrt). 

YOLO model training and engine file compilation with an easy to use GUI interface.

## Setup
1. Clone the repository, `git clone git@github.com:JohnsonLabJanelia/mechanic.git` and make sure YOLOv8-TensorRT and Nividida TensorRT are properly installed.

2. Install Tkinter if you haven't already with `sudo apt install python3-tk`.

3. Create a virtual environment in the repository directory: `python3 -m venv .venv`, and activate it `source .venv/bin/activate`. You may need to install some packages to do this on Linux. 

4. Install requirements: `pip install -r requirements.txt`.

5. Optionally create a .env file with any of the desired fields to auto input into the GUI.
```
YOLO_PATH = "/path/to/yolo"
TENSORT_RT_PATH = "/path/to/tensorrt"
OUTPUT_PATH = "/path/to/output"
GPU_DEVICE = "gpu#"
TASK = "task"
```
## Usage

1. Run mechanic: `python mechanic.py`.

2. All fields in .env file will be automatically inputted into the GUI and some fields will have default values. Modify fields to desired values, and ensure that required fields are filled in. 
    - Bold font means a field is required for all modes
    - If required fields are invalid, program won't run and popup will denote which field(s) is invalid.

3. Mechanic has three modes:
    - `TRAIN`: train a yolo model and output a .pt file
    - `CONVERT`: convert a .pt weights file to an .engine file to run on a specific gpu
    - `TRAIN & CONVERT`: perform both actions

4. Some fields only need to be inputted for specific modes, for example input file is only required if the mode is `CONVERT`.

5. When the mode is set to `TRAIN` or `TRAIN & CONVERT` the best.pt file is outputted in the latest train folder in `YOLO_PATH/runs/detect/train#` and copied into the user specified output directory.
