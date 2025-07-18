# mechanic

YOLO engine file compiler with an easy to use GUI interface.

### Usage
1. Clone the repository.

2. Install Tkinter if you haven't already with `sudo apt install python3-tk`.

3. Create a virtual environment: `python3 -m venv .venv`. You may need to install some packages to do this on Linux. 

4. Install requirements: `pip install -r requirements.txt`.

5. Optionally create a .env file with the following line `TENSOR_RT_PATH="path/to/tensorRT"`

4. Run mechanic.py: `python3 mechanic.py`

5. Configure your settings. Make sure to specify where your pytorch file is located. The TensorRT install location should be the location where trtexec is located. 

