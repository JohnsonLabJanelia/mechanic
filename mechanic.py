import tkinter as tk
from tkinter import ttk, filedialog, messagebox, font
import subprocess
import threading
import os
from pathlib import Path
from dotenv import load_dotenv

class TensorRTConverterGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("mechanic")
        self.root.geometry("900x1000")

        load_dotenv()

        # File name and path variables
        self.yolo_path = tk.StringVar(value=os.getenv('YOLO_PATH'))
        self.data_folder = tk.StringVar()
        self.model = tk.StringVar(value="yolov8n.pt")
        self.epochs = tk.StringVar(value="100")
        self.imgsz = tk.StringVar(value="640")

        # Conversion variables
        self.tensorrt_path = tk.StringVar(value=os.getenv('TENSOR_RT_PATH'))
        self.pt_file_path = tk.StringVar()
        self.output_location = tk.StringVar(value=os.getenv('OUTPUT_PATH'))
        self.output_name = tk.StringVar(value="yolomodel")
        self.iou_threshold = tk.StringVar(value="0.5")
        self.confidence_threshold = tk.StringVar(value="0.25")
        self.topk = tk.StringVar(value="1")
        self.input_shape = tk.StringVar(value="1 3 640 640")
        self.gpu_device = tk.StringVar(value=os.getenv("GPU_DEVICE") if os.getenv("GPU_DEVICE") and os.getenv("GPU_DEVICE").isdigit() else "0")
        self.precision = tk.StringVar(value="fp16")

        # TRAIN & CONVERT, TRAIN, or CONVERT
        self.mode = tk.StringVar(value=os.getenv("MODE") if os.getenv("MODE") and os.getenv("MODE") in ["TRAIN & CONVERT", "TRAIN", "CONVERT"] else "TRAIN & CONVERT")

        self.setup_gui()
        
    def setup_gui(self):
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)

        style = ttk.Style(self.root)
        style.theme_use("clam")
        bold_font = font.Font(weight="bold")

        # Training Section
        ttk.Label(main_frame, text="Training", font=bold_font).grid(row=0, column=0, columnspan=3, pady=10)

        ttk.Label(main_frame, text="YOLOv8-TensorRT install location:").grid(row=1, column=0, sticky=tk.W, pady=5)
        ttk.Entry(main_frame, textvariable=self.yolo_path, width=50).grid(row=1, column=1, sticky=(tk.W, tk.E), padx=5)
        ttk.Button(main_frame, text="Browse", command=self.browse_yolo_path).grid(row=1, column=2, padx=5)

        ttk.Label(main_frame, text="Data location:").grid(row=2, column=0, sticky=tk.W, pady=5)
        ttk.Entry(main_frame, textvariable=self.data_folder, width=50).grid(row=2, column=1, sticky=(tk.W, tk.E), padx=5)
        ttk.Button(main_frame, text="Browse", command=self.browse_data_folder).grid(row=2, column=2, padx=5)

        ttk.Label(main_frame, text="Model:").grid(row=3, column=0, sticky=tk.W, pady=5)
        ttk.Combobox(main_frame, textvariable=self.model, values=["yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "yolov8l.pt", "yolov8x.pt"], state="readonly", width=48).grid(row=3, column=1, sticky=(tk.W, tk.E), padx=5)

        ttk.Label(main_frame, text="Epochs:").grid(row=4, column=0, sticky=tk.W, pady=5)
        ttk.Entry(main_frame, textvariable=self.epochs, width=50).grid(row=4, column=1, sticky=(tk.W, tk.E), padx=5)

        ttk.Label(main_frame, text="Image size:").grid(row=5, column=0, sticky=tk.W, pady=5)
        ttk.Entry(main_frame, textvariable=self.imgsz, width=50).grid(row=5, column=1, sticky=(tk.W, tk.E), padx=5)

        # Conversion Section
        ttk.Label(main_frame, text="Conversion", font=bold_font).grid(row=6, column=0, columnspan=3, pady=10)

        ttk.Label(main_frame, text="TensorRT install location:").grid(row=7, column=0, sticky=tk.W, pady=5)
        ttk.Entry(main_frame, textvariable=self.tensorrt_path, width=50).grid(row=7, column=1, sticky=(tk.W, tk.E), padx=5)
        ttk.Button(main_frame, text="Browse", command=self.browse_tensorrt_path).grid(row=7, column=2, padx=5)

        ttk.Label(main_frame, text="Input file location:").grid(row=8, column=0, sticky=tk.W, pady=5)
        ttk.Entry(main_frame, textvariable=self.pt_file_path, width=50).grid(row=8, column=1, sticky=(tk.W, tk.E), padx=5)
        ttk.Button(main_frame, text="Browse", command=self.browse_pt_file).grid(row=8, column=2, padx=5)

        ttk.Label(main_frame, text="Output location:").grid(row=9, column=0, sticky=tk.W, pady=5)
        ttk.Entry(main_frame, textvariable=self.output_location, width=50).grid(row=9, column=1, sticky=(tk.W, tk.E), padx=5)
        ttk.Button(main_frame, text="Browse", command=self.browse_output_location).grid(row=9, column=2, padx=5)

        ttk.Label(main_frame, text="Output file name:").grid(row=10, column=0, sticky=tk.W, pady=5)
        ttk.Entry(main_frame, textvariable=self.output_name, width=50).grid(row=10, column=1, sticky=(tk.W, tk.E), padx=5)

        ttk.Label(main_frame, text="IOU threshold(s):").grid(row=11, column=0, sticky=tk.W, pady=5)
        ttk.Entry(main_frame, textvariable=self.iou_threshold, width=50).grid(row=11, column=1, sticky=(tk.W, tk.E), padx=5)

        ttk.Label(main_frame, text="Confidence threshold(s):").grid(row=12, column=0, sticky=tk.W, pady=5)
        ttk.Entry(main_frame, textvariable=self.confidence_threshold, width=50).grid(row=12, column=1, sticky=(tk.W, tk.E), padx=5)

        ttk.Label(main_frame, text="Maximum bounding boxes:").grid(row=13, column=0, sticky=tk.W, pady=5)
        ttk.Entry(main_frame, textvariable=self.topk, width=50).grid(row=13, column=1, sticky=(tk.W, tk.E), padx=5)

        ttk.Label(main_frame, text="Input shape:").grid(row=14, column=0, sticky=tk.W, pady=5)
        ttk.Entry(main_frame, textvariable=self.input_shape, width=50).grid(row=14, column=1, sticky=(tk.W, tk.E), padx=5)

        ttk.Label(main_frame, text="GPU device:").grid(row=15, column=0, sticky=tk.W, pady=5)
        ttk.Entry(main_frame, textvariable=self.gpu_device, width=50).grid(row=15, column=1, sticky=(tk.W, tk.E), padx=5)

        ttk.Label(main_frame, text="Precision:").grid(row=16, column=0, sticky=tk.W, pady=5)
        ttk.Combobox(main_frame, textvariable=self.precision, values=["fp16", "int8"], state="readonly", width=47).grid(row=16, column=1, sticky=(tk.W, tk.E), padx=5)

        ttk.Label(main_frame, text="Mode:").grid(row=17, column=0, sticky=tk.W, pady=20)
        ttk.Combobox(main_frame, textvariable=self.mode, values=["TRAIN & CONVERT", "TRAIN", "CONVERT"], state="readonly", width=47).grid(row=17, column=1, sticky=(tk.W, tk.E), padx=5, pady=20)

        # Run button and progress text area
        self.run_button = ttk.Button(main_frame, text="Run", command=self.start)
        self.run_button.grid(row=18, column=1, pady=5)

        ttk.Label(main_frame, text="Progress:").grid(row=19, column=0, sticky=(tk.W, tk.N), pady=5)

        text_frame = ttk.Frame(main_frame)
        text_frame.grid(row=19, column=1, columnspan=1, sticky=(tk.W, tk.E, tk.N, tk.S), pady=5)
        text_frame.columnconfigure(0, weight=1)
        text_frame.rowconfigure(0, weight=1)

        self.progress_text = tk.Text(text_frame, height=15, wrap=tk.WORD)
        scrollbar = ttk.Scrollbar(text_frame, orient=tk.VERTICAL, command=self.progress_text.yview)
        self.progress_text.configure(yscrollcommand=scrollbar.set)

        self.progress_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))

        main_frame.rowconfigure(19, weight=1)
        
    def browse_yolo_path(self):
        directory = filedialog.askdirectory(title="Select YOLOv8-TensorRT installation directory")
        if directory:
            self.yolo_path.set(directory)
            
    def browse_tensorrt_path(self):
        directory = filedialog.askdirectory(title="Select TensorRT installation directory")
        if directory:
            self.tensorrt_path.set(directory)

    def browse_data_folder(self):
        directory = filedialog.askdirectory(title="Select data folder")
        if directory:
            self.data_folder.set(directory)
    
    def browse_pt_file(self):
        file_path = filedialog.askopenfilename(title="Select .pt file", filetypes=[("PyTorch Model", "*.pt")])
        if file_path:
            self.pt_file_path.set(file_path)

    def browse_output_location(self):
        directory = filedialog.askdirectory(title="Select output directory")
        if directory:
            self.output_location.set(directory)
            
    def log_message(self, message):
        """Add message to progress text area"""
        self.progress_text.insert(tk.END, message + "\n")
        self.progress_text.see(tk.END)
        self.root.update_idletasks()
        
    def validate_inputs(self):
        """Validate all required inputs"""
        if not self.yolo_path.get() and self.mode.get() in ["TRAIN & CONVERT", "TRAIN"]:
            messagebox.showerror("Error", "Please select YOLOv8-TensorRT installation path")
            return False
        
        if not os.path.exists(self.yolo_path.get()) and self.mode.get() in ["TRAIN & CONVERT", "TRAIN"]:
            messagebox.showerror("Error", "YOLOv8-TensorRT installation path does not exist")
            return False
            
        if not self.tensorrt_path.get() and self.mode.get() in ["TRAIN & CONVERT", "CONVERT"]:
            messagebox.showerror("Error", "Please select TensorRT installation path")
            return False
            
        if not os.path.exists(self.tensorrt_path.get()) and self.mode.get() in ["TRAIN & CONVERT", "CONVERT"]:
            messagebox.showerror("Error", "TensorRT installation path does not exist")
            return False
        
        if not self.data_folder.get() and self.mode.get() in ["TRAIN", "TRAIN & CONVERT"]:
            messagebox.showerror("Error", "Please select a data folder")
            return False
        
        if not os.path.exists(self.data_folder.get()) and self.mode.get() in ["TRAIN", "TRAIN & CONVERT"]:
            messagebox.showerror("Error", "Data folder does not exist")
            return False
            
        if not self.output_name.get() and self.mode.get() in ["CONVERT", "TRAIN & CONVERT"]:
            messagebox.showerror("Error", "Please enter an output file name")
            return False
            
        if not self.output_location.get() and self.mode.get() in ["CONVERT", "TRAIN & CONVERT"]:
            messagebox.showerror("Error", "Please select an output location")
            return False
            
        if not os.path.exists(self.output_location.get()) and self.mode.get() in ["CONVERT", "TRAIN & CONVERT"]:
            messagebox.showerror("Error", "Output location does not exist")
            return False
        
        if not self.pt_file_path.get() and self.mode.get() == "CONVERT":
            messagebox.showerror("Error", "Please select a .pt file")
            return False

        if not os.path.exists(self.pt_file_path.get()) and self.mode.get() == "CONVERT":
            messagebox.showerror("Error", ".pt file location does not exist")
            return False
        
        if self.input_shape.get() != f"1 3 {self.imgsz.get()} {self.imgsz.get()}" and self.mode.get() == "TRAIN & CONVERT":
            messagebox.showerror("Error", "3rd and 4th values of Input shape must match the image size")
            return False
            
        return True
    
    def set_pt_path(self):
        """Get the path to the best.pt file from the latest training run"""
        runs_path = os.path.join(self.yolo_path.get(), "runs", "detect")
        if not os.path.exists(runs_path):
            messagebox.showerror("Error", "Runs path does not exist")
            return None

        new_run = "train"
        # Filter directories that start with "train" and sort by the numeric part
        sorted_dirs = sorted(
            [d for d in os.listdir(runs_path) if d.startswith("train") and d[5:].isdigit()],
            key=lambda d: int(d[5:])  # Extract and sort by the numeric part
        )

        if sorted_dirs:
            last_run = sorted_dirs[-1][5:]  # Get the numeric part of the last directory
            new_run = f"train{int(last_run) + 1}"
        else:
            new_run = "train2"

        weights_path = os.path.join(runs_path, new_run, "weights", "best.pt")
        self.pt_file_path.set(weights_path)
        
    def start(self):
        """Start the process in a separate thread"""
        if not self.validate_inputs():
            return
            
        self.run_button.config(state='disabled')
        
        self.progress_text.delete(1.0, tk.END)
        
        train_conversion_thread = threading.Thread(target=self.run)
        train_conversion_thread.daemon = True
        train_conversion_thread.start()
        
    def run(self):
        """Run the commands"""
        try:
            # Get the user's current shell
            user_shell = os.environ.get('SHELL', '/bin/sh')
            
            self.log_message("Starting...")
            self.log_message(f"Using shell: {user_shell}")
            self.log_message(f"YOLOv8-TensorRT path: {self.yolo_path.get()}")
            self.log_message(f"Model: {self.model.get()}")
            self.log_message(f"Data folder: {self.data_folder.get()}")
            self.log_message(f"Epochs: {self.epochs.get()}")
            self.log_message(f"Image size: {self.imgsz.get()}")
            self.log_message(f"TensorRT path: {self.tensorrt_path.get()}")
            self.log_message(f"Output name: {self.output_name.get()}")
            self.log_message(f"Output location: {self.output_location.get()}")
            self.log_message(f"IOU threshold(s): {self.iou_threshold.get().split(' ')}")
            self.log_message(f"Confidence threshold(s): {self.confidence_threshold.get().split(' ')}")
            self.log_message(f"Top K: {self.topk.get()}")
            self.log_message(f"Input shape: {self.input_shape.get()}")
            self.log_message(f"GPU device: {self.gpu_device.get()}")
            self.log_message(f"Precision: {self.precision.get()}")
            self.log_message("-" * 50)
            
            train_commands = [
                f"source {self.yolo_path.get()}/.venv/bin/activate",
                f"pip install ultralytics",
                f"yolo "
                f"task=detect "
                f"mode=train "
                f"model={self.model.get()} " 
                f"data={self.data_folder.get()}/data.yaml "
                f"epochs={self.epochs.get()} "
                f"imgsz={self.imgsz.get()} ",
            ]

            convert_commands = [
                f"source {os.getcwd()}/.venv/bin/activate",
                f"pip install -r {os.getcwd()}/requirements.txt",
            ]

            iou_thresholds = self.iou_threshold.get().split(" ")
            confidence_thresholds = self.confidence_threshold.get().split(" ")
            
            if self.mode.get() == "TRAIN & CONVERT":
                self.set_pt_path()

            for iou_thresh in iou_thresholds:
                for confidence_thresh in confidence_thresholds:
                    iou_conf_suffix= f"_iou{''.join(iou_thresh.split('.'))}_conf{''.join(confidence_thresh.split('.'))}"
                    convert_commands+= [
                        f"python3 {os.getcwd()}/export_det.py --weights {self.pt_file_path.get()} "
                        f"--iou-thres {iou_thresh} " 
                        f"--conf-thres {confidence_thresh} "
                        f"--topk {self.topk.get()} "
                        f"--input-shape {self.input_shape.get()} "
                        f"--device cuda:{self.gpu_device.get()} "
                        f"--opset 11 "
                        f"--sim ",
                        f"{self.tensorrt_path.get()}/trtexec --onnx={self.pt_file_path.get().replace('.pt', '.onnx')} "
                        f"--saveEngine={os.path.join(self.output_location.get(), self.output_name.get() + iou_conf_suffix + '.engine').replace('.pt', '')} "
                        f"--device={self.gpu_device.get()} "
                        f"--{self.precision.get()}" if self.precision.get() else "fp16"
                    ]


            if self.mode.get() == "TRAIN & CONVERT":
                commands = train_commands + convert_commands
            elif self.mode.get() == "TRAIN":
                commands = train_commands
            elif self.mode.get() == "CONVERT":
                commands = convert_commands

            if not commands:
                self.log_message("Please add your  commands to the run method")
                return
            
            for i, command in enumerate(commands, 1):
                self.log_message(f"Step {i}: Running command...")
                self.log_message(f"Command: {command}")
                
                process = subprocess.Popen(
                    command,
                    shell=True,
                    executable=user_shell,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    universal_newlines=True,
                    cwd=self.tensorrt_path.get()
                )
                
                for line in process.stdout:
                    self.log_message(line.strip())
                
                return_code = process.wait()
                
                if return_code != 0:
                    self.log_message(f"Error: Command failed with return code {return_code}")
                    return
                else:
                    self.log_message(f"Step {i} completed successfully")
                    
            self.log_message("-" * 50)
            self.log_message("Training and conversion(s) completed successfully!")
            
        except Exception as e:
            self.log_message(f"Error during training or conversion(s): {str(e)}")
            
        finally:
            self.root.after(0, lambda: self.run_button.config(state='normal'))

def main():
    root = tk.Tk()
    app = TensorRTConverterGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()