import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import subprocess
import threading
import os
from pathlib import Path

class TensorRTConverterGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("mechanic")
        self.root.geometry("800x700")
        
        self.pt_file_path = tk.StringVar()
        self.tensorrt_path = tk.StringVar()
        self.output_name = tk.StringVar()
        self.output_location = tk.StringVar()
        
        self.iou_threshold = tk.StringVar(value="0.5")
        self.confidence_threshold = tk.StringVar(value="0.25")
        self.topk = tk.StringVar(value="1")
        self.input_shape = tk.StringVar(value="1 3 640 640")
        self.gpu_device = tk.StringVar(value="0")
        self.precision = tk.StringVar(value="fp32")
        
        self.setup_gui()
        
    def setup_gui(self):
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        
        ttk.Label(main_frame, text="Select .pt file:").grid(row=0, column=0, sticky=tk.W, pady=5)
        ttk.Entry(main_frame, textvariable=self.pt_file_path, width=50).grid(row=0, column=1, sticky=(tk.W, tk.E), padx=5)
        ttk.Button(main_frame, text="Browse", command=self.browse_pt_file).grid(row=0, column=2, padx=5)
        
        ttk.Label(main_frame, text="TensorRT install location:").grid(row=1, column=0, sticky=tk.W, pady=5)
        ttk.Entry(main_frame, textvariable=self.tensorrt_path, width=50).grid(row=1, column=1, sticky=(tk.W, tk.E), padx=5)
        ttk.Button(main_frame, text="Browse", command=self.browse_tensorrt_path).grid(row=1, column=2, padx=5)
        
        ttk.Label(main_frame, text="Output file name:").grid(row=2, column=0, sticky=tk.W, pady=5)
        ttk.Entry(main_frame, textvariable=self.output_name, width=50).grid(row=2, column=1, sticky=(tk.W, tk.E), padx=5)
        
        ttk.Label(main_frame, text="Output location:").grid(row=3, column=0, sticky=tk.W, pady=5)
        ttk.Entry(main_frame, textvariable=self.output_location, width=50).grid(row=3, column=1, sticky=(tk.W, tk.E), padx=5)
        ttk.Button(main_frame, text="Browse", command=self.browse_output_location).grid(row=3, column=2, padx=5)
        
        ttk.Label(main_frame, text="IOU threshold:").grid(row=4, column=0, sticky=tk.W, pady=5)
        ttk.Entry(main_frame, textvariable=self.iou_threshold, width=50).grid(row=4, column=1, sticky=(tk.W, tk.E), padx=5)
        
        ttk.Label(main_frame, text="Confidence threshold:").grid(row=5, column=0, sticky=tk.W, pady=5)
        ttk.Entry(main_frame, textvariable=self.confidence_threshold, width=50).grid(row=5, column=1, sticky=(tk.W, tk.E), padx=5)
        
        ttk.Label(main_frame, text="Maximum bounding boxes:").grid(row=6, column=0, sticky=tk.W, pady=5)
        ttk.Entry(main_frame, textvariable=self.topk, width=50).grid(row=6, column=1, sticky=(tk.W, tk.E), padx=5)
        
        ttk.Label(main_frame, text="Input shape (HxWxC):").grid(row=7, column=0, sticky=tk.W, pady=5)
        ttk.Entry(main_frame, textvariable=self.input_shape, width=50).grid(row=7, column=1, sticky=(tk.W, tk.E), padx=5)
        
        ttk.Label(main_frame, text="GPU device:").grid(row=8, column=0, sticky=tk.W, pady=5)
        ttk.Entry(main_frame, textvariable=self.gpu_device, width=50).grid(row=8, column=1, sticky=(tk.W, tk.E), padx=5)
        
        ttk.Label(main_frame, text="Precision:").grid(row=9, column=0, sticky=tk.W, pady=5)
        ttk.Combobox(main_frame, textvariable=self.precision, values=["int8", "fp16", "fp32"], state="readonly", width=47).grid(row=9, column=1, sticky=(tk.W, tk.E), padx=5)
        
        self.convert_button = ttk.Button(main_frame, text="Convert", command=self.start_conversion)
        self.convert_button.grid(row=10, column=1, pady=20)
        
        ttk.Label(main_frame, text="Progress:").grid(row=11, column=0, sticky=(tk.W, tk.N), pady=5)
        
        text_frame = ttk.Frame(main_frame)
        text_frame.grid(row=11, column=1, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), pady=5)
        text_frame.columnconfigure(0, weight=1)
        text_frame.rowconfigure(0, weight=1)
        
        self.progress_text = tk.Text(text_frame, height=15, wrap=tk.WORD)
        scrollbar = ttk.Scrollbar(text_frame, orient=tk.VERTICAL, command=self.progress_text.yview)
        self.progress_text.configure(yscrollcommand=scrollbar.set)
        
        self.progress_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        
        main_frame.rowconfigure(11, weight=1)
        
    def browse_pt_file(self):
        filename = filedialog.askopenfilename(
            title="Select .pt file",
            filetypes=[("PyTorch files", "*.pt"), ("All files", "*.*")]
        )
        if filename:
            self.pt_file_path.set(filename)
            
    def browse_tensorrt_path(self):
        directory = filedialog.askdirectory(title="Select TensorRT installation directory")
        if directory:
            self.tensorrt_path.set(directory)
            
    def browse_output_location(self):
        directory = filedialog.askdirectory(title="Select output directory")
        if directory:
            self.output_location.set(directory)
            
    def log_message(self, message):
        """Add message to progress text area"""
        self.progress_text.insert(tk.END, message + "\n")
        self.progress_text.see(tk.END)
        self.root.update_idletasks()
        
    def parse_threshold_values(self, threshold_string):
        """Parse comma-separated threshold values"""
        values = []
        for value in threshold_string.split(','):
            value = value.strip()
            if value:
                try:
                    values.append(float(value))
                except ValueError:
                    raise ValueError(f"Invalid threshold value: {value}")
        return values
    
    def generate_filename(self, base_name, conf_threshold, iou_threshold):
        """Generate filename with confidence and IOU threshold suffixes"""
        name_without_ext = os.path.splitext(base_name)[0]
        extension = os.path.splitext(base_name)[1] if os.path.splitext(base_name)[1] else '.engine'
        
        return f"{name_without_ext}_conf{conf_threshold}_iou{iou_threshold}{extension}"
        
    def validate_inputs(self):
        """Validate all required inputs"""
        if not self.pt_file_path.get():
            messagebox.showerror("Error", "Please select a .pt file")
            return False
            
        if not os.path.exists(self.pt_file_path.get()):
            messagebox.showerror("Error", "Selected .pt file does not exist")
            return False
            
        if not self.tensorrt_path.get():
            messagebox.showerror("Error", "Please select TensorRT installation path")
            return False
            
        if not os.path.exists(self.tensorrt_path.get()):
            messagebox.showerror("Error", "TensorRT installation path does not exist")
            return False
            
        if not self.output_name.get():
            messagebox.showerror("Error", "Please enter an output file name")
            return False
            
        if not self.output_location.get():
            messagebox.showerror("Error", "Please select an output location")
            return False
            
        if not os.path.exists(self.output_location.get()):
            messagebox.showerror("Error", "Output location does not exist")
            return False
            
        return True
        
    def start_conversion(self):
        """Start the conversion process in a separate thread"""
        if not self.validate_inputs():
            return
            
        self.convert_button.config(state='disabled')
        
        self.progress_text.delete(1.0, tk.END)
        
        conversion_thread = threading.Thread(target=self.run_conversion)
        conversion_thread.daemon = True
        conversion_thread.start()
        
    def run_conversion(self):
        """Run the conversion commands"""
        try:
            self.log_message("Starting TensorRT conversion...")
            self.log_message(f"Input file: {self.pt_file_path.get()}")
            self.log_message(f"TensorRT path: {self.tensorrt_path.get()}")
            self.log_message(f"Output name: {self.output_name.get()}")
            self.log_message(f"Output location: {self.output_location.get()}")
            self.log_message(f"IOU threshold: {self.iou_threshold.get()}")
            self.log_message(f"Confidence threshold: {self.confidence_threshold.get()}")
            self.log_message(f"Top K: {self.topk.get()}")
            self.log_message(f"Input shape: {self.input_shape.get()}")
            self.log_message(f"GPU device: {self.gpu_device.get()}")
            self.log_message(f"Precision: {self.precision.get()}")
            self.log_message("-" * 50)
            
            # TODO: add commands
            commands = [
                # f". {os.getcwd()}/venv/bin/activate",
                f"pip install -r {os.getcwd()}/requirements.txt",
                f"python {os.getcwd()}/export_det.py --weights {self.pt_file_path.get()} "
                f"--iou-thres {self.iou_threshold.get()} " 
                f"--conf-thres {self.confidence_threshold.get()} "
                f"--topk {self.topk.get()} "
                f"--input-shape {self.input_shape.get()} "
                f"--device cuda:{self.gpu_device.get()} "
                f"--opset 11 "
                f"--sim ",
                f"{self.tensorrt_path.get()}/trtexec --onnx={self.pt_file_path.get().replace('.pt', '.onnx')} "
                f"--saveEngine={os.path.join(self.output_location.get(), self.output_name.get()).replace('.pt', '.engine')} "
                f"--device={self.gpu_device.get()} "
                f"--{self.precision.get()}" if self.precision.get() else "fp32"
            ]
            
            if not commands:
                self.log_message("Please add your conversion commands to the run_conversion method")
                return
            
            for i, command in enumerate(commands, 1):
                self.log_message(f"Step {i}: Running command...")
                self.log_message(f"Command: {command}")
                
                process = subprocess.Popen(
                    command,
                    shell=True,
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
            self.log_message("Conversion completed successfully!")
            
        except Exception as e:
            self.log_message(f"Error during conversion: {str(e)}")
            
        finally:
            self.root.after(0, lambda: self.convert_button.config(state='normal'))

def main():
    root = tk.Tk()
    app = TensorRTConverterGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()