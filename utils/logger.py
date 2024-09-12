import json
import psutil
import torch
import os
from config import GlobalConfig  # Import GlobalConfig

class Logger:
    def __init__(self, filename, verbose=False):
        """Initialize the logger.

        Args:
            filename (str): The name of the JSON file for logging.
            verbose (bool): Whether to print logs to the console. Default is True.
        """
        self.verbose = verbose  # Set the verbose mode
        self.output_path = os.path.join(GlobalConfig.json_output_folder, f"{filename}.json")
        
        # Print the initialization message only if verbose is True
        if self.verbose:
            print(f"Logger initialized, output path: {self.output_path}")

        # Ensure the output directory exists
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
        
        # Remove the old log file if it exists
        if os.path.exists(self.output_path):
            os.remove(self.output_path)
        
        # Initialize log data
        self.log_data = {
            'device': GlobalConfig.device,
            'frames': [],
            'total_frames': 0
        }
        
        # Write initial log data to file
        with open(self.output_path, 'w') as f:
            json.dump(self.log_data, f, indent=4)

    def log_frame(self, frame_number, model_name, detections, inference_time):
        """Log data for a specific frame.

        Args:
            frame_number (int): The frame number being logged.
            model_name (str): The name of the model used for inference.
            detections (list): List of detected objects in the frame.
            inference_time (float): The inference time for the frame.
        """
        frame_log = {
            'frame_number': frame_number,
            'model': model_name,
            'inference_time': round(inference_time, 3),
            'cpu_usage': psutil.cpu_percent(),
            'cpu_ram_usage': psutil.virtual_memory().percent,
            'gpu_vram_usage': round(torch.cuda.memory_allocated() / 1e6, 2) if torch.cuda.is_available() else 0,
            'gpu_vram_reserved': round(torch.cuda.memory_reserved() / 1e6, 2) if torch.cuda.is_available() else 0,
            'detections': []
        }
        
        for detection in detections:
            frame_log['detections'].append({
                'label': detection['label'],
                'box': [int(coord) for coord in detection['box']],
                'score': round(detection['score'], 2)
            })

        self.log_data['frames'].append(frame_log)
        
        # Only print the log if verbose mode is enabled
        if self.verbose:
            print(f"Frame {frame_number} logged: {frame_log}")

        # Write the log incrementally to the file
        with open(self.output_path, 'r+') as f:
            data = json.load(f)
            data['frames'].append(frame_log)
            f.seek(0)
            json.dump(data, f, indent=4)

    def save_log(self):
        """Save the complete log to the output file."""
        self.log_data['total_frames'] = len(self.log_data['frames'])
        
        # Save the log data to the output file
        with open(self.output_path, 'w') as f:
            json.dump(self.log_data, f, indent=4)
        
        # Always print the final log saved message, even if verbose is False
        print(f"Log saved to {self.output_path}")
