import json
import psutil
import torch
import os
from config import GlobalConfig

class Logger:
    def __init__(self, model_name, model_identifier, model_parameters, model_official_site, verbose=False):
        """Initialize the Logger class for recording inference and system data.

        Args:
            model_name (str): The name of the model being used, also used for the log file name.
            model_identifier (str): Specific identifier for the model (e.g., version or architecture).
            model_parameters (str): Model parameters such as weight size or version number.
            model_official_site (str): URL of the model's official website.
            verbose (bool, optional): If True, logs will be printed to the console. Default is False.
        """
        if not model_name:
            raise ValueError("Logger filename cannot be empty.")

        self.verbose = verbose
        self.output_path = os.path.join(GlobalConfig.json_output_folder, f"{model_name}.json")

        # Print log path if verbose is enabled
        if self.verbose:
            print(f"Logger initialized, output path: {self.output_path}")

        # Ensure the output folder exists
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)

        # Remove any existing log file to start fresh
        if os.path.exists(self.output_path):
            os.remove(self.output_path)

        # Initialize log data structure
        self.log_data = {
            'device': GlobalConfig.device,
            'model': model_name,
            'model_identifier': model_identifier,
            'parameters': model_parameters,
            'official_site': model_official_site,
            'total_frames': 0,
            'total_inference_time_ms': 0.0,
            'total_time_ms': 0.0,
            'total_gpu_memory_used': 0.0,
            'total_gpu_memory_reserved': 0.0,
            'frames': []  # Frame-specific logs will be appended here
        }

        # Write the initial structure to the log file
        with open(self.output_path, 'w') as f:
            json.dump(self.log_data, f, indent=4)

    def log_frame(self, frame_number, detections, inference_time_ms, total_time_ms):
        """Logs information about a specific frame, including detections and system usage.

        Args:
            frame_number (int): Index of the frame being logged.
            detections (list): List of detected objects, each containing a label, bounding box, and score.
            inference_time_ms (float): Inference time for the frame in milliseconds.
            total_time_ms (float): Total time spent processing the frame in milliseconds.
        """
        # Prepare the frame-specific log
        frame_log = {
            'frame_number': frame_number,
            'inference_time_ms': round(inference_time_ms, 2),
            'total_time_ms': round(total_time_ms, 2),
            'cpu_usage': psutil.cpu_percent(),  # Current CPU usage percentage
            'cpu_ram_usage': psutil.virtual_memory().percent,  # Current RAM usage percentage
            'gpu_vram_usage': round(torch.cuda.memory_allocated() / 1e6, 2) if torch.cuda.is_available() else 0,  # In MB
            'gpu_vram_reserved': round(torch.cuda.memory_reserved() / 1e6, 2) if torch.cuda.is_available() else 0,  # In MB
            'detections': []
        }

        # Log all detections (label, bounding box, and score)
        for detection in detections:
            frame_log['detections'].append({
                'label': detection['label'],
                'box': [int(coord) for coord in detection['box']],  # Ensure the bounding box coordinates are integers
                'score': round(detection['score'], 2)  # Round score to 2 decimal places
            })

        # Append the frame log to the overall log data
        self.log_data['frames'].append(frame_log)

        # Update cumulative statistics
        self.log_data['total_inference_time_ms'] += inference_time_ms
        self.log_data['total_frames'] += 1

        # Optionally print frame log information if verbose mode is enabled
        if self.verbose:
            print(f"Frame {frame_number} logged: inference time {inference_time_ms:.2f}ms, total time {total_time_ms:.2f}ms")

    def set_total_time(self, total_time_ms):
        """Sets the total time spent on processing all frames.

        Args:
            total_time_ms (float): Total time for processing in milliseconds.
        """
        self.log_data['total_time_ms'] = round(total_time_ms, 2)

    def log_summary(self):
        """Finalize and log the summary information after all frames have been processed.

        This method logs the maximum GPU memory used during the processing if CUDA is available.
        """
        # Only log GPU memory metrics if a CUDA device is being used
        if GlobalConfig.device == 'cuda' and torch.cuda.is_available():
            self.log_data['total_gpu_memory_used'] = round(torch.cuda.max_memory_allocated() / 1e6, 2)  # MB
            self.log_data['total_gpu_memory_reserved'] = round(torch.cuda.max_memory_reserved() / 1e6, 2)  # MB

        # Round total inference time to two decimal places for consistency
        self.log_data['total_inference_time_ms'] = round(self.log_data['total_inference_time_ms'], 2)

        # Write the final log data to the JSON file
        with open(self.output_path, 'w') as f:
            json.dump(self.log_data, f, indent=4)

        print(f"Log saved to {self.output_path}")

    def save_log(self):
        """Save the log data by calling the log_summary method."""
        self.log_summary()
