# utils/logger.py

import json
import psutil
import torch
import os
from config import GlobalConfig  # Ensure this import points to your config module

class Logger:
    def __init__(self, filename, verbose=False):
        """Initialize the logger.

        Args:
            filename (str): The name of the JSON file for logging.
            verbose (bool): Whether to print logs to the console. Default is False.
        """
        if not filename:
            raise ValueError("Logger filename cannot be empty.")

        self.verbose = verbose
        self.output_path = os.path.join(GlobalConfig.json_output_folder, f"{filename}.json")

        if self.verbose:
            print(f"Logger initialized, output path: {self.output_path}")

        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)

        if os.path.exists(self.output_path):
            os.remove(self.output_path)

        # Infer model_name from filename
        model_name = os.path.splitext(filename)[0]

        # Initialize log_data with 'frames' at the end for better readability
        self.log_data = {
            'device': GlobalConfig.device,
            'model': model_name,  # Store model name at the root
            'total_frames': 0,
            'total_inference_time_ms': 0.0,
            'total_time_ms': 0.0,
            'total_gpu_memory_used': 0.0,
            'total_gpu_memory_cached': 0.0,
            'frames': []  # Place 'frames' at the end
        }

        # Write initial log data to file
        with open(self.output_path, 'w') as f:
            json.dump(self.log_data, f, indent=4)

    def log_frame(self, frame_number, model_name, detections, inference_time_ms, total_time_ms):
        """Log data for a specific frame.

        Args:
            frame_number (int): The frame number being logged.
            model_name (str): The name of the model used for inference.
            detections (list): List of detected objects in the frame.
            inference_time_ms (float): The inference time for the frame in milliseconds.
            total_time_ms (float): The total time for processing the frame in milliseconds.
        """
        frame_log = {
            'frame_number': frame_number,
            'inference_time_ms': round(inference_time_ms, 2),
            'total_time_ms': round(total_time_ms, 2),
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

        # Update internal state
        self.log_data['frames'].append(frame_log)
        self.log_data['total_inference_time_ms'] += inference_time_ms
        self.log_data['total_frames'] += 1

        if self.verbose:
            print(f"Frame {frame_number} logged with inference time {inference_time_ms:.2f}ms and total time {total_time_ms:.2f}ms.")

    def set_total_time(self, total_time_ms):
        """Set the total processing time at the root.

        Args:
            total_time_ms (float): The total processing time in milliseconds.
        """
        self.log_data['total_time_ms'] = round(total_time_ms, 2)

    def log_summary(self):
        """Log the summary data after processing all frames."""
        if GlobalConfig.device == 'cuda':
            self.log_data["total_gpu_memory_used"] = round(torch.cuda.max_memory_allocated() / 1e6, 2)  # MB
            self.log_data["total_gpu_memory_cached"] = round(torch.cuda.max_memory_reserved() / 1e6, 2)  # MB

        # Round the total_inference_time_ms to two decimal places
        self.log_data['total_inference_time_ms'] = round(self.log_data['total_inference_time_ms'], 2)

        # Write the complete log data to the file
        with open(self.output_path, 'w') as f:
            json.dump(self.log_data, f, indent=4)

        print(f"Log saved to {self.output_path}")

    def save_log(self):
        """Save the complete log to the output file."""
        self.log_summary()
