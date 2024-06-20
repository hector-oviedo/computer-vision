import json
import psutil
import torch
import os
from config import GlobalConfig  # Import GlobalConfig

class Logger:
    def __init__(self, filename):
        self.output_path = os.path.join(GlobalConfig.json_output_folder, f"{filename}.json")
        print(f"Logger initialized, output path: {self.output_path}")
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)  # Ensure the directory exists
        if os.path.exists(self.output_path):
            os.remove(self.output_path)
        self.log_data = {
            'device': GlobalConfig.device,
            'frames': [],
            'total_frames': 0
        }
        # Initialize log file
        with open(self.output_path, 'w') as f:
            json.dump(self.log_data, f, indent=4)

    def log_frame(self, frame_number, model_name, detections, inference_time):
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
        print(f"Frame {frame_number} logged: {frame_log}")
        
        # Write the log incrementally
        with open(self.output_path, 'r+') as f:
            data = json.load(f)
            data['frames'].append(frame_log)
            f.seek(0)
            json.dump(data, f, indent=4)

    def save_log(self):
        self.log_data['total_frames'] = len(self.log_data['frames'])
        with open(self.output_path, 'w') as f:
            json.dump(self.log_data, f, indent=4)
        print(f"Log saved to {self.output_path}")