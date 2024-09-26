# utils/range_video_generator.py

import os
import shutil
import json
from config import GlobalConfig

def copy_and_rename_frames(src_dir, dest_dir, init, ending):
    """
    Copies and renames frames from src_dir to dest_dir within the specified range.
    """
    os.makedirs(dest_dir, exist_ok=True)
    frame_numbers = range(init, ending + 1)
    for idx, n in enumerate(frame_numbers, start=0):
        src_frame = os.path.join(src_dir, f"frame_{n:04d}.png")
        dest_frame = os.path.join(dest_dir, f"frame_{idx:04d}.png")
        if os.path.exists(src_frame):
            shutil.copyfile(src_frame, dest_frame)
        else:
            print(f"Warning: {src_frame} does not exist.")

def process_json_file(src_json_path, dest_json_path, init, ending):
    """
    Processes the JSON file to include only frames within the specified range
    and adjusts the frame numbers accordingly. Also recalculates summary statistics.
    """
    with open(src_json_path, 'r') as f:
        data = json.load(f)
    
    # Adjust total_frames
    total_frames = ending - init + 1
    data['total_frames'] = total_frames

    # Initialize accumulators for totals
    total_inference_time_ms = 0.0
    total_time_ms = 0.0
    gpu_memory_used_list = []
    gpu_memory_reserved_list = []

    # Adjust frames array
    new_frames = []
    for frame in data['frames']:
        frame_number = frame['frame_number']
        # Assuming frame_number starts from 0 in JSON
        if init <= frame_number <= ending:
            # Adjust frame_number to start from 0
            frame['frame_number'] = frame_number - init
            new_frames.append(frame)
            
            # Accumulate totals
            total_inference_time_ms += frame.get('inference_time_ms', 0.0)
            total_time_ms += frame.get('total_time_ms', 0.0)
            gpu_memory_used_list.append(frame.get('gpu_vram_usage', 0.0))
            gpu_memory_reserved_list.append(frame.get('gpu_vram_reserved', 0.0))
    
    data['frames'] = new_frames

    # Update total inference time and total time
    data['total_inference_time_ms'] = round(total_inference_time_ms, 2)
    data['total_time_ms'] = round(total_time_ms, 2)

    # Update total GPU memory used and reserved
    # Assuming we want the maximum values observed in the range
    data['total_gpu_memory_used'] = max(gpu_memory_used_list) if gpu_memory_used_list else 0.0
    data['total_gpu_memory_reserved'] = max(gpu_memory_reserved_list) if gpu_memory_reserved_list else 0.0

    # Save the modified JSON
    with open(dest_json_path, 'w') as f:
        json.dump(data, f, indent=4)

def main():
    # Constants (Set your desired values here)
    DEST_FOLDER_NAME = "video_4"             # Name of the destination folder
    INIT_FRAME = 2087                        # Starting frame number (inclusive)
    ENDING_FRAME = 2656                      # Ending frame number (inclusive)

    # 1 video 0 to 820
    # 2 video 821 to 1342
    # 3 video 1343 to 2086
    # 4 video 2087 to 2656

    # Paths from GlobalConfig
    frames_input_folder = GlobalConfig.frames_input_folder         # 'data/frames/'
    frames_output_folder = GlobalConfig.frames_output_folder       # 'output/frames/'
    json_output_folder = GlobalConfig.json_output_folder           # 'output/logs/'
    
    # Get project root directory (one level up from utils/)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, '..'))

    # Paths
    src_project_dir = project_root
    dest_project_dir = os.path.join(project_root, DEST_FOLDER_NAME)
    
    # Copy and rename frames in data/frames
    src_frames_dir = os.path.join(src_project_dir, frames_input_folder)
    dest_frames_dir = os.path.join(dest_project_dir, frames_input_folder)
    copy_and_rename_frames(src_frames_dir, dest_frames_dir, INIT_FRAME, ENDING_FRAME)
    
    # Process each model in output/frames
    src_output_frames_dir = os.path.join(src_project_dir, frames_output_folder)
    dest_output_frames_dir = os.path.join(dest_project_dir, frames_output_folder)
    
    if not os.path.exists(src_output_frames_dir):
        print("No output frames directory found.")
    else:
        for model_name in os.listdir(src_output_frames_dir):
            model_dir = os.path.join(src_output_frames_dir, model_name)
            if os.path.isdir(model_dir):
                # Copy and rename model frames
                src_model_frames_dir = model_dir
                dest_model_frames_dir = os.path.join(dest_output_frames_dir, model_name)
                copy_and_rename_frames(src_model_frames_dir, dest_model_frames_dir, INIT_FRAME, ENDING_FRAME)
    
    # Process logs
    src_logs_dir = os.path.join(src_project_dir, json_output_folder)
    dest_logs_dir = os.path.join(dest_project_dir, json_output_folder)
    os.makedirs(dest_logs_dir, exist_ok=True)
    
    if not os.path.exists(src_logs_dir):
        print("No logs directory found.")
    else:
        for json_file in os.listdir(src_logs_dir):
            if json_file.endswith(".json"):
                src_json_path = os.path.join(src_logs_dir, json_file)
                dest_json_path = os.path.join(dest_logs_dir, json_file)
                process_json_file(src_json_path, dest_json_path, INIT_FRAME, ENDING_FRAME)

if __name__ == "__main__":
    main()
