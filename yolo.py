# yolo.py

import os
import sys
import cv2
import torch
import time
import gc
import shutil
import numpy as np
from ultralytics import YOLO

# Add the utils directory to the Python path
script_dir = os.path.dirname(os.path.abspath(__file__))
utils_dir = os.path.join(script_dir, 'utils')
sys.path.append(utils_dir)

from config import GlobalConfig
from logger import Logger

class YOLOv8Processor:
    def __init__(self, frames_input_folder, frames_output_folder, output_json):
        """Initialize the YOLOv8Processor class."""
        self.model_name = "yolov8xseg"
        self.frames_input_folder = frames_input_folder
        self.frames_output_folder = frames_output_folder
        self.logger = Logger(output_json)
        self.device = GlobalConfig.device
        self.model = YOLO('models/yolov8x-seg.pt').to(self.device)
        self.class_color_mapping = {}

    def process_frames(self):
        """Process the input frames, perform inference, and save results."""
        frame_files = sorted([f for f in os.listdir(self.frames_input_folder) if f.endswith('.png')])
        total_frames = len(frame_files)

        if total_frames == 0:
            print("Error: No frames found in the specified folder.")
            return

        max_frames_to_process = int(total_frames * GlobalConfig.video_percentage)
        frame_id = 0

        model_frames_output_dir = os.path.join(self.frames_output_folder, self.model_name)

        if os.path.exists(model_frames_output_dir):
            if os.listdir(model_frames_output_dir):
                shutil.rmtree(model_frames_output_dir)
                print("Warning: Output folder had files which were cleaned.")
        os.makedirs(model_frames_output_dir, exist_ok=True)

        # Record script start time
        script_start_time = time.perf_counter()

        for frame_file in frame_files[:max_frames_to_process]:
            frame_path = os.path.join(self.frames_input_folder, frame_file)
            frame = cv2.imread(frame_path)

            if frame is None:
                print(f"Error: Could not read frame {frame_file}. Skipping.")
                continue

            print(f"Processing frame {frame_id + 1}/{max_frames_to_process}...")

            # Record frame start time
            frame_start_time = time.perf_counter()

            # Resize frame outside the timing
            resized_frame = cv2.resize(frame, (GlobalConfig.resize_size, GlobalConfig.resize_size))

            # Perform inference and capture the results
            # The 'verbose=False' suppresses console logs
            results = self.model(resized_frame, device=self.device, verbose=False)

            # Extract inference time from the results
            # Assuming YOLOv8's Results object has a 'speed' dictionary with 'inference' time in milliseconds
            inference_time_ms = results[0].speed.get('inference', 0.0)  # Already in ms

            # Annotate the frame
            annotated_frame, frame_data = self.annotate_frame(frame, results)

            # Save the annotated frame
            frame_filename = os.path.join(model_frames_output_dir, f'frame_{frame_id:04d}.png')
            cv2.imwrite(frame_filename, annotated_frame)

            # Record frame end time
            frame_end_time = time.perf_counter()

            # Compute total time for frame processing in ms
            frame_total_time_ms = (frame_end_time - frame_start_time) * 1000  # ms

            # Log the results
            self.logger.log_frame(frame_id, self.model_name, frame_data['results'], inference_time_ms, frame_total_time_ms)

            frame_id += 1

            # Clear cache and free memory
            del resized_frame, results, annotated_frame, frame_data
            torch.cuda.empty_cache()
            gc.collect()

        # Record script end time
        script_end_time = time.perf_counter()

        # Compute total script time in ms
        total_script_time_ms = (script_end_time - script_start_time) * 1000  # ms

        # Set total_time in logger
        self.logger.set_total_time(total_script_time_ms)

        # Log summary data
        self.logger.save_log()
        print("Processing complete. Frames and log saved.")

    def annotate_frame(self, frame, results_frame):
        """Annotates the input frame with YOLO detection and segmentation results."""
        frame_data = {'results': []}
        overlay = frame.copy()

        for result in results_frame:
            boxes = result.boxes
            masks = result.masks
            class_ids = result.boxes.cls
            scores = result.boxes.conf

            if masks is not None:
                for box, class_id, score, mask in zip(boxes.xyxy, class_ids, scores, masks.data):
                    label = self.model.names[int(class_id)]
                    box = box.tolist()
                    mask = mask.cpu().numpy().astype(np.uint8)

                    if int(class_id) not in self.class_color_mapping:
                        self.class_color_mapping[int(class_id)] = {
                            'box_color': GlobalConfig.object_colors[len(self.class_color_mapping) % len(GlobalConfig.object_colors)],
                            'segmentation_color': GlobalConfig.segmentation_colors[len(self.class_color_mapping) % len(GlobalConfig.segmentation_colors)]
                        }

                    box_color = self.class_color_mapping[int(class_id)]['box_color']
                    segmentation_color = self.class_color_mapping[int(class_id)]['segmentation_color']

                    mask_resized = cv2.resize(mask, (frame.shape[1], frame.shape[0]))
                    color_mask = np.zeros_like(frame, dtype=np.uint8)
                    color_mask[mask_resized > 0] = segmentation_color
                    alpha_mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
                    alpha_mask[mask_resized > 0] = int(255 * GlobalConfig.instance_segmentation_transparency)

                    color_mask = cv2.cvtColor(color_mask, cv2.COLOR_RGB2RGBA)
                    color_mask[:, :, 3] = alpha_mask
                    frame_rgba = cv2.cvtColor(frame, cv2.COLOR_RGB2RGBA)
                    overlay = cv2.addWeighted(frame_rgba, 1, color_mask, 0.5, 0)
                    frame = cv2.cvtColor(overlay, cv2.COLOR_RGBA2RGB)

                    cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), box_color, GlobalConfig.box_border_width)
                    label_text = f"{label} {score:.2f}"
                    (text_width, text_height), baseline = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                    cv2.rectangle(frame, (int(box[0]), int(box[1]) - text_height - baseline), 
                                  (int(box[0]) + text_width, int(box[1])), 
                                  GlobalConfig.label_box_background, -1)
                    cv2.putText(frame, label_text, (int(box[0]), int(box[1]) - baseline), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, GlobalConfig.label_text_color, 1, lineType=cv2.LINE_AA)

                    frame_data['results'].append({
                        'label': label,
                        'box': [int(coord) for coord in box],
                        'score': float(score)
                    })

        return frame, frame_data

if __name__ == "__main__":
    # Use configuration to specify input and output folders
    frames_input_folder = GlobalConfig.frames_input_folder
    frames_output_folder = GlobalConfig.frames_output_folder

    # Initialize processor with output_json as model_name
    output_json = "yolov8xseg"

    processor = YOLOv8Processor(frames_input_folder, frames_output_folder, output_json)
    processor.process_frames()
