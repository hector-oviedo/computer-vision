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
        """Initialize the YOLOv8Processor class.

        Args:
            frames_input_folder (str): Directory where the input frames are stored.
            frames_output_folder (str): Directory where the output frames will be saved.
            output_json (str): File path for saving the log data in JSON format.
        """

        # Name of the model where we will save the frames, and name of the output file
        self.model_name = "yolov8xseg"

        self.frames_input_folder = frames_input_folder
        self.frames_output_folder = frames_output_folder
        self.logger = Logger(output_json)
        self.device = GlobalConfig.device
        self.model = YOLO('models/yolov8x-seg.pt').to(self.device)
        self.class_color_mapping = {}

    def process_frames(self):
        """Process the input frames, perform inference, and save results.

        This method reads frames from the input folder, processes them using YOLOv8,
        logs the inference time per frame, and saves the annotated frames in the output folder.
        """
        # Get all frame files from the input folder and sort them
        frame_files = sorted([f for f in os.listdir(self.frames_input_folder) if f.endswith('.png')])
        total_frames = len(frame_files)

        # If no frames are found, exit
        if total_frames == 0:
            print("Error: No frames found in the specified folder.")
            return

        # Determine the number of frames to process based on the video_percentage setting
        max_frames_to_process = int(total_frames * GlobalConfig.video_percentage)

        # Initialize frame counter and start the total time tracking
        total_inference_time = 0  # Track cumulative inference time
        frame_id = 0

        # Ensure the model-specific subfolder exists within the frames output folder
        model_frames_output_dir = os.path.join(self.frames_output_folder, self.model_name)

        # Clear the folder if it contains any files
        if os.path.exists(model_frames_output_dir):
            if os.listdir(model_frames_output_dir):  # Check if the folder is not empty
                # Clear the folder
                shutil.rmtree(model_frames_output_dir)
                print("Warning: Output folder had files which were cleaned.")
        os.makedirs(model_frames_output_dir, exist_ok=True)

        # Process each frame within the specified percentage
        for frame_file in frame_files[:max_frames_to_process]:
            frame_path = os.path.join(self.frames_input_folder, frame_file)
            frame = cv2.imread(frame_path)

            # Ensure the frame is readable, if not, skip it
            if frame is None:
                print(f"Error: Could not read frame {frame_file}.")
                continue

            # Log progress as `frame_id / total_frames_to_process`
            print(f"Processing frame {frame_id + 1}/{max_frames_to_process}...")

            # Measure inference time for each frame
            start_time = time.time()
            resized_frame = cv2.resize(frame, (GlobalConfig.resize_size, GlobalConfig.resize_size))
            results_frame = self.model(resized_frame, device=self.device)
            inference_time = time.time() - start_time

            # Accumulate inference time for overall calculation
            total_inference_time += inference_time

            # Annotate the frame based on YOLO inference results
            annotated_frame, frame_data = self.annotate_frame(frame, results_frame)

            # Save the annotated frame in the model-specific subfolder
            frame_filename = os.path.join(model_frames_output_dir, f'frame_{frame_id:04d}.png')
            cv2.imwrite(frame_filename, annotated_frame)

            # Log the results for each frame, including inference time
            self.logger.log_frame(frame_id, self.model_name, frame_data['results'], inference_time)

            # Update the frame counter
            frame_id += 1

            # Clear cache and free up memory after each frame
            del resized_frame, results_frame
            torch.cuda.empty_cache()
            gc.collect()

        # After processing all frames, log the total inference time and other data
        self.logger.log_data["total_inference_time"] = round(total_inference_time, 3)
        self.logger.log_data["total_frames"] = frame_id
        if self.device == 'cuda':
            self.logger.log_data["total_gpu_memory_used"] = round(torch.cuda.max_memory_allocated() / 1e6, 2)  # MB
            self.logger.log_data["total_gpu_memory_cached"] = round(torch.cuda.max_memory_reserved() / 1e6, 2)  # MB

        # Save the final log data
        self.logger.save_log()
        print("Processing complete. Frames and log saved.")

    def annotate_frame(self, frame, results_frame):
        """Annotates the input frame with YOLO detection and segmentation results.

        Args:
            frame (ndarray): The original input frame.
            results_frame (Results): The inference results from YOLOv8.

        Returns:
            tuple: The annotated frame and the data to be logged.
        """
        frame_data = {'results': []}
        overlay = frame.copy()  # Copy the original frame for overlaying the annotations

        # Loop through all detection results in the frame
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

                    # Assign color for class segmentation and bounding boxes
                    if int(class_id) not in self.class_color_mapping:
                        self.class_color_mapping[int(class_id)] = {
                            'box_color': GlobalConfig.object_colors[len(self.class_color_mapping) % len(GlobalConfig.object_colors)],
                            'segmentation_color': GlobalConfig.segmentation_colors[len(self.class_color_mapping) % len(GlobalConfig.segmentation_colors)]
                        }
                    
                    box_color = self.class_color_mapping[int(class_id)]['box_color']
                    segmentation_color = self.class_color_mapping[int(class_id)]['segmentation_color']

                    # Resize mask and apply segmentation and bounding boxes
                    mask_resized = cv2.resize(mask, (frame.shape[1], frame.shape[0]))
                    color_mask = np.zeros_like(frame, dtype=np.uint8)
                    color_mask[mask_resized > 0] = segmentation_color
                    alpha_mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
                    alpha_mask[mask_resized > 0] = int(255 * GlobalConfig.instance_segmentation_transparency)

                    # Overlay the segmentation and bounding box on the frame
                    color_mask = cv2.cvtColor(color_mask, cv2.COLOR_RGB2RGBA)
                    color_mask[:, :, 3] = alpha_mask
                    frame_rgba = cv2.cvtColor(frame, cv2.COLOR_RGB2RGBA)
                    overlay = cv2.addWeighted(frame_rgba, 1, color_mask, 0.5, 0)
                    frame = cv2.cvtColor(overlay, cv2.COLOR_RGBA2RGB)

                    # Draw the bounding box and label
                    cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), box_color, GlobalConfig.box_border_width)
                    label_text = f"{label} {score:.2f}"
                    (text_width, text_height), baseline = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                    cv2.rectangle(frame, (int(box[0]), int(box[1]) - text_height - baseline), (int(box[0]) + text_width, int(box[1])), GlobalConfig.label_box_background, -1)
                    cv2.putText(frame, label_text, (int(box[0]), int(box[1]) - baseline), cv2.FONT_HERSHEY_SIMPLEX, 0.5, GlobalConfig.label_text_color, 1, lineType=cv2.LINE_AA)

                    # Store the result data for logging
                    frame_data['results'].append({
                        'label': label,
                        'box': box,
                        'score': float(score)
                    })

        return frame, frame_data


if __name__ == "__main__":
    # Instantiate the processor class
    processor = YOLOv8Processor(GlobalConfig.frames_input_folder, GlobalConfig.frames_output_folder, "")

    # Use configuration to specify input and output folders
    frames_input_folder = GlobalConfig.frames_input_folder
    frames_output_folder = GlobalConfig.frames_output_folder

    # Use processor.model_name directly since `self` isn't available in the main scope
    output_json =  f'{processor.model_name}_results'

    processor = YOLOv8Processor(frames_input_folder, frames_output_folder, output_json)
    processor.process_frames()
