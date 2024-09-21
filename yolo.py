import os
import sys
import cv2
import torch
import time
import gc
import shutil
import numpy as np
from ultralytics import YOLO

# Add the utils directory to the Python path for custom modules
script_dir = os.path.dirname(os.path.abspath(__file__))
utils_dir = os.path.join(script_dir, 'utils')
sys.path.append(utils_dir)

from config import GlobalConfig
from logger import Logger

class YOLOv8Processor:
    def __init__(self, frames_input_folder, frames_output_folder, model_name, model_identifier, model_parameters, model_official_site):
        """
        Initialize the YOLOv8Processor class for performing inference on video frames.

        Args:
            frames_input_folder (str): Path to the folder containing input frames.
            frames_output_folder (str): Path to the folder where output frames will be saved.
            model_name (str): The name of the model used for logging purposes.
            model_identifier (str): Unique identifier for the model (e.g., version or type).
            model_parameters (str): Information about model weights or parameters.
            model_official_site (str): URL to the official model site (e.g., documentation).
        """
        self.frames_input_folder = frames_input_folder
        self.frames_output_folder = frames_output_folder
        self.device = GlobalConfig.device  # Device configuration (CPU or GPU)
        
        # Load YOLOv8 model, specify the path to model weights
        self.model = YOLO('models/yolov8x-seg.pt').to(self.device)
        self.class_color_mapping = {}  # To map class IDs to colors for drawing

        # Model information for logging
        self.model_name = model_name
        self.model_identifier = model_identifier
        self.model_parameters = model_parameters
        self.model_official_site = model_official_site

        # Logger to track the processing and inference metrics
        self.logger = Logger(self.model_name, self.model_identifier, self.model_parameters, self.model_official_site)

    def process_frames(self):
        """
        Process all frames in the input folder, perform inference using the YOLOv8 model,
        and save the results in the output folder.
        """
        # Gather all PNG files from the input folder
        frame_files = sorted([f for f in os.listdir(self.frames_input_folder) if f.endswith('.png')])
        total_frames = len(frame_files)

        # If no frames are found, print an error message and exit
        if total_frames == 0:
            print("Error: No frames found in the specified folder.")
            return

        # Determine how many frames to process based on the percentage defined in config
        max_frames_to_process = int(total_frames * GlobalConfig.video_percentage)
        frame_id = 0

        # Define the output directory for processed frames
        model_frames_output_dir = os.path.join(self.frames_output_folder, self.model_name)

        # Clear the output folder if it already contains files
        if os.path.exists(model_frames_output_dir):
            if os.listdir(model_frames_output_dir):
                shutil.rmtree(model_frames_output_dir)
                print("Warning: Output folder had files which were cleaned.")
        os.makedirs(model_frames_output_dir, exist_ok=True)

        # Record the start time of the script
        script_start_time = time.perf_counter()

        # Process each frame
        for frame_file in frame_files[:max_frames_to_process]:
            frame_path = os.path.join(self.frames_input_folder, frame_file)
            frame = cv2.imread(frame_path)

            # If a frame can't be read, skip it
            if frame is None:
                print(f"Error: Could not read frame {frame_file}. Skipping.")
                continue

            # Log the processing progress
            print(f"Processing frame {frame_id + 1}/{max_frames_to_process}...")

            # Record frame start time for time metrics
            frame_start_time = time.perf_counter()

            # Resize frame to the required dimensions (defined in GlobalConfig)
            resized_frame = cv2.resize(frame, (GlobalConfig.resize_size, GlobalConfig.resize_size))

            # Perform inference using YOLOv8 model, suppress additional console logs with verbose=False
            results = self.model(resized_frame, device=self.device, verbose=False)

            # Extract inference time from YOLOv8's results object
            inference_time_ms = results[0].speed.get('inference', 0.0)  # Speed is returned in milliseconds

            # Annotate the frame with detection and segmentation results
            annotated_frame, frame_data = self.annotate_frame(frame, results)

            # Save the annotated frame to the output directory
            frame_filename = os.path.join(model_frames_output_dir, f'frame_{frame_id:04d}.png')
            cv2.imwrite(frame_filename, annotated_frame)

            # Record the frame end time to calculate total processing time
            frame_end_time = time.perf_counter()
            frame_total_time_ms = (frame_end_time - frame_start_time) * 1000  # Convert time to milliseconds

            # Log frame details using the Logger class
            self.logger.log_frame(frame_id, frame_data['results'], inference_time_ms, frame_total_time_ms)

            # Increment frame counter
            frame_id += 1

            # Clear cache and free memory to avoid memory buildup
            del resized_frame, results, annotated_frame, frame_data
            torch.cuda.empty_cache()
            gc.collect()

        # Record the script end time
        script_end_time = time.perf_counter()

        # Calculate the total script execution time
        total_script_time_ms = (script_end_time - script_start_time) * 1000  # Convert to milliseconds

        # Log the total time in the Logger
        self.logger.set_total_time(total_script_time_ms)

        # Save the final log summary
        self.logger.save_log()
        print("Processing complete. Frames and log saved.")

    def annotate_frame(self, frame, results_frame):
        """
        Annotate a frame with YOLOv8 detection and segmentation results.

        Args:
            frame (numpy.ndarray): The input frame (image) in BGR format.
            results_frame: The inference results from the YOLOv8 model.

        Returns:
            tuple: 
                - frame (numpy.ndarray): The annotated frame with bounding boxes and masks.
                - frame_data (dict): Information about the detections (label, box, score).
        """
        frame_data = {'results': []}  # To store the detection data
        overlay = frame.copy()  # Copy of the frame for overlaying segmentation masks

        # Iterate over each detection result in the frame
        for result in results_frame:
            boxes = result.boxes  # Bounding boxes for detected objects
            masks = result.masks  # Segmentation masks
            class_ids = result.boxes.cls  # Class IDs of detected objects
            scores = result.boxes.conf  # Confidence scores of detections

            # If there are masks, process them along with the boxes
            if masks is not None:
                for box, class_id, score, mask in zip(boxes.xyxy, class_ids, scores, masks.data):
                    # Get the class label from the model's class names
                    label = self.model.names[int(class_id)]
                    box = box.tolist()  # Convert tensor to list for drawing
                    mask = mask.cpu().numpy().astype(np.uint8)  # Convert mask to NumPy array

                    # Assign colors for bounding boxes and masks
                    if int(class_id) not in self.class_color_mapping:
                        self.class_color_mapping[int(class_id)] = {
                            'box_color': GlobalConfig.object_colors[len(self.class_color_mapping) % len(GlobalConfig.object_colors)],
                            'segmentation_color': GlobalConfig.segmentation_colors[len(self.class_color_mapping) % len(GlobalConfig.segmentation_colors)]
                        }

                    box_color = self.class_color_mapping[int(class_id)]['box_color']
                    segmentation_color = self.class_color_mapping[int(class_id)]['segmentation_color']

                    # Resize the mask to the original frame size
                    mask_resized = cv2.resize(mask, (frame.shape[1], frame.shape[0]))

                    # Create an overlay with the segmentation mask
                    color_mask = np.zeros_like(frame, dtype=np.uint8)
                    color_mask[mask_resized > 0] = segmentation_color
                    alpha_mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
                    alpha_mask[mask_resized > 0] = int(255 * GlobalConfig.instance_segmentation_transparency)

                    # Convert the color mask to RGBA and blend with the original frame
                    color_mask = cv2.cvtColor(color_mask, cv2.COLOR_RGB2RGBA)
                    color_mask[:, :, 3] = alpha_mask
                    frame_rgba = cv2.cvtColor(frame, cv2.COLOR_RGB2RGBA)
                    overlay = cv2.addWeighted(frame_rgba, 1, color_mask, 0.5, 0)
                    frame = cv2.cvtColor(overlay, cv2.COLOR_RGBA2RGB)

                    # Draw bounding box and label on the frame
                    cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), box_color, GlobalConfig.box_border_width)
                    label_text = f"{label} {score:.2f}"
                    (text_width, text_height), baseline = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                    cv2.rectangle(frame, (int(box[0]), int(box[1]) - text_height - baseline), 
                                  (int(box[0]) + text_width, int(box[1])), 
                                  GlobalConfig.label_box_background, -1)
                    cv2.putText(frame, label_text, (int(box[0]), int(box[1]) - baseline), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, GlobalConfig.label_text_color, 1, lineType=cv2.LINE_AA)

                    # Append detection information to the frame data
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

    # Model details for logging
    model_name = "yolo"
    model_identifier = "yolov8xseg"
    model_parameters = "71.8M"
    model_official_site = "https://docs.ultralytics.com/models/"

    # Instantiate the processor and start processing frames
    processor = YOLOv8Processor(frames_input_folder, frames_output_folder, model_name, model_identifier, model_parameters, model_official_site)
    processor.process_frames()
