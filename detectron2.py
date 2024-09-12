import os
import sys
import cv2
import torch
import time
import gc
import shutil
import numpy as np
from detectron2.utils.logger import setup_logger
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.utils.visualizer import Visualizer, ColorMode

# Add the utils directory to the Python path
script_dir = os.path.dirname(os.path.abspath(__file__))
utils_dir = os.path.join(script_dir, 'utils')
sys.path.append(utils_dir)

from config import GlobalConfig
from logger import Logger

# Set up Detectron2 logger
setup_logger()

class Detectron2Processor:
    def __init__(self, frames_input_folder, frames_output_folder, output_json):
        """Initialize the Detectron2Processor class.

        Args:
            frames_input_folder (str): Directory where the input frames are stored.
            frames_output_folder (str): Directory where the output frames will be saved.
            output_json (str): File path for saving the log data in JSON format.
        """

        # Name of the model where we will save the frames, and name of the output file
        self.model_name = "detectron2"

        self.frames_input_folder = frames_input_folder
        self.frames_output_folder = frames_output_folder
        self.logger = Logger(output_json)
        self.device = GlobalConfig.device
        self.cfg = self.initialize_model()
        self.predictor = DefaultPredictor(self.cfg)
        self.metadata = MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0])

    def initialize_model(self):
        """Initializes the Detectron2 model and configuration."""
        cfg = get_cfg()
        cfg.merge_from_file(model_zoo.get_config_file("COCO-PanopticSegmentation/panoptic_fpn_R_101_3x.yaml"))
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # Set confidence threshold for predictions
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-PanopticSegmentation/panoptic_fpn_R_101_3x.yaml")
        return cfg

    def process_frames(self):
        """Process the input frames, perform inference, and save results.

        This method reads frames from the input folder, processes them using Detectron2,
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
            processed_frame, frame_data = self.process_frame(frame)
            inference_time = time.time() - start_time

            # Accumulate inference time for overall calculation
            total_inference_time += inference_time

            # Save the annotated frame in the model-specific subfolder
            frame_filename = os.path.join(model_frames_output_dir, f'frame_{frame_id:04d}.png')
            cv2.imwrite(frame_filename, processed_frame)

            # Log the results for each frame, including inference time
            self.logger.log_frame(frame_id, self.model_name, frame_data['detections'], inference_time)

            # Update the frame counter
            frame_id += 1

            # Clear cache and free up memory after each frame
            del processed_frame, frame_data
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

    def process_frame(self, frame):
        """
        Runs inference on a single frame using Detectron2 and returns the annotated frame and detections.

        Args:
            frame (ndarray): The frame to process.

        Returns:
            tuple: Annotated frame and smoothed detections.
        """
        outputs = self.predictor(frame)
        instances = outputs["instances"].to("cpu")
        
        frame_data = {'detections': []}
        boxes = instances.pred_boxes.tensor.numpy()
        scores = instances.scores.numpy()
        classes = instances.pred_classes.numpy()

        # Collect detection data for logging
        for box, score, cls in zip(boxes, scores, classes):
            frame_data['detections'].append({
                'label': self.metadata.thing_classes[cls],
                'box': box.tolist(),
                'score': round(float(score), 2)
            })

        # Draw detections on the frame
        v = Visualizer(frame[:, :, ::-1], self.metadata, instance_mode=ColorMode.IMAGE)
        out = v.draw_instance_predictions(instances)
        result_frame = out.get_image()[:, :, ::-1]  # Convert back to BGR for OpenCV

        return result_frame, frame_data


if __name__ == "__main__":
    # Instantiate the processor class
    processor = Detectron2Processor(GlobalConfig.frames_input_folder, GlobalConfig.frames_output_folder, "")

    # Use configuration to specify input and output folders
    frames_input_folder = GlobalConfig.frames_input_folder
    frames_output_folder = GlobalConfig.frames_output_folder

    # Use processor.model_name directly since `self` isn't available in the main scope
    output_json = f'{processor.model_name}_results'

    processor = Detectron2Processor(frames_input_folder, frames_output_folder, output_json)
    processor.process_frames()
