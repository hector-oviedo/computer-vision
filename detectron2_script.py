import os
import sys
import cv2
import torch
import time
import gc
import shutil
from detectron2.utils.logger import setup_logger
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.utils.visualizer import Visualizer, ColorMode

# Add the utils directory to the Python path for custom imports
script_dir = os.path.dirname(os.path.abspath(__file__))
utils_dir = os.path.join(script_dir, 'utils')
sys.path.append(utils_dir)

from config import GlobalConfig
from logger import Logger

# Set up Detectron2 logger to suppress unnecessary logs
setup_logger()

class Detectron2Processor:
    def __init__(self, frames_input_folder, frames_output_folder, model_name, model_identifier, model_parameters, model_official_site):
        """
        Initializes the Detectron2Processor class to perform object detection and segmentation.

        Args:
            frames_input_folder (str): Path to the folder containing input frames.
            frames_output_folder (str): Path to the folder where output frames will be saved.
            model_name (str): Name of the model used for logging purposes.
            model_identifier (str): Unique identifier for the model (e.g., version or type).
            model_parameters (str): Information about the model's weights or parameters.
            model_official_site (str): URL to the official model site (e.g., documentation).
        """
        self.frames_input_folder = frames_input_folder
        self.frames_output_folder = frames_output_folder
        self.device = GlobalConfig.device  # Device configuration (CPU or GPU)
        
        # Initialize the model configuration and predictor
        self.cfg = self.initialize_model()
        self.predictor = DefaultPredictor(self.cfg)
        
        # Retrieve metadata (like class labels) for the model's dataset
        self.metadata = MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0])

        # Model information for logging
        self.model_name = model_name
        self.model_identifier = model_identifier
        self.model_parameters = model_parameters
        self.model_official_site = model_official_site

        # Initialize the logger to track the processing and inference metrics
        self.logger = Logger(self.model_name, self.model_identifier, self.model_parameters, self.model_official_site)

    def initialize_model(self):
        """
        Initializes the Detectron2 model configuration and loads the pre-trained weights.

        Returns:
            cfg (detectron2.config.CfgNode): The configuration object for the model.
        """
        cfg = get_cfg()
        # Load pre-configured model architecture for panoptic segmentation
        cfg.merge_from_file(model_zoo.get_config_file("COCO-PanopticSegmentation/panoptic_fpn_R_101_3x.yaml"))
        
        # Set the confidence threshold for model predictions
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  
        
        # Load the pre-trained model weights from the model zoo
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-PanopticSegmentation/panoptic_fpn_R_101_3x.yaml")
        
        # Set the device (CPU or GPU) for model inference
        cfg.MODEL.DEVICE = self.device
        return cfg

    def process_frames(self):
        """
        Processes all frames in the input folder by performing inference using the Detectron2 model
        and saves the annotated frames and logs in the output folder.
        """
        # Gather all PNG files from the input folder
        frame_files = sorted([f for f in os.listdir(self.frames_input_folder) if f.endswith('.png')])
        total_frames = len(frame_files)

        # Exit if no frames are found
        if total_frames == 0:
            print("Error: No frames found in the specified folder.")
            return

        # Calculate the number of frames to process based on the percentage defined in config
        max_frames_to_process = int(total_frames * GlobalConfig.video_percentage)
        total_inference_time = 0.0
        frame_id = 0

        # Define the output directory for processed frames
        model_frames_output_dir = os.path.join(self.frames_output_folder, self.model_name)

        # Clear the output folder if it already contains files
        if os.path.exists(model_frames_output_dir):
            if os.listdir(model_frames_output_dir):
                shutil.rmtree(model_frames_output_dir)
                print(f"Warning: Output folder '{model_frames_output_dir}' had files which were cleaned.")
        os.makedirs(model_frames_output_dir, exist_ok=True)

        # Record the start time of the script
        script_start_time = time.perf_counter()

        # Process each frame
        for frame_file in frame_files[:max_frames_to_process]:
            frame_path = os.path.join(self.frames_input_folder, frame_file)
            frame = cv2.imread(frame_path)

            # If a frame can't be read, skip it
            if frame is None:
                print(f"Error: Could not read frame {frame_file}.")
                continue

            print(f"Processing frame {frame_id + 1}/{max_frames_to_process}...")

            # Record the start time for this frame
            frame_start_time = time.perf_counter()

            # Perform inference on the frame
            with torch.no_grad():
                outputs = self.predictor(frame)

            # Record the end time for inference
            frame_end_time = time.perf_counter()
            inference_time_ms = (frame_end_time - frame_start_time) * 1000  # Convert to milliseconds

            # Extract detections from the model's output
            detections = []
            instances = outputs["instances"].to("cpu")  # Move results to CPU
            boxes = instances.pred_boxes.tensor.numpy()  # Bounding boxes
            scores = instances.scores.numpy()  # Confidence scores
            classes = instances.pred_classes.numpy()  # Class IDs

            # Collect detections (class label, bounding box, and confidence score)
            for box, score, cls in zip(boxes, scores, classes):
                detections.append({
                    'label': self.metadata.thing_classes[cls],  # Get the class label from metadata
                    'box': box.tolist(),
                    'score': round(float(score), 2)
                })

            # Use Detectron2's Visualizer to annotate the frame with detected instances
            v = Visualizer(frame[:, :, ::-1], self.metadata, instance_mode=ColorMode.IMAGE)
            out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
            processed_frame = out.get_image()[:, :, ::-1]  # Convert back to BGR format for OpenCV

            # Compute the total processing time for this frame in milliseconds
            frame_total_time_ms = (frame_end_time - frame_start_time) * 1000

            # Log the frame's processing details using the Logger
            self.logger.log_frame(
                frame_number=frame_id,
                detections=detections,  # Log the extracted detections (label, box, score)
                inference_time_ms=inference_time_ms,
                total_time_ms=frame_total_time_ms
            )

            # Save the annotated frame to the output directory
            frame_filename = os.path.join(model_frames_output_dir, f'frame_{frame_id:04d}.png')
            cv2.imwrite(frame_filename, processed_frame)

            # Increment the frame counter
            frame_id += 1

            # Clear cache and free memory after processing each frame
            del outputs
            torch.cuda.empty_cache()
            gc.collect()

        # Record the total script execution time
        total_script_time_ms = (time.perf_counter() - script_start_time) * 1000
        self.logger.set_total_time(total_script_time_ms)

        # Save the final log with total processing time and all frame data
        self.logger.save_log()

        print("Processing complete. Frames and log saved.")

if __name__ == "__main__":
    # Model details for logging
    model_name = "detectron2"
    model_identifier = "panoptic_fpn_R_101_3x"
    model_parameters = "65.0M"
    model_official_site = "https://github.com/facebookresearch/detectron2"

    # Input and output folder paths (defined in GlobalConfig)
    frames_input_folder = GlobalConfig.frames_input_folder
    frames_output_folder = GlobalConfig.frames_output_folder

    # Instantiate the processor and start processing frames
    processor = Detectron2Processor(frames_input_folder, frames_output_folder, model_name, model_identifier, model_parameters, model_official_site)
    processor.process_frames()
