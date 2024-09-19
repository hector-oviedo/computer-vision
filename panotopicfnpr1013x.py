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
        """Initialize the Detectron2Processor class."""

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
        cfg.MODEL.DEVICE = self.device
        return cfg

    def process_frames(self):
        """Process the input frames, perform inference, and save results."""

        frame_files = sorted([f for f in os.listdir(self.frames_input_folder) if f.endswith('.png')])
        total_frames = len(frame_files)

        if total_frames == 0:
            print("Error: No frames found in the specified folder.")
            return

        max_frames_to_process = int(total_frames * GlobalConfig.video_percentage)
        total_inference_time = 0.0
        frame_id = 0

        model_frames_output_dir = os.path.join(self.frames_output_folder, self.model_name)

        if os.path.exists(model_frames_output_dir):
            if os.listdir(model_frames_output_dir):
                shutil.rmtree(model_frames_output_dir)
                print(f"Warning: Output folder '{model_frames_output_dir}' had files which were cleaned.")
        os.makedirs(model_frames_output_dir, exist_ok=True)

        script_start_time = time.perf_counter()

        for frame_file in frame_files[:max_frames_to_process]:
            frame_path = os.path.join(self.frames_input_folder, frame_file)
            frame = cv2.imread(frame_path)

            if frame is None:
                print(f"Error: Could not read frame {frame_file}.")
                continue

            print(f"Processing frame {frame_id + 1}/{max_frames_to_process}...")

            frame_start_time = time.perf_counter()

            # Inference and timing
            with torch.no_grad():
                outputs = self.predictor(frame)

            frame_end_time = time.perf_counter()
            inference_time_ms = (frame_end_time - frame_start_time) * 1000

            # Extract detections (label, box, score) from outputs
            detections = []
            instances = outputs["instances"].to("cpu")
            boxes = instances.pred_boxes.tensor.numpy()
            scores = instances.scores.numpy()
            classes = instances.pred_classes.numpy()

            for box, score, cls in zip(boxes, scores, classes):
                detections.append({
                    'label': self.metadata.thing_classes[cls],
                    'box': box.tolist(),
                    'score': round(float(score), 2)
                })

            # Use Detectron2's Visualizer to annotate the frame
            v = Visualizer(frame[:, :, ::-1], self.metadata, instance_mode=ColorMode.IMAGE)
            out = v.draw_instance_predictions(outputs["instances"].to("cpu"))

            processed_frame = out.get_image()[:, :, ::-1]

            # Compute total time for frame processing in milliseconds
            frame_total_time_ms = (frame_end_time - frame_start_time) * 1000

            # Log frame data using the Logger
            self.logger.log_frame(
                frame_number=frame_id,
                model_name=self.model_name,
                detections=detections,  # Log the extracted detections
                inference_time_ms=inference_time_ms,
                total_time_ms=frame_total_time_ms
            )

            # Save the annotated frame
            frame_filename = os.path.join(model_frames_output_dir, f'frame_{frame_id:04d}.png')
            cv2.imwrite(frame_filename, processed_frame)

            # Update the frame counter
            frame_id += 1

            # Clear cache and free up memory after each frame
            del outputs
            torch.cuda.empty_cache()
            gc.collect()

        total_script_time_ms = (time.perf_counter() - script_start_time) * 1000
        self.logger.set_total_time(total_script_time_ms)
        self.logger.save_log()

        print("Processing complete. Frames and log saved.")

if __name__ == "__main__":
    output_json = "detectron2"

    frames_input_folder = GlobalConfig.frames_input_folder
    frames_output_folder = GlobalConfig.frames_output_folder

    processor = Detectron2Processor(frames_input_folder, frames_output_folder, output_json)
    processor.process_frames()
