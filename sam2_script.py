import os
import sys
import cv2
import torch
import numpy as np
from PIL import Image
from sam2.build_sam import build_sam2_video_predictor
from sam2_detect_objects import detect_objects

import warnings
import time
import shutil
import gc
import tempfile

# Add the utils directory to the Python path for custom modules
script_dir = os.path.dirname(os.path.abspath(__file__))
utils_dir = os.path.join(script_dir, 'utils')
sys.path.append(utils_dir)

from config import GlobalConfig
from logger import Logger

# Suppress specific warnings from torch, if needed
warnings.filterwarnings("ignore", category=FutureWarning, module='torch')


def convert_png_to_jpg(frames_input_folder, start_frame, end_frame, temp_dir):
    """
    Convert .png frames within the specified range to .jpg and save them in a temporary directory.

    Args:
        frames_input_folder (str): Path to the input frames directory containing .png files.
        start_frame (int): Starting frame number (inclusive).
        end_frame (int): Ending frame number (inclusive).
        temp_dir (str): Path to the temporary directory where converted .jpg files will be saved.
    """
    for frame_num in range(start_frame, end_frame + 1):
        frame_name_png = f"frame_{frame_num:04d}.png"
        frame_name_jpg = f"{frame_num}.jpg"  # Converted frame names are numerical
        frame_path_png = os.path.join(frames_input_folder, frame_name_png)
        frame_path_jpg = os.path.join(temp_dir, frame_name_jpg)

        if not os.path.exists(frame_path_png):
            print(f"Warning: {frame_name_png} does not exist. Skipping.")
            continue

        try:
            # Open the .png file and convert it to .jpg format
            with Image.open(frame_path_png) as img:
                img.convert('RGB').save(frame_path_jpg, 'JPEG', quality=95)
        except Exception as e:
            print(f"Error converting {frame_name_png} to JPG: {e}")


class SAM2Processor:
    def __init__(self, frames_input_folder, frames_output_folder, model_name, model_identifier, model_parameters, model_official_site, start_frame=None, end_frame=None):
        """
        Initialize the SAM2Processor class for video frame segmentation using SAM2.

        Args:
            frames_input_folder (str): Directory containing input frames.
            frames_output_folder (str): Directory where the output frames will be saved.
            model_name (str): Name of the model (for logging purposes).
            model_identifier (str): Model identifier for logging.
            model_parameters (str): Model parameter details (e.g., number of weights).
            model_official_site (str): URL of the official model site.
            start_frame (int, optional): Starting frame number (inclusive). Defaults to None.
            end_frame (int, optional): Ending frame number (inclusive). Defaults to None.
        """
        self.frames_input_folder = frames_input_folder
        self.frames_output_folder = frames_output_folder
        self.device = GlobalConfig.device  # Device configuration (CPU/GPU)

        # Model information for logging
        self.model_name = model_name
        self.model_identifier = model_identifier
        self.model_parameters = model_parameters
        self.model_official_site = model_official_site

        # Initialize the Logger for tracking the processing and inference
        self.logger = Logger(self.model_name, self.model_identifier, self.model_parameters, self.model_official_site)

        # Initialize the first-frame flag
        self.is_first_frame = True

        # Load SAM2 model with the corresponding checkpoint and configuration
        sam2_checkpoint = "sam2_hiera_large.pt"  # Update with correct model path
        model_cfg = "sam2_hiera_l.yaml"          # Update with correct config file path
        self.predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint, device=self.device)

        # Retrieve and sort frame filenames
        all_frames = [
            p for p in os.listdir(self.frames_input_folder)
            if os.path.splitext(p)[-1].lower() in [".jpg", ".jpeg", ".png"]
        ]

        # Extract and sort based on the frame number (assuming numerical frame names)
        try:
            self.frame_names = sorted(
                all_frames,
                key=lambda p: int(os.path.splitext(p)[0])  # Extract frame number from filename
            )
        except ValueError as ve:
            print(f"Error parsing frame numbers: {ve}")
            self.frame_names = []

        # Filter frames if a specific range is defined
        if start_frame is not None and end_frame is not None:
            self.frame_names = [
                f for f in self.frame_names
                if start_frame <= int(os.path.splitext(f)[0]) <= end_frame
            ]
            if not self.frame_names:
                raise ValueError(f"No frames found in the range {start_frame} to {end_frame}.")

        # Initialize inference state using the SAM2 model
        self.inference_state = self.predictor.init_state(video_path=self.frames_input_folder)


    def process_frames(self):
        """
        Process the input frames by performing segmentation and saving annotated results.
        """
        total_frames = len(self.frame_names)
        if total_frames == 0:
            print("Error: No frames found to process.")
            return

        # Define the number of frames to process based on the percentage in the config
        max_frames_to_process = int(total_frames * GlobalConfig.video_percentage)

        # Define output directory for saving processed frames
        model_frames_output_dir = os.path.join(self.frames_output_folder, self.model_name)

        # Clean the output directory if it already exists
        if os.path.exists(model_frames_output_dir):
            if os.listdir(model_frames_output_dir):  # Check if the folder contains files
                shutil.rmtree(model_frames_output_dir)
                print(f"Warning: Output folder '{model_frames_output_dir}' had files which were cleaned.")
        os.makedirs(model_frames_output_dir, exist_ok=True)

        # Record the start time of the script
        script_start_time = time.perf_counter()

        print(f"Detected {len(self.frame_names)} frames to process.")

        # Loop through the frames and process them
        for idx, frame_name in enumerate(self.frame_names[:max_frames_to_process]):
            frame_path = os.path.join(self.frames_input_folder, frame_name)
            try:
                # Load the frame as a NumPy array
                frame = np.array(Image.open(frame_path))
            except Exception as e:
                print(f"Error: Could not read frame {frame_name}. Exception: {e}")
                continue

            print(f"Processing frame {idx + 1}/{max_frames_to_process}: {frame_name}")

            frame_start_time = time.perf_counter()

            # For the first frame, detect objects and initialize the predictor with the detected points
            if self.is_first_frame:
                objects = detect_objects(frame)  # Detect objects in the frame

                if objects:
                    # Extract object points and labels (assumed to be '1' for all)
                    points = np.array([obj['coords'] for obj in objects], dtype=np.float32)
                    labels = np.array([1] * len(objects), dtype=np.int32)

                    # Add new points to the predictor
                    _, out_obj_ids, out_mask_logits = self.predictor.add_new_points(
                        inference_state=self.inference_state,
                        frame_idx=idx,
                        obj_id=1,  # Starting object ID
                        points=points,
                        labels=labels,
                    )

                # Set the flag to False after processing the first frame
                self.is_first_frame = False

            # Propagate the segmentation mask across frames
            video_segments = {}
            for out_frame_idx, out_obj_ids, out_mask_logits in self.predictor.propagate_in_video(self.inference_state):
                video_segments[out_frame_idx] = {
                    out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
                    for i, out_obj_id in enumerate(out_obj_ids)
                }

            # Get the mask for the current frame (if any)
            mask = video_segments.get(idx, {}).get(1)  # Assuming object ID = 1
            if mask is not None:
                mask = mask.squeeze()  # Ensure the mask has two dimensions

                # Color and blend the mask onto the frame
                if len(mask.shape) == 2:
                    mask_colored = np.zeros_like(frame, dtype=np.uint8)
                    mask_colored[mask > 0] = (0, 255, 0)  # Green mask

                    # Apply transparency to the mask
                    alpha_mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
                    alpha_mask[mask > 0] = int(255 * GlobalConfig.instance_segmentation_transparency)

                    # Convert frame and mask to RGBA format for blending
                    frame_rgba = cv2.cvtColor(frame, cv2.COLOR_RGB2RGBA)
                    color_mask = cv2.cvtColor(mask_colored, cv2.COLOR_BGR2RGBA)
                    color_mask[:, :, 3] = alpha_mask

                    # Blend the mask with the original frame
                    overlay = cv2.addWeighted(frame_rgba, 1, color_mask, GlobalConfig.instance_segmentation_transparency, 0)
                    frame = cv2.cvtColor(overlay, cv2.COLOR_RGBA2RGB)

                    # Extract the bounding box around the mask
                    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    if contours:
                        largest_contour = max(contours, key=cv2.contourArea)
                        x, y, w, h = cv2.boundingRect(largest_contour)
                        box = [x, y, x + w, y + h]
                        detections = [{
                            'label': 'person',  # Assuming 'person' as the detected object
                            'box': box,
                            'score': 1.0  # Assuming perfect confidence
                        }]
                    else:
                        detections = []
                else:
                    print(f"Warning: Mask for frame {frame_name} is not 2D, skipping mask application.")
                    detections = []
            else:
                detections = []

            # Extract the actual frame number from the filename
            frame_number = int(os.path.splitext(frame_name)[0])

            # Annotate the frame with bounding boxes and labels
            for det in detections:
                label, box, score = det['label'], det['box'], det['score']
                x1, y1, x2, y2 = box

                # Draw bounding box and label on the frame
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 1)  # Thin red bounding box
                label_text = f"{label} {score:.2f}"
                (text_width, text_height), baseline = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
                y1_label = max(y1 - text_height - baseline, 0)

                # Draw a black background rectangle for the label
                cv2.rectangle(frame, (x1, y1_label), (x1 + text_width, y1), (0, 0, 0), -1)

                # Place the white text label on top of the rectangle
                cv2.putText(frame, label_text, (x1, y1 - baseline), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

            # Save the annotated frame to the output directory
            annotated_frame_path = os.path.join(model_frames_output_dir, frame_name)
            try:
                cv2.imwrite(annotated_frame_path, frame)
            except Exception as e:
                print(f"Error: Could not save annotated frame {frame_name}. Exception: {e}")
                continue

            # Record the frame processing time
            frame_end_time = time.perf_counter()
            inference_time_ms = (frame_end_time - frame_start_time) * 1000
            frame_total_time_ms = inference_time_ms

            # Log the frame data (frame number, detections, timing)
            self.logger.log_frame(
                frame_number=frame_number,
                detections=detections,
                inference_time_ms=inference_time_ms,
                total_time_ms=frame_total_time_ms
            )

            # Clear memory and caches after processing the frame
            del mask, mask_colored, frame
            torch.cuda.empty_cache()
            gc.collect()

        # Calculate the total script execution time
        total_script_time_ms = (time.perf_counter() - script_start_time) * 1000
        self.logger.set_total_time(total_script_time_ms)
        self.logger.save_log()

        print("Processing complete. Annotated frames and log saved.")


if __name__ == "__main__":
    # Model information for logging purposes
    model_name = "sam2"
    model_identifier = "SAM2"
    model_parameters = "224.4M"
    model_official_site = "https://github.com/facebookresearch/segment-anything-2"

    # Define input and output folders (from GlobalConfig)
    frames_input_folder = GlobalConfig.frames_input_folder
    frames_output_folder = GlobalConfig.frames_output_folder

    # Define the range of frames to process
    start_frame = 2626  # Starting frame number
    end_frame = 2656    # Ending frame number

    # Create a temporary directory to convert .png to .jpg
    with tempfile.TemporaryDirectory() as temp_jpg_dir:
        print(f"Created temporary directory for .jpg frames at: {temp_jpg_dir}")

        # Convert .png frames to .jpg format for processing
        convert_png_to_jpg(frames_input_folder, start_frame, end_frame, temp_jpg_dir)

        # Verify conversion success
        converted_frames = [
            f for f in os.listdir(temp_jpg_dir)
            if os.path.splitext(f)[-1].lower() == '.jpg'
        ]
        if not converted_frames:
            print("Error: No .jpg frames were converted. Exiting.")
            sys.exit(1)
        else:
            print(f"Successfully converted {len(converted_frames)} frames to .jpg format.")

        # Initialize SAM2Processor with the temporary .jpg frames directory
        processor = SAM2Processor(
            frames_input_folder=temp_jpg_dir,  # Use the temporary directory for .jpg frames
            frames_output_folder=frames_output_folder,
            model_name=model_name,
            model_identifier=model_identifier,
            model_parameters=model_parameters,
            model_official_site=model_official_site,
            start_frame=start_frame,
            end_frame=end_frame
        )
        processor.process_frames()

        # Temporary directory and its contents will be deleted here automatically
        print(f"Temporary directory {temp_jpg_dir} has been deleted.")