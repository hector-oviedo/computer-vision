import os
import sys
import cv2
import torch
import numpy as np
from effdet import create_model

import warnings
import time
import shutil
import gc
import psutil  # For gathering CPU and RAM usage metrics

# Add the utils directory to the Python path for easy import of custom modules
script_dir = os.path.dirname(os.path.abspath(__file__))
utils_dir = os.path.join(script_dir, 'utils')
sys.path.append(utils_dir)

from config import GlobalConfig
from logger import Logger

# Optionally suppress specific warnings from torch (such as FutureWarning)
warnings.filterwarnings("ignore", category=FutureWarning, module='torch')

class EfficientDetProcessor:
    def __init__(self, frames_input_folder, frames_output_folder, model_name, model_identifier, model_parameters, model_official_site):
        """
        Initializes the EfficientDetProcessor class.

        Args:
            frames_input_folder (str): Directory where the input frames (images) are stored.
            frames_output_folder (str): Directory where the processed frames will be saved.
            model_name (str): Name of the model to be used for inference.
            model_identifier (str): A unique identifier for logging purposes.
            model_parameters (str): Information about the model parameters (e.g., number of parameters).
            model_official_site (str): URL to the official site for reference (e.g., GitHub repo).
        """
        # Store model details and directory paths
        self.model_name = model_name
        self.model_identifier = model_identifier
        self.model_parameters = model_parameters
        self.model_official_site = model_official_site

        # Paths to input and output folders
        self.frames_input_folder = frames_input_folder
        self.frames_output_folder = frames_output_folder

        # Logger instance to log the processing details
        self.logger = Logger(self.model_name, self.model_identifier, self.model_parameters, self.model_official_site)

        # Set the device (CPU/GPU) for inference, based on GlobalConfig
        self.device = GlobalConfig.device

        # Initialize the model (EfficientDet)
        self.model = self.initialize_model()

        # Mapping COCO class labels to human-readable names (defined in GlobalConfig)
        self.class_names = GlobalConfig.COCO_LABELS

    def initialize_model(self):
        """
        Initializes the EfficientDet model with the specified parameters.

        The model is set to evaluation mode, which disables layers like dropout.
        
        Returns:
            torch.nn.Module: The initialized EfficientDet model.
        """
        # Create the model with pretrained weights and set it to 'predict' mode
        model = create_model(self.model_identifier, bench_task='predict', num_classes=90, pretrained=True)
        model = model.to(self.device)  # Move model to the selected device (GPU/CPU)
        model.eval()  # Set the model to evaluation mode for inference
        return model

    def process_frame(self, frame):
        """
        Processes a single frame by performing inference and annotating it with detection results.

        Args:
            frame (numpy.ndarray): Input frame (image) in BGR format as read by OpenCV.

        Returns:
            tuple: 
                processed_frame (numpy.ndarray): Frame with bounding boxes and labels drawn.
                frame_data (dict): Inference results, including bounding boxes, labels, and confidence scores.
        """
        # EfficientDet-D7x expects a 1536x1536 input, resize the frame accordingly
        scale_size = 1536
        frame_resized = cv2.resize(frame, (scale_size, scale_size))

        # Normalize the frame (convert pixel values to range [0, 1]) and convert to tensor
        frame_tensor = torch.from_numpy(frame_resized / 255.0).permute(2, 0, 1).unsqueeze(0).float().to(self.device)

        # Measure inference time
        start_time = time.perf_counter()

        # Perform model inference
        with torch.no_grad():  # Disable gradient computation for efficiency
            results = self.model(frame_tensor)

        # End timing
        end_time = time.perf_counter()
        inference_time_ms = (end_time - start_time) * 1000  # Convert time to milliseconds

        # Convert results from torch.Tensor to numpy.ndarray for easier manipulation
        if isinstance(results, torch.Tensor):
            results = results.detach().cpu().numpy()
        elif isinstance(results, list) and isinstance(results[0], torch.Tensor):
            results = results[0].detach().cpu().numpy()
        else:
            raise ValueError("Unexpected model output type.")

        # Handle the model's output format (Nx6 where each row is [x1, y1, x2, y2, score, class_id])
        if results.ndim == 3:
            results = results[0]  # In case the output has an extra dimension (e.g., [1, N, 6])
        elif results.ndim != 2:
            raise ValueError(f"Unexpected results shape: {results.shape}")

        # Extract bounding boxes, scores, and labels from the output
        boxes = results[:, :4]
        scores = results[:, 4]
        labels = results[:, 5].astype(int)  # Class IDs should be integers

        # Initialize a data structure to store frame inference results
        frame_data = {'results': [], 'inference_time_ms': inference_time_ms}

        # Filter results based on confidence threshold
        filtered_indices = np.where(scores >= GlobalConfig.confidence_threshold)[0]

        # Process each detected object
        for i in filtered_indices:
            box = boxes[i].tolist()
            score = scores[i]
            label = labels[i]

            # Get the human-readable label name from the class ID
            label_name = self.class_names.get(label, f'class_{label}')

            # Rescale bounding boxes back to the original frame size
            scaled_box = [
                int(box[0] * frame.shape[1] / scale_size),
                int(box[1] * frame.shape[0] / scale_size),
                int(box[2] * frame.shape[1] / scale_size),
                int(box[3] * frame.shape[0] / scale_size)
            ]

            # Add detection details to the frame data
            frame_data['results'].append({
                'label': label_name,
                'box': scaled_box,
                'score': round(float(score), 2)
            })

            # Draw the bounding box and label on the frame
            box_color = GlobalConfig.object_colors[label % len(GlobalConfig.object_colors)]
            box_border_width = GlobalConfig.box_border_width  # Thickness of the box lines

            # Draw bounding box
            cv2.rectangle(frame, (scaled_box[0], scaled_box[1]), (scaled_box[2], scaled_box[3]), box_color, box_border_width)

            # Prepare the label text (e.g., "person 0.98")
            label_text = f"{label_name} {score:.2f}"
            font_scale = 0.5
            font_thickness = 1  # Thickness of the text
            (text_width, text_height), baseline = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
            y1_label = max(scaled_box[1] - text_height - baseline, 0)

            # Draw black background rectangle behind the label text
            cv2.rectangle(frame, (scaled_box[0], y1_label), (scaled_box[0] + text_width, scaled_box[1]), (0, 0, 0), -1)

            # Draw white label text on top of the black rectangle
            cv2.putText(frame, label_text, (scaled_box[0], scaled_box[1] - baseline),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), font_thickness, cv2.LINE_AA)

        return frame, frame_data

    def process_frames(self):
        """
        Processes all the frames in the input folder, performs inference on each frame, 
        and saves the results (annotated frames and logs).
        """
        # Retrieve all frame filenames from the input folder and sort them
        frame_files = sorted([f for f in os.listdir(self.frames_input_folder) if f.endswith('.png')])
        total_frames = len(frame_files)

        # Exit if no frames are found
        if total_frames == 0:
            print(f"Error: No frames found in the folder: {self.frames_input_folder}")
            return

        # Determine the number of frames to process based on a user-defined percentage (in config)
        max_frames_to_process = int(total_frames * GlobalConfig.video_percentage)

        # Initialize time tracking variables
        total_inference_time = 0.0  # Cumulative inference time for all frames
        frame_id = 0

        # Create or clear the output directory for the processed frames
        model_frames_output_dir = os.path.join(self.frames_output_folder, self.model_identifier)
        if os.path.exists(model_frames_output_dir):
            if os.listdir(model_frames_output_dir):  # Only clear if the directory is not empty
                shutil.rmtree(model_frames_output_dir)
                print(f"Warning: Output folder '{model_frames_output_dir}' had files which were cleaned.")
        os.makedirs(model_frames_output_dir, exist_ok=True)

        # Start measuring the total script runtime
        script_start_time = time.perf_counter()

        # Process each frame up to the defined limit
        for frame_file in frame_files[:max_frames_to_process]:
            frame_path = os.path.join(self.frames_input_folder, frame_file)
            frame = cv2.imread(frame_path)

            # Skip frames that cannot be read
            if frame is None:
                print(f"Error: Could not read frame {frame_file}. Skipping.")
                continue

            # Display progress to the user
            print(f"Processing frame {frame_id + 1}/{max_frames_to_process}...")

            # Start timing for this specific frame
            frame_start_time = time.perf_counter()

            # Perform inference on the frame and retrieve the processed frame and results
            processed_frame, frame_data = self.process_frame(frame)

            # Accumulate the inference time for later analysis
            total_inference_time += frame_data['inference_time_ms']

            # Save the processed (annotated) frame to the output directory
            frame_filename = os.path.join(model_frames_output_dir, f'frame_{frame_id:04d}.png')
            cv2.imwrite(frame_filename, processed_frame)

            # Stop timing for this frame
            frame_end_time = time.perf_counter()
            frame_total_time_ms = (frame_end_time - frame_start_time) * 1000  # Convert to milliseconds

            # Log CPU, RAM, and GPU usage metrics
            cpu_usage = psutil.cpu_percent(interval=None)  # CPU usage as a percentage
            cpu_ram_usage = psutil.virtual_memory().percent  # RAM usage as a percentage
            gpu_vram_usage = round(torch.cuda.memory_allocated(self.device) / 1e6, 2)  # GPU memory used in MB
            gpu_vram_reserved = round(torch.cuda.memory_reserved(self.device) / 1e6, 2)  # GPU memory reserved in MB

            # Log the frame processing details using the Logger class
            try:
                self.logger.log_frame(
                    frame_number=frame_id,
                    detections=frame_data['results'],
                    inference_time_ms=frame_data['inference_time_ms'],
                    total_time_ms=frame_total_time_ms
                )
            except TypeError as e:
                print(f"Logger Error: {e}")
                print("Please ensure that the Logger class has a 'log_frame' method with the appropriate signature.")
                print("Skipping logging for this frame.")
                pass

            # Increment the frame counter
            frame_id += 1

            # Free up memory by clearing caches and running garbage collection
            del processed_frame, frame_data
            torch.cuda.empty_cache()
            gc.collect()

        # After processing all frames, calculate and log the total script runtime
        total_script_time_ms = (time.perf_counter() - script_start_time) * 1000
        self.logger.set_total_time(total_script_time_ms)

        # Save the final log data
        try:
            self.logger.save_log()
        except AttributeError as e:
            print(f"Logger Error: {e}")
            print("Please ensure that the Logger class has a 'save_log' method.")
            print("Skipping saving the log.")
            pass

        print("Processing complete. Frames and log saved.")


if __name__ == "__main__":
    # Model name specified by the user
    model_name = "efficientdet"
    
    # Model identifier for logging purposes
    model_identifier = "tf_efficientdet_d7x"
    
    # Model parameters (e.g., number of parameters, model size)
    model_parameters = "77.0M"
    
    # Official site for model reference
    model_official_site = "https://github.com/google/automl/tree/master/efficientdet"

    # Input and output directories from configuration
    frames_input_folder = GlobalConfig.frames_input_folder
    frames_output_folder = GlobalConfig.frames_output_folder

    # Instantiate the processor and start processing frames
    processor = EfficientDetProcessor(frames_input_folder, frames_output_folder, model_name, model_identifier, model_parameters, model_official_site)
    processor.process_frames()