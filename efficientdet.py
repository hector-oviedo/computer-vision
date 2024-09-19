import os
import sys
import cv2
import torch
import time
import gc
import shutil
import numpy as np
import psutil  # For CPU and RAM usage metrics
from effdet import create_model

# Add the utils directory to the Python path
script_dir = os.path.dirname(os.path.abspath(__file__))
utils_dir = os.path.join(script_dir, 'utils')
sys.path.append(utils_dir)

from config import GlobalConfig
from logger import Logger

class EfficientDetProcessor:
    def __init__(self, frames_input_folder, frames_output_folder, output_json):
        """Initialize the EfficientDetProcessor class.

        Args:
            frames_input_folder (str): Directory where the input frames are stored.
            frames_output_folder (str): Directory where the output frames will be saved.
            output_json (str): File path for saving the log data in JSON format.
        """
        # Name of the model where we will save the frames and log the output
        self.model_name = "efficientdet"

        self.frames_input_folder = frames_input_folder
        self.frames_output_folder = frames_output_folder
        self.logger = Logger(output_json)
        self.device = GlobalConfig.device
        self.model = self.initialize_model()

        # Mapping of COCO class names (loaded from config)
        self.class_names = GlobalConfig.COCO_LABELS

    def initialize_model(self):
        """Initializes the EfficientDet model."""
        model_name = 'tf_efficientdet_d7'  # Using EfficientDet-D7 as a large model
        model = create_model(model_name, bench_task='predict', num_classes=90, pretrained=True)
        model = model.to(self.device)
        model.eval()  # Set model to evaluation mode
        return model

    def process_frames(self):
        """Process the input frames, perform inference, and save results."""
        # Get all frame files from the input folder and sort them
        frame_files = sorted([f for f in os.listdir(self.frames_input_folder) if f.endswith('.png')])
        total_frames = len(frame_files)

        # If no frames are found, exit
        if total_frames == 0:
            print(f"Error: No frames found in the folder: {self.frames_input_folder}")
            return

        # Determine the number of frames to process based on the video_percentage setting
        max_frames_to_process = int(total_frames * GlobalConfig.video_percentage)

        # Initialize frame counter and start the total time tracking
        total_inference_time = 0.0  # Track cumulative inference time in milliseconds
        frame_id = 0

        # Ensure the model-specific subfolder exists within the frames output folder
        model_frames_output_dir = os.path.join(self.frames_output_folder, self.model_name)

        # Clear the folder if it contains any files
        if os.path.exists(model_frames_output_dir):
            if os.listdir(model_frames_output_dir):  # Check if the folder is not empty
                shutil.rmtree(model_frames_output_dir)
                print(f"Warning: Output folder '{model_frames_output_dir}' had files which were cleaned.")
        os.makedirs(model_frames_output_dir, exist_ok=True)

        # Record script start time
        script_start_time = time.perf_counter()

        # Process each frame within the specified percentage
        for frame_file in frame_files[:max_frames_to_process]:
            frame_path = os.path.join(self.frames_input_folder, frame_file)
            frame = cv2.imread(frame_path)

            # Ensure the frame is readable, if not, skip it
            if frame is None:
                print(f"Error: Could not read frame {frame_file}. Skipping.")
                continue

            # Log progress as `frame_id / total_frames_to_process`
            print(f"Processing frame {frame_id + 1}/{max_frames_to_process}...")

            # Record frame start time
            frame_start_time = time.perf_counter()

            # Measure inference time for each frame
            start_time = time.time()
            processed_frame, frame_data = self.process_frame(frame)
            inference_time = time.time() - start_time

            # Convert inference_time from seconds to milliseconds
            inference_time_ms = inference_time * 1000

            # Accumulate inference time for overall calculation
            total_inference_time += inference_time_ms

            # Save the annotated frame in the model-specific subfolder
            frame_filename = os.path.join(model_frames_output_dir, f'frame_{frame_id:04d}.png')
            cv2.imwrite(frame_filename, processed_frame)

            # Record frame end time
            frame_end_time = time.perf_counter()

            # Compute total time for frame processing in milliseconds
            frame_total_time_ms = (frame_end_time - frame_start_time) * 1000  # ms

            # Gather CPU and GPU metrics
            cpu_usage = psutil.cpu_percent(interval=None)  # Current CPU usage percentage
            cpu_ram_usage = psutil.virtual_memory().percent  # Current RAM usage percentage
            gpu_vram_usage = round(torch.cuda.memory_allocated(self.device) / 1e6, 2)  # Convert bytes to MB
            gpu_vram_reserved = round(torch.cuda.memory_reserved(self.device) / 1e6, 2)  # Convert bytes to MB

            # Log frame data using the Logger
            try:
                self.logger.log_frame(
                    frame_number=frame_id,
                    model_name=self.model_name,  # Changed from 'model' to 'model_name'
                    detections=frame_data['results'],
                    inference_time_ms=inference_time_ms,
                    total_time_ms=frame_total_time_ms
                )
            except TypeError as e:
                print(f"Logger Error: {e}")
                print("Please ensure that the Logger class has a 'log_frame' method with the appropriate signature.")
                print("Skipping logging for this frame.")
                pass

            # Update the frame counter
            frame_id += 1

            # Clear cache and free up memory after each frame
            del processed_frame, frame_data
            torch.cuda.empty_cache()
            gc.collect()

        # After processing all frames, set the total processing time
        total_script_time_ms = (time.perf_counter() - script_start_time) * 1000  # ms
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

    def process_frames(self):
        """Process the input frames, perform inference, and save results."""
        # Get all frame files from the input folder and sort them
        frame_files = sorted([f for f in os.listdir(self.frames_input_folder) if f.endswith('.png')])
        total_frames = len(frame_files)

        # If no frames are found, exit
        if total_frames == 0:
            print(f"Error: No frames found in the folder: {self.frames_input_folder}")
            return

        # Determine the number of frames to process based on the video_percentage setting
        max_frames_to_process = int(total_frames * GlobalConfig.video_percentage)

        # Initialize frame counter and start the total time tracking
        total_inference_time = 0.0  # Track cumulative inference time in milliseconds
        frame_id = 0

        # Ensure the model-specific subfolder exists within the frames output folder
        model_frames_output_dir = os.path.join(self.frames_output_folder, self.model_name)

        # Clear the folder if it contains any files
        if os.path.exists(model_frames_output_dir):
            if os.listdir(model_frames_output_dir):  # Check if the folder is not empty
                shutil.rmtree(model_frames_output_dir)
                print(f"Warning: Output folder '{model_frames_output_dir}' had files which were cleaned.")
        os.makedirs(model_frames_output_dir, exist_ok=True)

        # Record script start time
        script_start_time = time.perf_counter()

        # Process each frame within the specified percentage
        for frame_file in frame_files[:max_frames_to_process]:
            frame_path = os.path.join(self.frames_input_folder, frame_file)
            frame = cv2.imread(frame_path)

            # Ensure the frame is readable, if not, skip it
            if frame is None:
                print(f"Error: Could not read frame {frame_file}. Skipping.")
                continue

            # Log progress as frame_id / total_frames_to_process
            print(f"Processing frame {frame_id + 1}/{max_frames_to_process}...")

            # Record frame start time
            frame_start_time = time.perf_counter()

            # Preprocess the frame and prepare tensor
            scale_size = 1536  # EfficientDet-D7 expects 1536x1536 input
            frame_resized = cv2.resize(frame, (scale_size, scale_size))
            frame_tensor = torch.from_numpy(frame_resized / 255.0).permute(2, 0, 1).unsqueeze(0).float().to(self.device)

            # Initialize CUDA events for accurate timing
            if self.device.startswith('cuda'):
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                start_event.record()

            # Perform inference
            with torch.no_grad():
                results = self.model(frame_tensor)

            # Record end event and synchronize
            if self.device.startswith('cuda'):
                end_event.record()
                torch.cuda.synchronize()
                inference_time_ms = start_event.elapsed_time(end_event)  # Time in milliseconds
            else:
                # If not using CUDA, fallback to time.time()
                inference_time_ms = 0.0

            # Convert results to NumPy array
            if isinstance(results, torch.Tensor):
                results = results.detach().cpu().numpy()
            elif isinstance(results, list) and isinstance(results[0], torch.Tensor):
                results = results[0].detach().cpu().numpy()
            else:
                raise ValueError("Unexpected model output type.")

            # Extract boxes, scores, and labels
            if results.ndim == 3:
                # Shape [1, N, 6]
                results = results[0]
            elif results.ndim == 2:
                # Shape [N, 6]
                pass
            else:
                raise ValueError(f"Unexpected results shape: {results.shape}")

            boxes = results[:, :4]
            scores = results[:, 4]
            labels = results[:, 5].astype(int)

            frame_data = {'results': []}
            filtered_indices = np.where(scores >= GlobalConfig.confidence_threshold)[0]

            for i in filtered_indices:
                box = boxes[i].tolist()
                score = scores[i]
                label = labels[i]
                label_name = self.class_names.get(label, f'class_{label}')

                frame_data['results'].append({
                    'label': label_name,
                    'box': [int(coord) for coord in box],
                    'score': round(float(score), 2)
                })

                # Rescale bounding box to original frame size
                box[0] = int(box[0] * frame.shape[1] / scale_size)
                box[1] = int(box[1] * frame.shape[0] / scale_size)
                box[2] = int(box[2] * frame.shape[1] / scale_size)
                box[3] = int(box[3] * frame.shape[0] / scale_size)

                # Draw bounding box and label on the frame
                box_color = GlobalConfig.object_colors[label % len(GlobalConfig.object_colors)]
                cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), box_color, GlobalConfig.box_border_width)
                label_text = f"{label_name} {score:.2f}"
                (text_width, text_height), baseline = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(frame, (int(box[0]), int(box[1]) - text_height - baseline), 
                              (int(box[0]) + text_width, int(box[1])), 
                              GlobalConfig.label_box_background, -1)
                cv2.putText(frame, label_text, (int(box[0]), int(box[1]) - baseline), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, GlobalConfig.label_text_color, 1, lineType=cv2.LINE_AA)

            # Save the annotated frame
            frame_filename = os.path.join(model_frames_output_dir, f'frame_{frame_id:04d}.png')
            cv2.imwrite(frame_filename, frame)

            # Compute total time for frame processing in milliseconds
            frame_end_time = time.perf_counter()
            frame_total_time_ms = (frame_end_time - frame_start_time) * 1000  # ms

            # Log frame data using the Logger
            try:
                self.logger.log_frame(
                    frame_number=frame_id,
                    model_name=self.model_name,  # Changed from 'model' to 'model_name'
                    detections=frame_data['results'],
                    inference_time_ms=inference_time_ms,
                    total_time_ms=frame_total_time_ms
                )
            except TypeError as e:
                print(f"Logger Error: {e}")
                print("Please ensure that the Logger class has a 'log_frame' method with the appropriate signature.")
                print("Skipping logging for this frame.")
                pass

            # Accumulate inference time for overall calculation
            total_inference_time += inference_time_ms

            # Update the frame counter
            frame_id += 1

            # Clear cache and free up memory after each frame
            del frame_tensor, results, frame_data
            torch.cuda.empty_cache()
            gc.collect()

        # After processing all frames, set the total processing time and save the log
        total_script_time_ms = (time.perf_counter() - script_start_time) * 1000  # ms
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
    # Define the output JSON log file
    output_json = "efficientdet"

    # Use configuration to specify input and output folders
    frames_input_folder = GlobalConfig.frames_input_folder
    frames_output_folder = GlobalConfig.frames_output_folder

    # Instantiate the processor class with the new Logger
    processor = EfficientDetProcessor(frames_input_folder, frames_output_folder, output_json)
    processor.process_frames()
