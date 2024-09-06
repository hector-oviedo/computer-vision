import os
import sys
import cv2
import torch
import time
import gc
import shutil
import numpy as np
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
        total_inference_time = 0  # Track cumulative inference time
        frame_id = 0

        # Ensure the model-specific subfolder exists within the frames output folder
        model_frames_output_dir = os.path.join(self.frames_output_folder, self.model_name)

        # Clear the folder if it contains any files
        if os.path.exists(model_frames_output_dir):
            if os.listdir(model_frames_output_dir):  # Check if the folder is not empty
                shutil.rmtree(model_frames_output_dir)
                print(f"Warning: Output folder '{model_frames_output_dir}' had files which were cleaned.")
        os.makedirs(model_frames_output_dir, exist_ok=True)

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
            self.logger.log_frame(frame_id, self.model_name, frame_data['results'], inference_time)

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
        """Processes an individual frame and performs inference using EfficientDet.

        Args:
            frame (ndarray): The input frame to be processed.

        Returns:
            tuple: The annotated frame and data to be logged.
        """
        scale_size = 1536  # EfficientDet-D7 expects 1536x1536 input
        frame_resized = cv2.resize(frame, (scale_size, scale_size))
        frame_tensor = torch.from_numpy(frame_resized / 255.0).permute(2, 0, 1).unsqueeze(0).float().to(self.device)

        # Perform inference
        with torch.no_grad():
            results = self.model(frame_tensor)

        results = results[0].detach().cpu().numpy()
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
            cv2.rectangle(frame, (int(box[0]), int(box[1]) - text_height - baseline), (int(box[0]) + text_width, int(box[1])), GlobalConfig.label_box_background, -1)
            cv2.putText(frame, label_text, (int(box[0]), int(box[1]) - baseline), cv2.FONT_HERSHEY_SIMPLEX, 0.5, GlobalConfig.label_text_color, 1, lineType=cv2.LINE_AA)

        return frame, frame_data


if __name__ == "__main__":
    # Instantiate the processor class
    processor = EfficientDetProcessor(GlobalConfig.frames_input_folder, GlobalConfig.frames_output_folder, "")

    # Use configuration to specify input and output folders
    frames_input_folder = GlobalConfig.frames_input_folder
    frames_output_folder = GlobalConfig.frames_output_folder

    # Use processor.model_name directly for the output JSON
    output_json = f'{processor.model_name}_results'

    processor = EfficientDetProcessor(frames_input_folder, frames_output_folder, output_json)
    processor.process_frames()
