import os
import sys

# Add the utils directory to the Python path
script_dir = os.path.dirname(os.path.abspath(__file__))
utils_dir = os.path.join(script_dir, '/utils')
sys.path.append(utils_dir)

import cv2
import torch
import time
import numpy as np

from config import GlobalConfig
from logger import Logger
from video_generator import VideoGenerator

from ultralytics import YOLO


class YOLOv8Processor:
    def __init__(self, video_path, output_video, output_json):
        self.video_path = video_path
        self.logger = Logger(output_json)
        self.video_generator = VideoGenerator(output_video)
        self.device = GlobalConfig.device
        self.model = YOLO('models/yolov8x-seg.pt').to(self.device)
        self.class_color_mapping = {}

    def process_video(self):
        print(f"Processing video from {self.video_path}...")
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            print("Error: Could not open video.")
            return

        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"Video properties - FPS: {fps}, Width: {width}, Height: {height}, Total Frames: {total_frames}")

        self.video_generator.start((width, height), fps)

        frame_id = 0
        total_start_time = time.time()
        max_frames = int(total_frames * GlobalConfig.video_percentage)

        while cap.isOpened() and frame_id < max_frames:
            ret, frame = cap.read()
            if not ret:
                break

            print(f"Processing frame {frame_id}...")

            start_time = time.time()
            resized_frame = cv2.resize(frame, (GlobalConfig.resize_size, GlobalConfig.resize_size))
            results_frame = self.model(resized_frame, device=self.device)
            inference_time = time.time() - start_time

            annotated_frame, frame_data = self.annotate_frame(frame, results_frame)
            self.video_generator.write_frame(annotated_frame)

            self.logger.log_frame(frame_id, 'YOLOv8Seg', frame_data['results'], inference_time)
            frame_id += 1

            # Clear cache to free up memory
            del resized_frame, results_frame
            torch.cuda.empty_cache()

            # Collect garbage to free up memory
            import gc
            gc.collect()

        total_end_time = time.time()
        self.logger.log_data["total_time"] = round(total_end_time - total_start_time, 3)
        self.logger.log_data["total_frames"] = frame_id
        if self.device == 'cuda':
            self.logger.log_data["total_gpu_memory_used"] = round(torch.cuda.max_memory_allocated() / 1e6, 2)  # MB
            self.logger.log_data["total_gpu_memory_cached"] = round(torch.cuda.max_memory_reserved() / 1e6, 2)  # MB

        cap.release()
        self.video_generator.finish()
        self.logger.save_log()

        print("Processing complete. Video and log saved.")

    def annotate_frame(self, frame, results_frame):
        frame_data = {'results': []}
        overlay = frame.copy()  # Create an overlay for drawing the masks

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

                    # Assign a color to each class
                    if int(class_id) not in self.class_color_mapping:
                        self.class_color_mapping[int(class_id)] = {
                            'box_color': GlobalConfig.object_colors[len(self.class_color_mapping) % len(GlobalConfig.object_colors)],
                            'segmentation_color': GlobalConfig.segmentation_colors[len(self.class_color_mapping) % len(GlobalConfig.segmentation_colors)]
                        }
                    
                    box_color = self.class_color_mapping[int(class_id)]['box_color']
                    segmentation_color = self.class_color_mapping[int(class_id)]['segmentation_color']

                    # Resize mask to match the frame dimensions
                    mask_resized = cv2.resize(mask, (frame.shape[1], frame.shape[0]))

                    # Create an overlay with the alpha channel
                    color_mask = np.zeros_like(frame, dtype=np.uint8)
                    color_mask[mask_resized > 0] = segmentation_color

                    alpha_mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
                    alpha_mask[mask_resized > 0] = int(255 * GlobalConfig.instance_segmentation_transparency)

                    # Convert color_mask to BGRA
                    color_mask = cv2.cvtColor(color_mask, cv2.COLOR_RGB2RGBA)
                    color_mask[:, :, 3] = alpha_mask  # Add alpha channel to the mask

                    # Convert frame to RGBA
                    frame_rgba = cv2.cvtColor(frame, cv2.COLOR_RGB2RGBA)

                    # Blend the overlay with the original frame
                    overlay = cv2.addWeighted(frame_rgba, 1, color_mask, 0.5, 0)

                    # Convert back to RGB
                    frame = cv2.cvtColor(overlay, cv2.COLOR_RGBA2RGB)

                    # Draw the bounding box and label
                    cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), box_color, GlobalConfig.box_border_width)
                    label_text = f"{label} {score:.2f}"
                    
                    # Calculate text size
                    (text_width, text_height), baseline = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                    
                    # Draw label box
                    cv2.rectangle(frame, (int(box[0]), int(box[1]) - text_height - baseline), (int(box[0]) + text_width, int(box[1])), GlobalConfig.label_box_background, -1)
                    cv2.putText(frame, label_text, (int(box[0]), int(box[1]) - baseline), cv2.FONT_HERSHEY_SIMPLEX, 0.5, GlobalConfig.label_text_color, 1, lineType=cv2.LINE_AA)

                    # Draw the segmentation border if enabled
                    if GlobalConfig.instance_segmentation_border:
                        contours, _ = cv2.findContours(mask_resized, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        cv2.drawContours(frame, contours, -1, segmentation_color, GlobalConfig.instance_segmentation_border_width)

                    frame_data['results'].append({
                        'label': label,
                        'box': box,
                        'score': float(score)
                    })

        return frame, frame_data

if __name__ == "__main__":
    video_path = os.path.join(GlobalConfig.video_output_folder, 'combined_video.mp4')
    output_video = 'yolo_output'
    output_json = 'yolo_results'
    processor = YOLOv8Processor(video_path, output_video, output_json)
    processor.process_video()
