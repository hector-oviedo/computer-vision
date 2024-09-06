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
from effdet import create_model

from config import GlobalConfig
from logger import Logger
from video_generator import VideoGenerator

class EfficientDetProcessor:
    def __init__(self, video_path, output_video, output_json):
        self.video_path = video_path
        self.logger = Logger(output_json)
        self.video_generator = VideoGenerator(output_video)
        self.device = GlobalConfig.device
        self.model = self.initialize_model()
        self.class_names = GlobalConfig.COCO_LABELS

    def initialize_model(self):
        model_name = 'tf_efficientdet_d7'  # Use the largest and most updated model
        model = create_model(model_name, bench_task='predict', num_classes=90, pretrained=True)
        model = model.to(self.device)
        model.eval()
        return model

    def process_frame(self, frame):
        scale_size = 1536  # EfficientDet-D7 expects 1536x1536 input
        frame_resized = cv2.resize(frame, (scale_size, scale_size))
        frame_tensor = torch.from_numpy(frame_resized / 255.0).permute(2, 0, 1).unsqueeze(0).float().to(self.device)

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

            box[0] = int(box[0] * frame.shape[1] / scale_size)
            box[1] = int(box[1] * frame.shape[0] / scale_size)
            box[2] = int(box[2] * frame.shape[1] / scale_size)
            box[3] = int(box[3] * frame.shape[0] / scale_size)

            box_color = GlobalConfig.object_colors[label % len(GlobalConfig.object_colors)]
            cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), box_color, GlobalConfig.box_border_width)
            label_text = f"{label_name} {score:.2f}"
            (text_width, text_height), baseline = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(frame, (int(box[0]), int(box[1]) - text_height - baseline), (int(box[0]) + text_width, int(box[1])), GlobalConfig.label_box_background, -1)
            cv2.putText(frame, label_text, (int(box[0]), int(box[1]) - baseline), cv2.FONT_HERSHEY_SIMPLEX, 0.5, GlobalConfig.label_text_color, 1, lineType=cv2.LINE_AA)
        
        return frame, frame_data

    def process_video(self):
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
            processed_frame, frame_data = self.process_frame(frame)
            inference_time = time.time() - start_time

            self.video_generator.write_frame(processed_frame)
            self.logger.log_frame(frame_id, 'EfficientDet', frame_data['results'], inference_time)
            frame_id += 1

            # Clear cache to free up memory
            del processed_frame, frame_data
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

if __name__ == '__main__':
    video_path = os.path.join(GlobalConfig.video_output_folder, 'combined_video.mp4')
    output_video = 'efficientdet_output'
    output_json = 'efficientdet_results'
    processor = EfficientDetProcessor(video_path, output_video, output_json)
    processor.process_video()
