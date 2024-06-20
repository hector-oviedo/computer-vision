import cv2
import os
from config import GlobalConfig  # Import GlobalConfig

class VideoGenerator:
    def __init__(self, filename):
        self.output_path = os.path.join(GlobalConfig.video_output_folder, f"{filename}.mp4")
        if os.path.exists(self.output_path):
            os.remove(self.output_path)
        self.video_writer = None

    def start(self, frame_size, fps=30):
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.video_writer = cv2.VideoWriter(self.output_path, fourcc, fps, frame_size)

    def write_frame(self, frame):
        if self.video_writer is not None:
            self.video_writer.write(frame)

    def finish(self):
        if self.video_writer is not None:
            self.video_writer.release()
            self.video_writer = None