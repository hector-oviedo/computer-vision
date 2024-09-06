import sys
import os
import cv2
import shutil

from config import GlobalConfig

class VideoFrameExtractor:
    def __init__(self, video_path, frames_output_folder):
        """
        Initializes the VideoFrameExtractor.

        Args:
            video_path (str): Path to the input video file.
            frames_output_folder (str): Path to the folder where extracted frames will be saved.
        """
        self.video_path = video_path
        self.frames_output_folder = frames_output_folder

    def clear_output_folder(self):
        """Clears the frames output folder if it contains any files."""
        if os.path.exists(self.frames_output_folder):
            if os.listdir(self.frames_output_folder):  # Check if the folder is not empty
                shutil.rmtree(self.frames_output_folder)  # Remove all files and subdirectories
                os.makedirs(self.frames_output_folder, exist_ok=True)  # Recreate the folder
                print(f"Warning: Output folder '{self.frames_output_folder}' had files which were cleaned.")
        else:
            os.makedirs(self.frames_output_folder, exist_ok=True)  # Create the folder if it doesn't exist

    def extract_frames(self):
        """Extracts frames from the video and saves them to the output folder."""
        # Clear the output folder first
        self.clear_output_folder()

        # Open the video file
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video file '{self.video_path}'.")
            return

        # Get video properties
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        max_frames = int(total_frames * GlobalConfig.video_percentage)

        frame_id = 0
        while cap.isOpened() and frame_id < max_frames:
            ret, frame = cap.read()
            if not ret:
                break

            # Save each frame as an image in the output folder
            frame_filename = os.path.join(self.frames_output_folder, f'frame_{frame_id:04d}.png')
            cv2.imwrite(frame_filename, frame)
            frame_id += 1

        cap.release()
        print(f"Extracted {frame_id} frames to '{self.frames_output_folder}'.")

if __name__ == "__main__":
    # Path to the video file and the folder to store the extracted frames
    video_path = os.path.join(GlobalConfig.video_output_folder, GlobalConfig.combined_video_name)
    frames_output_folder = GlobalConfig.frames_input_folder

    # Initialize the VideoFrameExtractor
    extractor = VideoFrameExtractor(video_path, frames_output_folder)

    # Extract frames from the video
    extractor.extract_frames()