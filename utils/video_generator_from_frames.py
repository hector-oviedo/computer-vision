import os
import cv2
from config import GlobalConfig

class VideoGenerator:
    def __init__(self, model_name):
        """
        Initializes the VideoGenerator class.

        Args:
            model_name (str): The name of the model (used for constructing input and output paths).
        """
        self.model_name = model_name

        # Construct input and output paths based on the model name and config variables
        self.frames_input_folder = os.path.join(GlobalConfig.frames_output_folder, model_name)
        self.video_output_folder = GlobalConfig.video_output_folder
        self.video_filename = os.path.join(self.video_output_folder, f'{model_name}.mp4')

        # Ensure output video folder exists
        os.makedirs(self.video_output_folder, exist_ok=True)

    def generate_video(self):
        """
        Generates a video from the frames in the input folder and saves it as a .mp4 file in the output folder.
        If a file with the same name already exists, it is removed before generating the new video.
        """
        # Check if the video file already exists, and remove it if it does
        if os.path.exists(self.video_filename):
            os.remove(self.video_filename)
            print(f"Warning: Existing video file '{self.video_filename}' was removed.")

        # List all image files in the input folder and sort them
        frame_files = sorted([f for f in os.listdir(self.frames_input_folder) if f.endswith('.png')])
        if not frame_files:
            print(f"No frames found in the folder: {self.frames_input_folder}")
            return

        # Get the size of the first frame to define video resolution
        first_frame = cv2.imread(os.path.join(self.frames_input_folder, frame_files[0]))
        if first_frame is None:
            print(f"Error: Could not read the first frame from '{self.frames_input_folder}'")
            return
        
        height, width, _ = first_frame.shape

        # Create a VideoWriter object to write the video
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = 30  # Define the FPS (can be adjusted as needed)
        out = cv2.VideoWriter(self.video_filename, fourcc, fps, (width, height))
        print(f"VideoWriter initialized, writing to '{self.video_filename}'")

        # Write each frame to the video
        for frame_file in frame_files:
            frame_path = os.path.join(self.frames_input_folder, frame_file)
            frame = cv2.imread(frame_path)
            if frame is None:
                print(f"Error: Could not read frame '{frame_file}'. Skipping.")
                continue

            out.write(frame)

        # Release the VideoWriter object
        out.release()
        print(f"Video saved to '{self.video_filename}'")

if __name__ == "__main__":
    # Example usage: Generate a video for the specified model
    model_name = "detectron2"  # Replace with the actual model name
    video_generator = VideoGenerator(model_name)
    video_generator.generate_video()