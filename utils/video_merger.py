import cv2
import os
from config import GlobalConfig

def combine_videos(video_files, output_file):
    """Combines multiple video files into one, applying center cropping and resizing.

    Args:
        video_files (list): List of video filenames to combine.
        output_file (str): The name of the output video file.
    """
    output_path = os.path.join(GlobalConfig.video_output_folder, output_file)
    
    # Check if the output folder exists, if not, create it
    if not os.path.exists(GlobalConfig.video_output_folder):
        print(f"Output folder '{GlobalConfig.video_output_folder}' does not exist, creating it.")
        os.makedirs(GlobalConfig.video_output_folder, exist_ok=True)
    
    # Remove the existing combined video file if it exists
    if os.path.exists(output_path):
        print(f"Output file '{output_path}' already exists, removing it.")
        os.remove(output_path)

    # Initialize a list to hold the video captures
    caps = []
    for v in video_files:
        video_path = os.path.join(GlobalConfig.video_input_folder, v)
        if not os.path.exists(video_path):
            print(f"Error: Input video file '{video_path}' not found.")
            return
        caps.append(cv2.VideoCapture(video_path))
        print(f"Opened video file: {video_path}")

    # Get properties from the first video (assuming all videos have the same properties)
    original_width = int(caps[0].get(cv2.CAP_PROP_FRAME_WIDTH))
    original_height = int(caps[0].get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(caps[0].get(cv2.CAP_PROP_FPS))
    print(f"Video properties - Width: {original_width}, Height: {original_height}, FPS: {fps}")

    # Create a VideoWriter object to write the combined video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (GlobalConfig.resize_size, GlobalConfig.resize_size))
    print(f"VideoWriter initialized, writing to '{output_path}'")

    # Function to process and write frames from a video capture
    def process_and_write_frames(cap, video_filename):
        print(f"Processing video: {video_filename}")
        while True:
            ret, frame = cap.read()
            if not ret:
                print(f"Finished processing video: {video_filename}")
                break
            
            # Center crop the frame to the specified crop size
            start_x = (original_width - GlobalConfig.crop_size) // 2
            start_y = (original_height - GlobalConfig.crop_size) // 2
            cropped_frame = frame[start_y:start_y+GlobalConfig.crop_size, start_x:start_x+GlobalConfig.crop_size]

            # Resize the cropped frame to the specified resize size
            resized_frame = cv2.resize(cropped_frame, (GlobalConfig.resize_size, GlobalConfig.resize_size))

            # Write the processed frame to the output video
            out.write(resized_frame)

    # Process each video capture
    for i, cap in enumerate(caps):
        video_filename = video_files[i]
        process_and_write_frames(cap, video_filename)
        cap.release()
        print(f"Released video capture for {video_filename}")

    # Release the VideoWriter object after all videos are processed
    out.release()
    print(f'Combined video saved to {output_path}')

if __name__ == "__main__":
    video_files = [
        '6059506-hd_1920_1080_30fps.mp4',
        '3696015-hd_1920_1080_24fps.mp4',
        '1508533-hd_1920_1080_25fps.mp4',
        '3769966-hd_1920_1080_25fps.mp4'
    ]
    combine_videos(video_files, GlobalConfig.combined_video_name)