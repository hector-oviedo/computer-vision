import cv2
import os
from config import GlobalConfig # Import GlobalConfig

def combine_videos(video_files, output_file):
    output_path = os.path.join(GlobalConfig.video_output_folder, output_file)
    # Remove the existing combined video file if it exists
    if os.path.exists(output_path):
        os.remove(output_path)

    # Initialize a list to hold the video captures
    caps = [cv2.VideoCapture(os.path.join(GlobalConfig.video_input_folder, v)) for v in video_files]

    # Get properties from the first video (assuming all videos have the same properties)
    original_width = int(caps[0].get(cv2.CAP_PROP_FRAME_WIDTH))
    original_height = int(caps[0].get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(caps[0].get(cv2.CAP_PROP_FPS))

    # Create a VideoWriter object to write the combined video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (GlobalConfig.resize_size, GlobalConfig.resize_size))

    # Function to process and write frames from a video capture
    def process_and_write_frames(cap):
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Center crop the frame to 1080x1080
            start_x = (original_width - GlobalConfig.crop_size) // 2
            start_y = (original_height - GlobalConfig.crop_size) // 2
            cropped_frame = frame[start_y:start_y+GlobalConfig.crop_size, start_x:start_x+GlobalConfig.crop_size]

            # Resize the cropped frame to 512x512
            resized_frame = cv2.resize(cropped_frame, (GlobalConfig.resize_size, GlobalConfig.resize_size))

            # Write the processed frame to the output video
            out.write(resized_frame)

    # Process each video capture
    for cap in caps:
        process_and_write_frames(cap)
        cap.release()

    # Release the VideoWriter object
    out.release()
    print(f'Combined video saved to {output_path}')