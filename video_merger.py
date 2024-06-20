import sys
import os

# Add the utils directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'utils'))

from utils.combine_videos import combine_videos

def main():
    video_files = [
        '6059506-hd_1920_1080_30fps.mp4',
        '3696015-hd_1920_1080_24fps.mp4',
        '1508533-hd_1920_1080_25fps.mp4',
        '3769966-hd_1920_1080_25fps.mp4'
    ]
    combine_videos(video_files, "combined_video.mp4")

if __name__ == "__main__":
    main()