import torch

class GlobalConfig:
    # Method to convert RGB to BGR
    @staticmethod
    def rgb_to_bgr(color):
        return color[::-1]

    # Video-combine processing settings
    crop_size = 1080  # Define crop square size (e.g., 1080 for 1080x1080 crop). Careful: dimensions must be x:y where x > y (e.g., 1920x1080)
    resize_size = 512  # Define the output dimension (512x512 for example is recommended for compatibility with EfficientDet)

    confidence_threshold = 0.5 # EfficientDet threshold score

    # border width of boxes for object recognition
    box_border_width = 1

    label_box_background = rgb_to_bgr.__func__((0, 0, 0))  # Black background
    label_text_color = rgb_to_bgr.__func__((255, 255, 255))  # White text

    instance_segmentation_border = True  # Enable/Disable border for instance segmentation
    instance_segmentation_border_width = 1
    instance_segmentation_transparency = 0.3

    # Colors for alternating objects
    object_colors = [
        rgb_to_bgr.__func__((255, 0, 0)),  # Red
        rgb_to_bgr.__func__((0, 255, 0)),  # Green
        rgb_to_bgr.__func__((0, 0, 255)),  # Blue
        rgb_to_bgr.__func__((255, 255, 0)),  # Yellow
        rgb_to_bgr.__func__((255, 0, 255)),  # Magenta
        rgb_to_bgr.__func__((0, 255, 255))   # Cyan
    ]

    # Colors for alternating segmentations
    segmentation_colors = [
        rgb_to_bgr.__func__((255, 0, 0)),  # Red
        rgb_to_bgr.__func__((0, 255, 0)),  # Green
        rgb_to_bgr.__func__((0, 0, 255)),  # Blue
        rgb_to_bgr.__func__((255, 255, 0)),  # Yellow
        rgb_to_bgr.__func__((255, 0, 255)),  # Magenta
        rgb_to_bgr.__func__((0, 255, 255))   # Cyan
    ]

    # General settings
    frames_input_folder = 'data/frames/'
    video_input_folder = 'data/videos/'

    frames_output_folder = 'output/frames/'
    video_output_folder = 'output/videos/'
    json_output_folder = 'output/logs/'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Name for the combined video
    combined_video_name = 'combined_video.mp4'

    # COCO labels:
    COCO_LABELS = {
        1: 'person', 2: 'bicycle', 3: 'car', 4: 'motorcycle', 5: 'airplane', 6: 'bus',
        7: 'train', 8: 'truck', 9: 'boat', 10: 'traffic light', 11: 'fire hydrant',
        12: 'stop sign', 13: 'parking meter', 14: 'bench', 15: 'bird', 16: 'cat',
        17: 'dog', 18: 'horse', 19: 'sheep', 20: 'cow', 21: 'elephant', 22: 'bear',
        23: 'zebra', 24: 'giraffe', 25: 'backpack', 26: 'umbrella', 27: 'handbag',
        28: 'tie', 29: 'suitcase', 30: 'frisbee', 31: 'skis', 32: 'snowboard',
        33: 'sports ball', 34: 'kite', 35: 'baseball bat', 36: 'baseball glove',
        37: 'skateboard', 38: 'surfboard', 39: 'tennis racket', 40: 'bottle',
        41: 'wine glass', 42: 'cup', 43: 'fork', 44: 'knife', 45: 'spoon', 46: 'bowl',
        47: 'banana', 48: 'apple', 49: 'sandwich', 50: 'orange', 51: 'broccoli',
        52: 'carrot', 53: 'hot dog', 54: 'pizza', 55: 'donut', 56: 'cake', 57: 'chair',
        58: 'couch', 59: 'potted plant', 60: 'bed', 61: 'dining table', 62: 'toilet',
        63: 'tv', 64: 'laptop', 65: 'mouse', 66: 'remote', 67: 'keyboard', 68: 'cell phone',
        69: 'microwave', 70: 'oven', 71: 'toaster', 72: 'sink', 73: 'refrigerator',
        74: 'book', 75: 'clock', 76: 'vase', 77: 'scissors', 78: 'teddy bear',
        79: 'hair drier', 80: 'toothbrush'
    }

    # Percentage of the video to process (0 to 1)
    video_percentage = 1