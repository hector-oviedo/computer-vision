import torch

class GlobalConfig:
    # Method to convert RGB to BGR
    @staticmethod
    def rgb_to_bgr(color):
        return color[::-1]

    # Video-combine processing settings
    crop_size = 1080  # Define crop square size (e.g., 1080 for 1080x1080 crop). Careful: dimensions must be x:y where x > y (e.g., 1920x1080)
    resize_size = 512  # Define the output dimension (512x512 recommended for compatibility with EfficientDet)

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
    video_input_folder = 'data/videos/'
    video_output_folder = 'data/output/videos/'
    json_output_folder = 'data/output/logs/'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Percentage of the video to process (0 to 1)
    video_percentage = 1

    # COCO labels:
    COCO_LABELS = {
        0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', 5: 'bus',
        6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic light', 10: 'fire hydrant',
        11: 'stop sign', 12: 'parking meter', 13: 'bench', 14: 'bird', 15: 'cat',
        16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow', 20: 'elephant', 21: 'bear',
        22: 'zebra', 23: 'giraffe', 24: 'backpack', 25: 'umbrella', 26: 'handbag',
        27: 'tie', 28: 'suitcase', 29: 'frisbee', 30: 'skis', 31: 'snowboard',
        32: 'sports ball', 33: 'kite', 34: 'baseball bat', 35: 'baseball glove',
        36: 'skateboard', 37: 'surfboard', 38: 'tennis racket', 39: 'bottle',
        40: 'wine glass', 41: 'cup', 42: 'fork', 43: 'knife', 44: 'spoon', 45: 'bowl',
        46: 'banana', 47: 'apple', 48: 'sandwich', 49: 'orange', 50: 'broccoli',
        51: 'carrot', 52: 'hot dog', 53: 'pizza', 54: 'donut', 55: 'cake', 56: 'chair',
        57: 'couch', 58: 'potted plant', 59: 'bed', 60: 'dining table', 61: 'toilet',
        62: 'tv', 63: 'laptop', 64: 'mouse', 65: 'remote', 66: 'keyboard', 67: 'cell phone',
        68: 'microwave', 69: 'oven', 70: 'toaster', 71: 'sink', 72: 'refrigerator',
        73: 'book', 74: 'clock', 75: 'vase', 76: 'scissors', 77: 'teddy bear',
        78: 'hair drier', 79: 'toothbrush'
    }
