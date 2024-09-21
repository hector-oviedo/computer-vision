# Computer Vision Module for Object Detection and Segmentation

## Introduction
This repository contains the tools for a research project focused on object detection and segmentation from video sequences. The primary goal is to test, implement, and compare multiple state-of-the-art (SoA) computer vision models for object recognition, specifically evaluating their performance in video-based tasks. This module is part of a broader project aimed at understanding and evaluating SoA models for video object recognition and segmentation in various scenarios.

The models used in this project include YOLOv8, EfficientDet, Detectron2, and SAM2, all of which are integrated into the provided pipeline to facilitate benchmarking and performance comparisons across different input videos.

## Task Overview
This project addresses a university task focused on developing a system for recognizing and segmenting objects in video sequences. The task requires the evaluation of three state-of-the-art computer vision models using a representative dataset, focusing on object detection and segmentation. The system should return the video with identified objects, their position, shape, and labels. Additionally, the task calls for an evaluation of the models based on validity, reliability, and objectivity, alongside a qualitative analysis.

**Key Objectives:**
- Develop a system for object detection and segmentation in videos.
- Evaluate three state-of-the-art models on a representative dataset.
- Analyze model performance in terms of validity, reliability, and objectivity.
- Provide a video demonstrating the best performing model with labeled objects.

## Installation

### Environment Setup

This project requires setting up two separate environments due to version compatibility constraints. YOLOv8, EfficientDet, and Detectron2 require a Python 3.8 environment, while SAM2 requires Python 3.10 or higher. Below are the steps to set up both environments.

#### Setting up YOLO, EfficientDet, and Detectron2 (Python 3.8):

1. Create a Python 3.8 environment using Conda or Miniconda:
    ```bash
    conda create --name cv_project python=3.8
    conda activate cv_project
    ```

2. Install the required libraries:
    ```bash
    pip install opencv-python
    pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    ```
	For more details: [Pytorch Documentation](https://pytorch.org/get-started/locally/)

3. Install the specific model libraries:
    - YOLOv8:
      ```bash
      pip install ultralytics
      ```
      For more details: [YOLO Documentation](https://docs.ultralytics.com/models/)
      
    - EfficientDet:
      ```bash
      pip install effdet
      ```
      For more details: [EfficientDet Repository](https://github.com/google/automl/tree/master/efficientdet)

    - Detectron2 (Linux recommended, Windows with WSL supported but not optimal):
      ```bash
      pip install 'git+https://github.com/facebookresearch/detectron2.git'
      ```
      For more details: [Detectron2 Installation Guide](https://detectron2.readthedocs.io/en/latest/tutorials/install.html)

#### Setting up SAM2 (Python 3.10 or higher):

1. Create a Python 3.10 environment using Conda or Miniconda:
    ```bash
    conda create --name cv_project_sam python=3.10
    conda activate cv_project_sam
    ```

2. Install the required libraries:
    ```bash
    pip install opencv-python
    pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    ```

3. Clone the SAM2 repository and install it:
    ```bash
    git clone https://github.com/facebookresearch/segment-anything-2.git
    cd segment-anything-2
    pip install -e ".[demo]"
    ```

4. Go back to the project folder:
    ```bash
    cd ..
    ```
For more details of SAM2: [segment-anything-2]((https://github.com/facebookresearch/segment-anything-2))

### Operating System Notes

- **Linux**: Highly recommended for better performance and full compatibility with all models.
- **Windows**: YOLO and EfficientDet can run natively, but Detectron2 and SAM2 are only supported via Windows Subsystem for Linux (WSL), which may reduce performance during inference.

## Project Workflow

This project involves several stages to process videos, run object detection and segmentation models, for later analyze the output. Below are the key steps for working with the provided tools.

### 1. Video Preparation
To begin, prepare the videos you wish to analyze by merging them into a single video using the `video_merger.py` script. This script allows you to combine multiple video files and crop or resize them for model compatibility. Later in this README you will find the videos used in my specific case study.


**Steps:**
1. Place your videos in the `data/videos/` folder (or the folder specified in `config.py`).
2. Edit the `video_files` variable in `video_merger.py` to include the filenames of your videos.
3. Run the script to combine the videos:
    ```bash
    python utils/video_merger.py
    ```

This will generate a combined video file based on the configuration settings.

### 2. Frame Extraction
After preparing the video, you can automatically extract individual frames using the `video_frame_extractor.py` script. This will convert the video into frames for model inference.

**Steps:**
1. Ensure your video path is correctly set in the `config.py` file (under `video_input_folder` and `combined_video_name`).
2. Run the following command:
    ```bash
    python utils/video_frame_extractor.py
    ```

The frames will be saved in the folder specified in the `config.py` file (`data/frames/` by default).

### 3. Model Inference
Once the frames are extracted, run the object detection and segmentation models on the frames using the following scripts for each model:

- **YOLOv8**:
    ```bash
    python yolo.py
    ```
- **EfficientDet**:
    ```bash
    python efficientdet.py
    ```
- **Detectron2**:
    ```bash
    python detectron2_script.py
    ```
- **SAM2**:
    ```bash
    python sam2_script.py
    ```

Each script will process the frames, perform inference, and save the results in the output folders. Detailed logs and performance metrics (e.g., inference time, GPU usage) will be saved as JSON files in the `output/logs/` directory.
**Important: Do not forget activate the correct environment for each script**

### 4. Video Generation
After model inference, you can generate a video from the processed frames using the `video_generator_from_frames.py` script. This step allows you to visualize the results in video form.

**Steps:**
1. Edit the `model_name` variable in `video_generator_from_frames.py` to match the model whose output you want to compile.
2. Run the script to generate the video:
    ```bash
    python utils/video_generator_from_frames.py
    ```

The generated video will be saved in the folder specified in the `config.py` file (`data/videos/` by default).

## Models JSON Output Structure

The results of the model inference, including processed frames, logs, and videos, are stored in specific folders within the project. Below is an overview of how the output is organized:

- **Processed Frames**:
  - The processed frames after model inference are saved in the `output/frames/[model_name]/` folder.
  - Each frame corresponds to an input frame from the video, overlaid with the bounding boxes and segmentation masks produced by the model.

- **Logs**:
  - Logs are generated as JSON files and saved in the `output/logs/` folder.
  - Each log file includes frame-by-frame performance metrics such as:
    - Inference time (ms)
    - CPU/GPU usage
    - Detected objects and their confidence scores
  - Example of a log file (`output/logs/yolo.json`):
    ```json
    {
      "device": "cuda",
      "model": "yolo",
      "model_identifier": "yolov8xseg",
      "total_frames": 2657,
      "total_inference_time_ms": 60498.5,
      "frames": [
        {
          "frame_number": 0,
          "inference_time_ms": 21.33,
          "cpu_usage": 6.3,
          "detections": [
            {
              "label": "person",
              "box": [207, 109, 322, 405],
              "score": 0.91
            }
			...
          ]
        }
		...
      ]
    }
    ```

- **Final Video**:
  - The final output videos generated after inference (optional step) are saved in the `data/videos/` folder.
  - Each video contains the processed frames, combining object detection and segmentation outputs for visualization.

## Limitations

- **SAM2 Limitations**:
  - **Segmentation Only**: SAM2 is a segmentation model and does not perform object detection natively. It is designed to work with object detection models, where the detections are passed to SAM2 for segmentation.
  - **Human Feedback**: SAM2 is optimized for interactive use, where a user clicks on an object to track in a video. In our case, we worked around this limitation by hardcoding the position in the `sam2_detect_objects.py` file.
  - **Short Testing**: Due to computational limitations and SAM2's complexity, we tested it on a small range of video frames.

- **Detectron2 & SAM2 on Windows**: Detectron2 & SAM2 does not natively run on Windows and requires Windows Subsystem for Linux (WSL), which can lead to performance issues during inference. Native Linux is recommended for better results.

- **EfficientDet and YOLO Performance**: While both models work well on Windows and Linux, Linux is recommended for optimal performance.

## Additional Resources

- **Frontend Repository**: 
  - A companion frontend repository has been developed to visualize the results, compare models, and analyze metrics such as inference time and GPU usage. You can view the frontend here:
    [Computer Vision Frontend](https://github.com/hector-oviedo/computer-vision-frontend)

- **Videos Used**:
  - The videos used for testing in this project are publicly available from Pexels:
    - [Street Video by George Morina](https://www.pexels.com/video/crowded-street-under-the-rain-6059506/)
    - [Top View Street Video by Kelly](https://www.pexels.com/video/top-view-footage-of-the-vehicles-crossing-the-roads-3696015/)
    - [Snow Forest Video by Anthony](https://www.pexels.com/video/a-group-of-deer-at-winter-1508533/)
    - [Interior House Video by Taryn Elliott](https://www.pexels.com/video/woman-in-green-romper-pouring-water-on-the-plant-3769966/)

### Repo Folder Structure

```
.
├── data/
│   ├── videos/                  # Unedited videos, combined video, and output videos after scripts processing (source videos)
│   ├── frames/                  # Processed video frames (to feed the models)
│
├── output/
│   ├── frames/                  # Subfolder per model (stores inference frames)
│   │   └── [model_name]/        # Inference frames of a specific model (e.g., YOLOv8 or EfficientDet)
│   │
│   ├── logs/                    # JSON files with frame-by-frame inference details (used for performance comparison)
│   │   └── [model_name].json    # Logs for each model's inference process (e.g., inference time, memory usage)
│
├── yolo.py                      # YOLO model script for performing inference
├── efficientdet.py              # EfficientDet model script for performing inference
├── detectron2_script.py         # Detectron2 model script for performing inference
├── sam2_script.py         		 # SAM2 model script for performing inference
├── sam2_detect_objects.py       # SAM2 helper script for simulate object detection

└── utils/                       # Utility scripts for various processes
    ├── config.py                # Central configuration for paths, colors, model settings, etc.
    ├── video_merger.py          # Script to merge multiple videos into one
    ├── video_frame_extractor.py # Extracts frames from a video for model input
    └── video_generator_from_frames.py # Combines inference frames into a video
```

## License

This project is licensed under the MIT License, which permits free use, distribution, and modification of the software for both personal and commercial purposes. This project is intended for educational purposes, and contributions to improve or expand the project are welcomed.

You are free to:
- Use the software for personal or commercial projects.
- Modify the software to suit your own needs.
- Share and distribute the software as you see fit.
- Contribute to this project through pull requests or by reporting issues.
