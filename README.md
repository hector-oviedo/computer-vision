# Computer Vision Module for IU University - Research Project

This repository contains a research tool and setup for a computer vision module developed as part of a project for IU University. The primary objective of this repository is to test, implement, and experiment with state-of-the-art object detection and segmentation models, specifically focusing on video inputs. The project primarily utilizes the YOLOv8 segmentation model (yolov8x-seg) due to its latest features supporting both object detection and segmentation tasks.

## Project Overview
### Objective
The main goal of this repository is to compare different computer vision models by applying them to video data and evaluating their performance in terms of object detection and segmentation. YOLOv8 was chosen for its simplicity and effectiveness, while EfficientDet was considered but ultimately discarded due to its complexity and challenges in implementing segmentation without a manual backbone model.

# Videos Used
The project uses a set of four diverse videos to test the models:

Street Video (6059506-hd_1920_1080_30fps.mp4): A clear video of people and cars on a street.
Top View Street Video (3696015-hd_1920_1080_24fps.mp4): A challenging top-view of street traffic, testing the model's inference on uncommon perspectives.
Snow Forest Video (1508533-hd_1920_1080_25fps.mp4): Features deers in a snowy environment, posing challenges with obstruction and uncommon object labels.
Interior House Video (3769966-hd_1920_1080_25fps.mp4): An indoor scene with a person walking around a room.
These videos are combined into a single video using the combine_videos.py script, located in the utils folder, to streamline the testing process.

# Videos from:

George Morina
https://www.pexels.com/video/crowded-street-under-the-rain-6059506/
December 3rd, 2020

Anthony
https://pexels.com/video/a-group-of-deer-at-winter-1508533/
October 14th, 2018

Kelly
https://www.pexels.com/video/top-view-footage-of-the-vehicles-crossing-the-roads-3696015/
February 7th, 2020

Taryn Elliott
https://www.pexels.com/video/woman-in-green-romper-pouring-water-on-the-plant-3769966/
February 19th, 2020


# Model Comparison
The repository contains scripts to test and evaluate two models:

- YOLOv8 Segmentation (yolo.py)

- EfficientDet (efficientdet.py)

- Detectron2 (detectron2.py) (only on Linux environments).

- SAM2 (still working on it).

# Documentation and Usage

### Overview

The utils folder contains essential scripts that facilitate the configuration of the project, handling of video data, frame extraction, and video generation processes for the models. These scripts will help with the workflow, making it easier to manage the input and output of data during model inference, and help us to compare easily the models.

The key scripts in the utils folder include:
- config.py – Central configuration file that stores parameters for the project, such as folder paths, color schemes, model settings, and more.
- video_merger.py – Combines multiple videos into a single file.
- video_frame_extractor.py – Extracts frames from a video and saves them into a folder for model inference.
- video_generator_from_frames.py – Combines processed frames back into a video after inference.

### Code Documentation and Comments

Each file in this repository is carefully documented and contains detailed comments to provide clear guidance on how the code works. The scripts are designed to be intuitive, with in-line explanations of key processes and decisions. This ensures that users can understand the purpose and functionality of each part of the code without needing to reference external documentation. To avoid redundancy, the individual code blocks will not be discussed in detail in this README. Instead, users are encouraged to explore the comments within the scripts themselves for deeper insights into how they work.

### Folder Structure

```
.
├── data/
│   ├── videos/                  # Unedited videos (source videos)
│   ├── frames/                  # Processed video frames (to feed the models)
│
├── models/                      # Store the model weights (YOLOv8, EfficientDet, etc.)
│
├── output/
│   ├── frames/                  # Subfolder per model (stores inference frames)
│   │   └── [model_name]/        # Inference frames of a specific model (e.g., YOLOv8 or EfficientDet)
│   │
│   ├── logs/                    # JSON files with frame-by-frame inference details (used for performance comparison)
│   │   └── [model_name].json    # Logs for each model's inference process (e.g., inference time, memory usage)
│   │
│   └── videos/                  # Combined video and output videos after scripts processing
│       └── [model_name].mp4     # Final output video generated from inference frames and the script `utils/video_generator_from_frames.py`
│
├── yolo.py                      # YOLOv8 model script for performing inference
├── efficientdet.py              # EfficientDet model script for performing inference
├── detectron2.py                # Detectron2 model script for performing inference
└── utils/                       # Utility scripts for various processes
    ├── config.py                # Central configuration for paths, colors, model settings, etc.
    ├── video_merger.py          # Script to merge multiple videos into one
    ├── video_frame_extractor.py # Extracts frames from a video for model input
    └── video_generator_from_frames.py # Combines inference frames into a video
```

# Inference Workflow

### Initial Setup - Check Configurations:
Before starting the process, ensure that filenames and folder paths are compatible with your environment by checking and editing `utils/config.py`. The configuration file allows you to define the video names, folder paths, and other necessary settings.

### Video Preparation - Merge or Use Single Video:
If you plan to combine multiple videos into one for testing, use the script `utils/video_merger.py`. This script merges videos listed in `config.py` and saves the combined video in the designated folder.
Skip this step if you intend to use a single video directly without merging, in this case, simply set up the `config.py` file to aim your video correctly.

### Frame Extraction:
After you have the desired video (merged or single), execute the script `utils/video_frame_extractor.py`. This script will take the video specified in `config.py`, extract its frames, and save them inside the `data/frames/` folder for model processing.

### Model Inference:
With the extracted frames in `data/frames/`, you can proceed with model inference by running the respective root scripts: `yolo.py` for YOLOv8 or `efficientdet.py` for EfficientDet, `detectron2.py` for Detectron2 (detectron2 only on Linux environments - read official docs [here](https://detectron2.readthedocs.io/en/latest/tutorials/install.html)).
These scripts load the corresponding model and process the extracted frames. The inference results, are saved as frames in the folder output/frames/[model_name]/.
In addition to the frames, detailed logs of the inference process (e.g., frame processing time, GPU/CPU usage) are saved as JSON files in `output/logs/`.

### Video Generation (Optional):
If you wish to compile the processed frames into a video, you can use the `utils/video_generator_from_frames.py` script.
For each model, go into the 'video_generator_from_frames.py' script and specify the model name for which you want to generate a video. This will compile the frames in output/frames/[model_name] into a video and save it as output/videos/[model_name].mp4.

By following these steps, you can seamlessly manage video preparation, frame extraction, model inference, and video generation in the specified folders, with logs to analyze the performance of each model.

This structure allows for clear separation of raw video data, intermediate frames, model-specific outputs, and final results, making it easy to compare model performance and results at each stage.

### PyTorch Installation

To install PyTorch with the appropriate CUDA configuration for your environment, please follow the official PyTorch installation guide: [PyTorch Installation](https://pytorch.org/get-started/locally/).

**Important Note:** The `requirements.txt` file provided in this repository does not include the PyTorch and CUDA-specific packages, as these should be installed according to your system's hardware and CUDA version. Please follow the instructions on the PyTorch website to ensure compatibility with your setup.

For ease of setup, we recommend using a Python 3.8 environment created with `conda`. To create and activate a new environment, use the following commands:

```
conda create -n cv_project python=3.8
conda activate cv_project
```

Once your environment is activated, follow the PyTorch installation instructions based on your operating system, Python version, and CUDA capabilities.

## Future Work

### React Frontend for Model Comparison

I am currently developing a **React-based frontend** that will allow users to easily load and visualize the inference data stored in the JSON files and compare the performance of different models. This frontend will enable side-by-side comparisons of:
- Inference times and performance metrics logged in the JSON files.
- Inference frames generated by the different models, allowing for a visual comparison of object detection and segmentation results.

Once complete, the frontend will be hosted in a separate repository, making it simple to integrate with the inference data generated by this project.

### SAM2 Model Integration

In addition to the current models (YOLOv8, EfficientDet, and Detectron2), I am working on integrating a script with the **SAM2 model** (Segment Anything Model v2) into this repo. SAM2 is a highly flexible segmentation model designed for a wide range of tasks, and its inclusion will allow us to extend the scope of the project by adding cutting-edge segmentation techniques.

