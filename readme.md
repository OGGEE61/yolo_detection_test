# YOLOv5 Video Annotation Pipeline

This project processes a video by extracting individual frames, annotating each frame with YOLOv5 detections for vehicles (car, truck, bus, motorcycle), and compiling the annotated frames back into an output video.

## Features
- **Frame Extraction:** Reads and saves every frame from the input video.
- **Object Detection & Annotation:** Utilizes YOLOv5 to detect vehicles in each frame. Each frame is annotated with bounding boxes and a counter overlay for the specified vehicle classes.
- **Video Compilation:** Assembles the annotated frames into a final output video.

## Prerequisites
- Python 3.x
- OpenCV (`pip install opencv-python`)
- PyTorch (`pip install torch`)
- YOLOv5 (accessed via `torch.hub.load` from Ultralytics)

## Usage
Execute the script from the command line by providing the path to the video:
```bash
python yolo_test.py <path_to_video>
```

- The script verifies the video file exists and is readable.
- Frames are saved in the `all_frames` directory.
- Annotated frames are saved in the `annotated_frames` directory.
- The final output video is generated as `output_video.mp4`.
