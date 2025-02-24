import cv2
import os
import sys
import glob
import torch

def extract_frames(video_path, output_dir):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Failed to open video: {video_path}")
        sys.exit(1)
    os.makedirs(output_dir, exist_ok=True)
    frame_number = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_path = os.path.join(output_dir, f"frame_{frame_number:04d}.jpg")
        cv2.imwrite(frame_path, frame)
        frame_number += 1
    cap.release()

def annotate_frames(input_dir, output_dir, model, vehicle_classes):
    os.makedirs(output_dir, exist_ok=True)
    for image_path in sorted(glob.glob(os.path.join(input_dir, "*.jpg"))):
        img = cv2.imread(image_path)
        if img is None:
            print(f"Could not read {image_path}")
            continue
        results = model(img)
        annotated_img = results.render()[0].copy()
        df = results.pandas().xyxy[0]
        vehicle_df = df[df['name'].isin(vehicle_classes)]
        counts = vehicle_df['name'].value_counts().to_dict()
        for cls in vehicle_classes:
            counts.setdefault(cls, 0)
        counter_text = " | ".join([f"{cls}: {counts[cls]}" for cls in vehicle_classes])
        height, width = annotated_img.shape[:2]
        position = (10, height - 10)
        cv2.putText(annotated_img, counter_text, position, 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        output_path = os.path.join(output_dir, os.path.basename(image_path))
        cv2.imwrite(output_path, annotated_img)
        print("Processed and saved:", output_path)

def create_video_from_frames(output_dir, mp4_output, fps=30):
    images = sorted(glob.glob(os.path.join(output_dir, "*.jpg")))
    if not images:
        raise ValueError("No images found in the annotated_frames directory.")
    frame = cv2.imread(images[0])
    height, width, _ = frame.shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(mp4_output, fourcc, fps, (width, height))
    for image_file in images:
        frame = cv2.imread(image_file)
        if frame is None:
            print(f"Skipping unreadable frame: {image_file}")
            continue
        video_writer.write(frame)
    video_writer.release()
    print(f"Video saved as {mp4_output}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python yolo_test.py <path_to_video>")
        sys.exit(1)
    video_path = sys.argv[1]
    if not os.path.isfile(video_path):
        print(f"Video file does not exist: {video_path}")
        sys.exit(1)
    frame_output_dir = "all_frames"
    annotated_output_dir = "annotated_frames"
    mp4_output = "output_video.mp4"
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
    vehicle_classes = ['car', 'truck', 'bus', 'motorcycle']
    extract_frames(video_path, frame_output_dir)
    annotate_frames(frame_output_dir, annotated_output_dir, model, vehicle_classes)
    create_video_from_frames(annotated_output_dir, mp4_output)
