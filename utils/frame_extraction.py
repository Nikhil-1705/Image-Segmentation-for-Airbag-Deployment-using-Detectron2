import cv2
import os

def extract_frames(video_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    for frame_number in range(frame_count):
        ret, frame = cap.read()
        if not ret:
            break
        frame_filename = f"{output_dir}/frame{frame_number:04d}.jpg"
        cv2.imwrite(frame_filename, frame)
    
    cap.release()
    print(f"Frames extracted and saved to {output_dir}")
