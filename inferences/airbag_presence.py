import cv2
from detectron2.engine import DefaultPredictor
from detectron2.data import MetadataCatalog

def detect_airbag_in_video(video_path, predictor, metadata):
    cap = cv2.VideoCapture(video_path)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        outputs = predictor(frame)
        if len(outputs["instances"]) > 0:
            print("Airbag detected!")
            cap.release()
            return True
    cap.release()
    return False
