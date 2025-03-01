import os
import cv2

def calculate_segmentation_area(segmentation):
    n = len(segmentation)
    area = 0
    for i in range(n):
        j = (i + 1) % n
        area += segmentation[i][0] * segmentation[j][1]
        area -= segmentation[j][0] * segmentation[i][1]
    return abs(area) / 2

def find_max_inflation_frame(annotations_dir, result_dir):
    max_area = 0
    max_frame = None
    for filename in os.listdir(annotations_dir):
        if filename.endswith(".txt"):
            with open(os.path.join(annotations_dir, filename), "r") as f:
                annotations = f.readlines()
            segmentations = [parse_coordinates(annotation) for annotation in annotations]
            for segmentation in segmentations:
                area = calculate_segmentation_area(segmentation)
                if area > max_area:
                    max_area = area
                    max_frame = filename
    if max_frame:
        image_path = os.path.join(result_dir, max_frame.replace(".txt", ".jpg"))
        max_inflation_frame = cv2.imread(image_path)
        cv2.imwrite("/content/MaxInflationFrame.jpg", max_inflation_frame)
        print("Maximum inflation frame saved.")
