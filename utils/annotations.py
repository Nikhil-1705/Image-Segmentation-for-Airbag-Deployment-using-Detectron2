import numpy as np
import cv2

def generate_annotations(image_dir, output_dir, predictor):
    os.makedirs(output_dir, exist_ok=True)
    for filename in os.listdir(image_dir):
        if filename.endswith((".jpg", ".png")):
            image_path = os.path.join(image_dir, filename)
            img = cv2.imread(image_path)
            outputs = predictor(img)
            
            if len(outputs["instances"].pred_masks) > 0:
                annotation_file_path = os.path.join(output_dir, os.path.splitext(filename)[0] + ".txt")
                with open(annotation_file_path, "w") as f:
                    for mask in outputs["instances"].pred_masks:
                        mask = mask.cpu().numpy().astype(np.uint8)
                        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        for contour in contours:
                            for point in contour:
                                x, y = point[0]
                                f.write(f"{x},{y} ")
                            f.write("\n")
                print(f"Annotations generated and saved to {annotation_file_path}")
