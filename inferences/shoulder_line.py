import cv2
import matplotlib.pyplot as plt

def check_on_line(segmentation, line_points):
    x1, y1 = line_points[0]
    x2, y2 = line_points[1]
    for point in segmentation:
        if (point[0] - x1) * (y2 - y1) == (point[1] - y1) * (x2 - x1):
            return True
    return False

def visualize_shoulder_line(image_path, annotation_path, line_points, save_path):
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    with open(annotation_path, "r") as file:
        annotations = file.readlines()
    segmentations = [parse_coordinates(annotation) for annotation in annotations]
    
    plt.figure(figsize=(10, 8))
    plt.imshow(img_rgb)
    plt.plot([line_points[0][0], line_points[1][0]], [line_points[0][1], line_points[1][1]], color='red', label='Shoulder Line')
    for segmentation in segmentations:
        x = [point[0] for point in segmentation]
        y = [point[1] for point in segmentation]
        plt.scatter(x, y, color='blue', s=5)
    plt.title('Image with Segmented Points and Shoulder Line')
    plt.axis('off')
    plt.legend()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
