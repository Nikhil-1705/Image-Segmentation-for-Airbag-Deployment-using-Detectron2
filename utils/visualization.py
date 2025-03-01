from detectron2.utils.visualizer import Visualizer, ColorMode
import cv2

def visualize_airbag_presence(image_path, predictor, metadata, output_dir):
    img = cv2.imread(image_path)
    outputs = predictor(img)
    visualizer = Visualizer(img[:, :, ::-1], metadata=metadata, scale=0.8, instance_mode=ColorMode.IMAGE_BW)
    
    if "instances" in outputs:
        instances = outputs["instances"].to("cpu")
        out = visualizer.draw_instance_predictions(instances)
    else:
        out = visualizer.draw_instance_predictions({})
    
    output_path = os.path.join(output_dir, os.path.basename(image_path))
    cv2.imwrite(output_path, out.get_image()[:, :, ::-1])
    
    if "instances" in outputs and len(outputs["instances"]) > 0:
        print(f"Airbag is present in {image_path}. Saved to {output_path}")
    else:
        print(f"No airbag in {image_path}. Saved to {output_path}")
