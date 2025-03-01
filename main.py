from data_preparation.dataset_loading import load_dataset
from data_preparation.data_splitting import register_datasets
from model.config import setup_config
from model.train import train_model
from model.inference import setup_predictor
from utils.frame_extraction import extract_frames
from utils.visualization import visualize_airbag_presence
from utils.annotations import generate_annotations
from inferences.airbag_presence import detect_airbag_in_video
from inferences.max_inflation import find_max_inflation_frame
from inferences.shoulder_line import visualize_shoulder_line

# Load dataset
dataset = load_dataset("OHbmvgNatuO4WaK5PtEp", "airbag-amymt", "airbag-glvdl", 3)
train_data_set_name, test_data_set_name, valid_data_set_name = register_datasets(dataset, "_annotations.coco.json")

# Setup config and train model
cfg = setup_config("mask_rcnn_R_101_FPN_3x", train_data_set_name, "output", 3, 200, 200, 0.001)
model_weights_path = train_model(cfg)

# Setup predictor
predictor = setup_predictor(cfg, model_weights_path)

# Extract frames from video
video_path = input("Enter path to your side view video file: ")
extract_frames(video_path, "/content/Output_frames")

# Visualize airbag presence
image_directory = "/content/Output_frames"
output_directory = "/content/Result"
os.makedirs(output_directory, exist_ok=True)
for image_file in os.listdir(image_directory):
    image_path = os.path.join(image_directory, image_file)
    visualize_airbag_presence(image_path, predictor, MetadataCatalog.get(train_data_set_name), output_directory)

# Generate annotations
generate_annotations(output_directory, "/content/annotations_text", predictor)

# Detect airbag in video
detect_airbag_in_video(video_path, predictor, MetadataCatalog.get(train_data_set_name))

# Find maximum inflation frame
find_max_inflation_frame("/content/annotations_text", "/content/Result")

# Visualize shoulder line
line_points = [(585, 130), (446, 130)]
visualize_shoulder_line("/content/Result/frame0001.jpg", "/content/annotations_text/frame0001.txt", line_points, "/content/shoulder_line.jpg")
