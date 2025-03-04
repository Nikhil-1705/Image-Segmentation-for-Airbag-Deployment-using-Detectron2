# Airbag Detection Using Detectron2

## 📌 Project Overview
This project utilizes **Detectron2**, a deep learning framework by Facebook AI, to detect airbags in **side-view vehicle videos**. It includes:
- **Model Training**: Training a Mask R-CNN model on a COCO-formatted dataset.
- **Inference & Segmentation**: Detecting airbags in video frames.
- **Annotation Generation**: Creating text-based annotations for detected objects.

## 🚀 Features
- **Dataset Handling**: Uses **Roboflow API** to load and split COCO dataset.
- **Frame Extraction**: Extracts frames from input videos.
- **Instance Segmentation**: Uses a trained **Mask R-CNN** model to detect airbags.
- **Visualization**: Draws bounding boxes and masks over detected airbags.
- **Annotation Generation**: Saves detected airbag coordinates into text files.
- **Output Management**: Saves results and compresses them into a zip file for easy sharing.

## 📂 Project Structure
```
📁 Airbag-Detection
├── 📄 training.py            # Training the model with Detectron2
├── 📄 inference.py           # Running inference and generating annotations
├── 📁 datasets               # Contains dataset downloaded via Roboflow
├── 📁 Output_frames          # Extracted video frames
├── 📁 Result                 # Processed images with segmentation
├── 📁 annotations_text       # Generated annotations for detected objects
├── 📄 README.md              # Project documentation
```

## 📌 Installation & Setup
### 🔹 Step 1: Clone the repository
```sh
git clone https://github.com/your-username/airbag-detection.git
cd airbag-detection
```
### 🔹 Step 2: Install dependencies
```sh
pip install -r requirements.txt
```
**OR** install the necessary libraries manually:
```sh
pip install torch torchvision torchaudio
pip install detectron2 roboflow opencv-python numpy
```
### 🔹 Step 3: Set up the dataset
- **Roboflow API Key:** Update `training.py` with your API key to download the dataset.

### 🔹 Step 4: Run the model
**Train the model**:
```sh
python training.py
```
**Run inference & generate annotations**:
```sh
python inference.py
```

## 📊 Results
- Extracted frames are stored in `Output_frames/`
- Processed images with segmentation are stored in `Result/`
- Annotation text files are stored in `annotations_text/`

## 📌 Future Improvements
- Optimize model performance with hyperparameter tuning.
- Deploy as a web app for user-friendly access.
- Improve annotation formats for better integration with labeling tools.

## 👨‍💻 Author
Developed by **[Nikhil Bhandari](https://linkedin.com/in/nikhil-bhandari17)** 🚀



![image alt](https://github.com/user-attachments/assets/a99a6e76-c9ec-44c5-adfe-67c53a9d0cfc)
![image_alt](https://github.com/user-attachments/assets/c43e1803-d642-43da-8865-c0ca841a9cd8)


