# 🚗 Airbag Deployment Analysis using Detectron2

## 📌 Overview
This project utilizes **Detectron2** for airbag deployment analysis in side-view car crash videos. It performs **image segmentation** to extract crucial frames that help analyze airbag behavior, specifically:

- **Stitch Opening Frame** 🧵➡️ First moment the airbag starts deploying.
- **Maximum Inflation Frame** 🎈📈 Frame where the airbag reaches full inflation.
- **Shoulder Line Frame** 🏋️‍♂️📏 Frame where the airbag crosses the driver's shoulder line.

## 🛠️ Technologies Used
- **Python** 🐍
- **Detectron2** 🧠
- **OpenCV** 🎥
- **COCO Dataset Format** 📊
- **Roboflow** 🤖

---

## 🚀 Problem Statements & Solutions

### 1️⃣ Stitch Opening Frame 🧵➡️
**Objective:** Identify the first frame where the airbag begins to emerge from the stitched panel.

**Logic Used:**
- Detect airbag segmentation mask across all frames.
- Find the first frame where the pixel area of the detected mask exceeds a predefined threshold.

### 2️⃣ Maximum Inflation Frame 🎈📈
**Objective:** Identify the frame where the airbag is fully inflated.

**Logic Used:**
- Track the area of the segmented airbag in each frame.
- The frame with the largest detected area is considered the maximum inflation frame.

### 3️⃣ Shoulder Line Frame 🏋️‍♂️📏
**Objective:** Detect the frame where the airbag crosses a driver’s shoulder level.

**Logic Used:**
- Define a **fixed horizontal line** at shoulder level based on prior calibration.
- Identify the first frame where the airbag mask overlaps this predefined line.

---

## 🔧 Installation
```bash
# Clone the repository
git clone https://github.com/yourusername/airbag-detection.git
cd airbag-detection

# Install dependencies
pip install -r requirements.txt
```

---

## 📌 Usage
```python
# Run airbag detection on a video
python airbag_analysis.py --video_path path_to_your_video.mp4
```

**Outputs:**
- **Extracted Frames** 📷
- **Segmented Airbag Images** 🖼️
- **Annotated Text Files** 📜

---

## 📊 Results & Findings
- **Accurate detection** of airbag deployment frames with minimal false positives.
- **Automated inference** for crucial airbag behavior metrics.
- **Scalable & adaptable** for different car models and crash test scenarios.

📩 *For improvements or suggestions, feel free to contribute!*



![image](https://github.com/user-attachments/assets/e018ae32-a346-48df-b3cd-8457d5501591)
![image](https://github.com/user-attachments/assets/799ac180-a1da-400a-969e-a62fcc13cade)
![image](https://github.com/user-attachments/assets/37b6d37f-3bb9-4ded-9647-f511a27972b2)


