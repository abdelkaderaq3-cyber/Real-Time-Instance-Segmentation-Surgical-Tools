# 🏥 Real-Time Instance Segmentation for Surgical Tools

## 🔍 Overview

This project focuses on real-time instance segmentation of surgical tools using computer vision and deep learning techniques.

## 🧠 Key Idea

The system detects and segments surgical instruments in real-time video frames, identifying each tool separately.

## 🧩 System Architecture

Input Frame
↓
Preprocessing
↓
Segmentation Model
↓
Instance Masks + Labels

``` 

## 📂 Project Structure

Real-Time-Instance-Segmentation-Surgical-Tools/
│
├── data/
│   └── sample_frames.txt
│
├── images/
│   ├── architecture.png
│   ├── pipeline.png
│   └── output_example.png
│
├── model/
│   └── segmentation_model.py
│
├── utils/
│   └── preprocessing.py
│
├── train.py
├── evaluate.py
├── main.py
├── requirements.txt
└── README.md

``` 

## 🖼️ Project Images

### 🔹 System Architecture

![Architecture](images/architecture.png)

### 🔹 Processing Pipeline

![Pipeline](images/pipeline.png)

### 🔹 Output Example

![Output](images/output_example.png)

## ⚙️ How It Works

1. Capture video frame
2. Preprocess image
3. Apply segmentation model
4. Generate masks for each tool
5. Display results in real-time

## ▶️ Run the Project

python main.py

## 📊 Example Output

Detected Tools: Scissors, Forceps
Masks Generated Successfully

## 🛠️ Technologies Used

* Python
* Computer Vision
* Deep Learning
* Instance Segmentation

## 🚀 Future Improvements

* Use Mask R-CNN or YOLOv8 Segmentation
* Improve real-time performance
* Add surgical dataset training

## 👨‍💻 Author

Abdelkader Abdelkarim

