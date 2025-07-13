# Computer Vision Blur Detector

This project is a deep learning-based computer vision system that classifies whether an image is blurred or not blurred, using a custom-built Convolutional Neural Network (CNN). The model is trained on a dataset containing both defocus-blurred and motion-blurred images, as well as sharp images, organized into labeled subdirectories.

The CNN architecture is designed from scratch and consists of multiple convolutional and max-pooling layers to extract spatial features such as edges, texture smoothness, and detail sharpness - all key indicators of image clarity. I trained the model on [this image dataset](https://www.kaggle.com/datasets/kwentar/blur-dataset) I found on Kaggle, saving it in src/data/. 

To interpret model decisions, Grad-CAM (Gradient-weighted Class Activation Mapping) is used to generate heatmaps that visualize the regions the CNN considered most important during classification. For instance, in blurred images, the model may focus on edge regions or areas where detail is lost to make its decision.

The project also includes a Gradio-based interface for real-time interaction. Users can upload an image and receive:

- A prediction: "Blurred" or "Not Blurred"
- The model's confidence score
- A Grad-CAM heatmap displayed next to the image, with color cues indicating focus intensity (red for high attention, blue for low)

---

## Model Architecture

A simple CNN with two convolutional layers followed by fully connected layers:

Conv2D (32 filters) → MaxPooling  
→ Conv2D (64 filters) → MaxPooling  
→ Flatten → Dense (64) → Output (1, Sigmoid)

- Loss: `binary_crossentropy`  
- Optimizer: `adam`  
- Metric: `accuracy`

---

## Grad-CAM Explanation

Grad-CAM overlays help visualize which regions the model focused on during classification:

- 🔴 Red / 🟡 Yellow: High   attention (important areas)
- 🔵 Blue: Low attention

This makes the CNN more explainable and shows *why* the model classified an image as blurred or not.

---

## Gradio Interface

After launching the app, you can:

Upload an image  
- See if it’s blurred or not blurred
- View a Grad-CAM heatmap overlay  
- Get the model’s confidence score
- Enjoy a clean side-by-side layout with white background and black text

---

## How to Use

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Train the model:
```bash
python train.py
```

3. Launch the app:
```bash
python app/gradio_app.py
````
---

## Requirements

- tensorflow  
- opencv-python  
- numpy  
- scikit-learn  
- gradio  
- matplotlib  
- pillow

Install with:
```bash
pip install -r requirements.txt
```

---

## Sample Output
![alt text](https://github.com/tpaidich/cv-blur-detector/blob/main/example%20output.png?raw=true)

---
