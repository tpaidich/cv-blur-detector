# Computer Vision Blur Detector

A computer vision project that classifies whether an image is **blurred** or **not blurred** using a custom-built Convolutional Neural Network (CNN). It includes a user-friendly Gradio interface and Grad-CAM visualizations to explain the model's decisions.

---

## Features

- Classifies images as blurred or not blurred
- Uses a custom CNN model
- Supports Grad-CAM visualizations to highlight what parts of the image influenced the decision
- Interactive Gradio web app with captioned overlays
- Side-by-side layout with prediction + heatmap
- Confidence score shown with each prediction

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

*(Insert your own screenshot of a Grad-CAM prediction here)*

---

## 📝 License

MIT License © 2025 Tanushree Paidichetty
