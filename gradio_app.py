import gradio as gr
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from preprocess import preprocess_single_image
from gradcam import get_gradcam_heatmap, overlay_gradcam_on_image

# Load model
model = load_model("saved_models/best_model.h5")
model.predict(np.zeros((1, 128, 128, 3)))  # Warm-up call
LAST_CONV_LAYER = "conv2d_1"

def classify_and_explain(img):
    img_array = preprocess_single_image(img)
    pred = model.predict(img_array)[0][0]
    label = "Blurred" if pred > 0.5 else "Not Blurred"
    confidence = round(float(pred) * 100, 2) if pred > 0.5 else round((1 - float(pred)) * 100, 2)
    label_with_conf = f"{label} ({confidence}%)"

    # Generate Grad-CAM
    heatmap = get_gradcam_heatmap(model, img_array, last_conv_layer_name=LAST_CONV_LAYER)
    original = np.array(img.resize((128, 128))).astype(np.uint8)
    gradcam = overlay_gradcam_on_image(original, heatmap)

    caption = (
        "Grad-CAM shows where the model focused while making its prediction. "
        "Red/yellow areas = strong influence. Blue = weak or no influence."
    )

    return label_with_conf, Image.fromarray(original), Image.fromarray(gradcam), caption

# Create Gradio UI with white background and black text
custom_theme = gr.themes.Base(
    primary_hue="gray",
    neutral_hue="gray"
).set(
    body_background_fill="#ffffff",
    body_text_color="#000000"
)

with gr.Blocks(theme=custom_theme, title="CV Blur Detector with Grad-CAM") as demo:
    gr.Markdown("## 📷Is this image blurred?", elem_id="title")
    gr.Markdown(
        "Upload an image to classify it as **blurred or not blurred**. "
        "You'll see a Grad-CAM heatmap showing what the model focused on.",
        elem_id="subtitle"
    )

    image_input = gr.Image(type="pil", label="Upload Image")

    run_btn = gr.Button("🔍 Classify and Visualize")

    label_output = gr.Textbox(label="Prediction (with confidence)")

    with gr.Row():
        original_image = gr.Image(type="pil", label="Original Image")
        gradcam_image = gr.Image(type="pil", label="Grad-CAM Heatmap")

    explanation = gr.Textbox(label="Explanation", lines=2)

    run_btn.click(
        fn=classify_and_explain,
        inputs=image_input,
        outputs=[label_output, original_image, gradcam_image, explanation]
    )

demo.launch()
