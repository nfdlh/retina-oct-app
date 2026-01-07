"""
Gradio Web Interface for Retinal OCT Classification with Grad-CAM++

A simple web interface to upload OCT images and get predictions with
explainable AI visualization.

Requirements:
    pip install gradio

Usage:
    python grad_cam_upload.py
"""

import gradio as gr
import numpy as np
from PIL import Image

from grad_cam import (
    CLASS_NAMES,
    DEFAULT_MODEL_PATH,
    apply_heatmap,
    generate_gradcam,
    get_device,
    load_model,
    predict,
    preprocess_image,
)


# Load model once at startup
print(f"Loading model on {get_device()}...")
MODEL = load_model(DEFAULT_MODEL_PATH)
print("Model loaded successfully!")


# Class descriptions for display
CLASS_INFO = {
    "CNV": {
        "name": "CNV (Choroidal Neovascularization)",
        "description": "Abnormal blood vessels growing in the choroid layer. These vessels leak fluid and blood, causing rapid vision loss.",
        "severity": "Requires immediate attention",
    },
    "DME": {
        "name": "DME (Diabetic Macular Edema)",
        "description": "Fluid buildup in the macula due to diabetes. Causes swelling and blurred or wavy central vision.",
        "severity": "Requires treatment",
    },
    "DRUSEN": {
        "name": "DRUSEN",
        "description": "Yellowish deposits under the retina. Early sign of age-related macular degeneration (AMD).",
        "severity": "Monitor regularly",
    },
    "NORMAL": {
        "name": "Normal",
        "description": "Healthy retina with no detected abnormalities.",
        "severity": "No issues detected",
    },
}


def analyze_image(
    image: Image.Image,
    cam_method: str = "gradcam++",
    overlay_alpha: float = 0.5,
) -> tuple:
    """
    Analyze an OCT image and return prediction with Grad-CAM visualization.

    Args:
        image: Input PIL Image
        cam_method: CAM method to use
        overlay_alpha: Heatmap overlay transparency

    Returns:
        Tuple of (visualization_image, result_text, confidence_dict)
    """
    if image is None:
        return None, "Please upload an image.", {}

    # Convert to RGB if needed
    if image.mode != "RGB":
        image = image.convert("RGB")

    # Preprocess
    input_tensor, rgb_img = preprocess_image(image)

    # Predict
    predicted_idx, probabilities = predict(MODEL, input_tensor)
    predicted_class = CLASS_NAMES[predicted_idx]

    # Generate Grad-CAM
    grayscale_cam = generate_gradcam(
        MODEL, input_tensor, predicted_idx, method=cam_method
    )

    # Create visualization
    heatmap_overlay = apply_heatmap(rgb_img, grayscale_cam, alpha=overlay_alpha)

    # Build confidence dictionary for Gradio Label component
    confidences = {
        CLASS_NAMES[i]: float(probabilities[0][i]) for i in range(len(CLASS_NAMES))
    }

    # Build result text
    info = CLASS_INFO[predicted_class]
    result_text = f"""
## {info["name"]}

**Status:** {info["severity"]}

### Description
{info["description"]}

### Confidence
{confidences[predicted_class] * 100:.1f}%
"""

    return heatmap_overlay, result_text, confidences


def create_interface() -> gr.Blocks:
    """Create and configure the Gradio interface."""

    with gr.Blocks(
        title="Retinal OCT Analysis",
        theme=gr.themes.Soft(),
    ) as demo:
        gr.Markdown(
            """
            # Retinal OCT Damage Detection

            Upload an OCT (Optical Coherence Tomography) image to detect retinal conditions.

            **Supported conditions:** CNV, DME, DRUSEN, or Normal

            > **Disclaimer:** This is a research tool for educational purposes only.
            > Not intended for clinical diagnosis. Always consult a qualified ophthalmologist.
            """
        )

        with gr.Row():
            with gr.Column(scale=1):
                # Input section
                input_image = gr.Image(
                    label="Upload OCT Image",
                    type="pil",
                    height=300,
                )

                with gr.Accordion("Advanced Options", open=False):
                    cam_method = gr.Dropdown(
                        choices=["gradcam++", "gradcam", "layercam", "eigencam"],
                        value="gradcam++",
                        label="XAI Method",
                    )
                    overlay_alpha = gr.Slider(
                        minimum=0.1,
                        maximum=0.9,
                        value=0.5,
                        step=0.1,
                        label="Heatmap Opacity",
                    )

                analyze_btn = gr.Button("Analyze Image", variant="primary")

            with gr.Column(scale=1):
                # Output section
                output_image = gr.Image(
                    label="Grad-CAM++ Visualization",
                    height=300,
                )
                confidence_label = gr.Label(
                    label="Class Probabilities",
                    num_top_classes=4,
                )
                result_text = gr.Markdown(label="Diagnosis Result")

        # Connect the analyze button
        analyze_btn.click(
            fn=analyze_image,
            inputs=[input_image, cam_method, overlay_alpha],
            outputs=[output_image, result_text, confidence_label],
        )

        # Also analyze when image is uploaded
        input_image.change(
            fn=analyze_image,
            inputs=[input_image, cam_method, overlay_alpha],
            outputs=[output_image, result_text, confidence_label],
        )

        gr.Markdown(
            """
            ---
            ### About This Tool

            This application uses a **VGG-16** deep learning model trained on retinal OCT images
            to classify four conditions:

            | Condition | Description |
            |-----------|-------------|
            | **CNV** | Choroidal Neovascularization - abnormal blood vessel growth |
            | **DME** | Diabetic Macular Edema - fluid buildup from diabetes |
            | **DRUSEN** | Lipid deposits under retina (early AMD sign) |
            | **NORMAL** | Healthy retina |

            The **Grad-CAM++** visualization shows which regions of the image the model
            focuses on when making its prediction (red = high attention, blue = low attention).

            ---
            *WQF7002 AI Techniques - Group Assignment | SDG 3: Good Health and Well-being*
            """
        )

    return demo


if __name__ == "__main__":
    demo = create_interface()
    demo.launch(
        share=False,  # Set to True to create a public link
        server_name="0.0.0.0",  # Allow external access
        server_port=7860,
    )
