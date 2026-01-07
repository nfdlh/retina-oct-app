"""
Retinal OCT Classification - Hugging Face Spaces Application

This is the entry point for Hugging Face Spaces deployment.
It provides a Gradio interface for the retinal OCT classification model.

Usage:
    - Hugging Face Spaces will automatically run this file
    - For local testing: python app.py
"""

import os
from pathlib import Path
from typing import Dict, Tuple

import gradio as gr
import numpy as np
import torch
from PIL import Image
from torch import nn
from torchvision import models, transforms

# =============================================================================
# Constants
# =============================================================================
CLASS_NAMES = ["CNV", "DME", "DRUSEN", "NORMAL"]
MODEL_PATH = "VGG16_OCT_Retina_trained_model.pt"

CLASS_INFO = {
    "CNV": {
        "name": "Choroidal Neovascularization",
        "description": "Abnormal blood vessel growth in the choroid layer.",
        "severity": "High",
        "action": "Requires immediate attention. Treatment: anti-VEGF injections.",
    },
    "DME": {
        "name": "Diabetic Macular Edema",
        "description": "Fluid accumulation in the macula due to diabetes.",
        "severity": "High",
        "action": "Requires treatment. Treatment: anti-VEGF, laser therapy.",
    },
    "DRUSEN": {
        "name": "Drusen (Early AMD)",
        "description": "Yellowish deposits under the retina, early sign of AMD.",
        "severity": "Moderate",
        "action": "Regular monitoring required. Lifestyle modifications recommended.",
    },
    "NORMAL": {
        "name": "Normal Retina",
        "description": "Well-organized retinal layers with no abnormalities.",
        "severity": "None",
        "action": "No issues detected. Continue regular examinations.",
    },
}


# =============================================================================
# Model Functions
# =============================================================================
def get_device() -> torch.device:
    """Get the best available device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_model() -> nn.Module:
    """Load the trained VGG-16 model."""
    device = get_device()
    model = models.vgg16_bn()
    model.classifier[-1] = nn.Linear(4096, 4)

    if Path(MODEL_PATH).exists():
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    else:
        print(f"Warning: Model file not found at {MODEL_PATH}")

    model.to(device)
    model.eval()
    return model


def get_transform() -> transforms.Compose:
    """Get image transformation pipeline."""
    return transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
        ]
    )


# Load model at startup
print("Loading model...")
MODEL = load_model()
TRANSFORM = get_transform()
print(f"Model loaded on {get_device()}")


# =============================================================================
# Prediction Function
# =============================================================================
def predict(image: Image.Image) -> Tuple[Dict[str, float], str]:
    """
    Predict the class of an OCT image.

    Args:
        image: PIL Image to classify

    Returns:
        Tuple of (class_probabilities, detailed_result_text)
    """
    if image is None:
        return {}, "Please upload an image."

    # Preprocess
    img = image.convert("RGB")
    input_tensor = TRANSFORM(img).unsqueeze(0).to(get_device())

    # Predict
    with torch.no_grad():
        outputs = MODEL(input_tensor)
        probabilities = torch.softmax(outputs, dim=1)[0]

    # Create results
    probs_dict = {
        CLASS_NAMES[i]: float(probabilities[i]) for i in range(len(CLASS_NAMES))
    }

    # Find predicted class
    predicted_idx = torch.argmax(probabilities).item()
    predicted_class = CLASS_NAMES[predicted_idx]
    confidence = float(probabilities[predicted_idx]) * 100

    # Get class info
    info = CLASS_INFO[predicted_class]

    # Format detailed result
    result_text = f"""
## Prediction: {info["name"]}

**Confidence:** {confidence:.1f}%
**Severity Level:** {info["severity"]}

### Description
{info["description"]}

### Recommended Action
{info["action"]}

---
*Note: This is a clinical decision support tool. Results should be verified by qualified ophthalmologists.*
"""

    return probs_dict, result_text


# =============================================================================
# Gradio Interface
# =============================================================================
TITLE = "Retinal OCT Classification"
DESCRIPTION = """
Upload an OCT (Optical Coherence Tomography) image to detect retinal conditions.

This AI model classifies OCT images into four categories:
- **CNV**: Choroidal Neovascularization
- **DME**: Diabetic Macular Edema
- **DRUSEN**: Early Age-related Macular Degeneration
- **NORMAL**: Healthy Retina

**Model:** VGG-16 with Batch Normalization | **Accuracy:** 98.66%
"""

ARTICLE = """
### About This Project

This application was developed as part of **WQF7002 AI Techniques** (2025/2026) at University of Malaya,
supporting **SDG 3: Good Health and Well-being**.

The model was trained on 84,495 OCT images from the Kaggle Retinal OCT dataset.

**Disclaimer:** This tool is for educational and research purposes. It is not intended for clinical diagnosis.
"""

# Example images (if available)
examples = []
for class_name in CLASS_NAMES:
    example_path = f"data/{class_name}.jpeg"
    if Path(example_path).exists():
        examples.append(example_path)

# Build interface
demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil", label="Upload OCT Image"),
    outputs=[
        gr.Label(num_top_classes=4, label="Classification Results"),
        gr.Markdown(label="Detailed Analysis"),
    ],
    title=TITLE,
    description=DESCRIPTION,
    article=ARTICLE,
    examples=examples if examples else None,
    cache_examples=False,
    theme=gr.themes.Soft(),
)

if __name__ == "__main__":
    demo.launch()
