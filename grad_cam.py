"""
Grad-CAM++ Implementation for Retinal OCT Classification

This module provides explainable AI visualization for the VGG-16 retinal OCT
classification model using Grad-CAM++ and related methods.

Requirements:
    pip install grad-cam opencv-python

Usage:
    python grad_cam.py --image path/to/oct_image.jpg --output output.jpg
"""

import argparse
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
import torch
from PIL import Image
from torch import nn
from torchvision import models, transforms


# Class labels for retinal OCT classification
CLASS_NAMES = ["CNV", "DME", "DRUSEN", "NORMAL"]

# Default model path
DEFAULT_MODEL_PATH = "VGG16_OCT_Retina_trained_model.pt"


def get_device() -> torch.device:
    """Get the best available device (CUDA, MPS, or CPU)."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_model(model_path: str = DEFAULT_MODEL_PATH) -> nn.Module:
    """
    Load the trained VGG-16 model for retinal OCT classification.

    Args:
        model_path: Path to the trained model weights (.pt file)

    Returns:
        Loaded and configured VGG-16 model in eval mode
    """
    device = get_device()
    model = models.vgg16_bn()
    model.classifier[-1] = nn.Linear(4096, 4)  # 4 classes: CNV, DME, DRUSEN, NORMAL
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model


def get_transform() -> transforms.Compose:
    """Get the image transformation pipeline for VGG-16 input.

    Note: No normalization is applied because the model was trained
    without ImageNet normalization.
    """
    return transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            # No normalization - model was trained without it
        ]
    )


def load_image(image_path: str) -> Image.Image:
    """
    Load an image from file path.

    Args:
        image_path: Path to the image file

    Returns:
        PIL Image in RGB format
    """
    return Image.open(image_path).convert("RGB")


def preprocess_image(img: Image.Image) -> Tuple[torch.Tensor, np.ndarray]:
    """
    Preprocess image for model input and visualization.

    Args:
        img: PIL Image to preprocess

    Returns:
        Tuple of (input_tensor, rgb_image_normalized)
        - input_tensor: Tensor ready for model input (1, 3, 224, 224)
        - rgb_image_normalized: Numpy array normalized to [0, 1] for visualization
    """
    transform = get_transform()
    input_tensor = transform(img).unsqueeze(0).to(get_device())

    # Prepare image for visualization (resize and normalize to 0-1)
    img_resized = img.resize((224, 224))
    rgb_img = np.array(img_resized) / 255.0

    return input_tensor, rgb_img.astype(np.float32)


def predict(model: nn.Module, input_tensor: torch.Tensor) -> Tuple[int, torch.Tensor]:
    """
    Get prediction from model.

    Args:
        model: The VGG-16 model
        input_tensor: Preprocessed input tensor

    Returns:
        Tuple of (predicted_class_index, output_probabilities)
    """
    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()

    return predicted_class, probabilities


def generate_gradcam(
    model: nn.Module,
    input_tensor: torch.Tensor,
    target_class: Optional[int] = None,
    method: str = "gradcam++",
) -> np.ndarray:
    """
    Generate Grad-CAM++ heatmap for the input image.

    Args:
        model: The VGG-16 model
        input_tensor: Preprocessed input tensor
        target_class: Target class index (None = use predicted class)
        method: CAM method to use ('gradcam', 'gradcam++', 'scorecam', 'layercam')

    Returns:
        Grayscale CAM heatmap as numpy array (224, 224)
    """
    try:
        from pytorch_grad_cam import (
            GradCAM,
            GradCAMPlusPlus,
            ScoreCAM,
            LayerCAM,
            EigenCAM,
        )
        from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
    except ImportError:
        raise ImportError(
            "pytorch-grad-cam is required. Install with: pip install pytorch-grad-cam"
        )

    # Target layer for VGG-16: last convolutional layer
    # VGG-16 structure: features -> avgpool -> classifier
    # features[-1] is the last ReLU, features[-3] is the last Conv2d
    target_layers = [model.features[-1]]

    # Select CAM method
    cam_methods = {
        "gradcam": GradCAM,
        "gradcam++": GradCAMPlusPlus,
        "scorecam": ScoreCAM,
        "layercam": LayerCAM,
        "eigencam": EigenCAM,
    }

    if method.lower() not in cam_methods:
        raise ValueError(
            f"Unknown method: {method}. Choose from {list(cam_methods.keys())}"
        )

    cam_class = cam_methods[method.lower()]

    # Set up targets
    targets = None
    if target_class is not None:
        targets = [ClassifierOutputTarget(target_class)]

    # Generate CAM
    with cam_class(model=model, target_layers=target_layers) as cam:
        grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
        grayscale_cam = grayscale_cam[0, :]  # Get first image in batch

    return grayscale_cam


def apply_heatmap(
    rgb_img: np.ndarray,
    grayscale_cam: np.ndarray,
    colormap: int = cv2.COLORMAP_JET,
    alpha: float = 0.5,
) -> np.ndarray:
    """
    Apply CAM heatmap overlay on the original image.

    Args:
        rgb_img: Original RGB image normalized to [0, 1]
        grayscale_cam: Grayscale CAM heatmap
        colormap: OpenCV colormap for heatmap
        alpha: Transparency of heatmap overlay (0-1)

    Returns:
        Visualization image with heatmap overlay (RGB, uint8)
    """
    # Convert grayscale CAM to heatmap
    heatmap = cv2.applyColorMap(np.uint8(255 * grayscale_cam), colormap)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    heatmap = heatmap.astype(np.float32) / 255.0

    # Overlay heatmap on original image
    visualization = (1 - alpha) * rgb_img + alpha * heatmap
    visualization = np.clip(visualization, 0, 1)

    return np.uint8(255 * visualization)


def create_comparison_image(
    original: np.ndarray,
    heatmap_overlay: np.ndarray,
    prediction: int,
    probabilities: torch.Tensor,
) -> np.ndarray:
    """
    Create a side-by-side comparison image with prediction info.

    Args:
        original: Original RGB image (224, 224, 3)
        heatmap_overlay: Image with heatmap overlay (224, 224, 3)
        prediction: Predicted class index
        probabilities: Class probabilities

    Returns:
        Comparison image with labels
    """
    # Convert original to uint8 if needed
    if original.max() <= 1.0:
        original = np.uint8(255 * original)

    # Create side-by-side image
    height, width = original.shape[:2]
    comparison = np.zeros((height + 60, width * 2 + 20, 3), dtype=np.uint8)
    comparison[:, :, :] = 255  # White background

    # Place images
    comparison[50 : 50 + height, 5 : 5 + width] = original
    comparison[50 : 50 + height, 15 + width : 15 + 2 * width] = heatmap_overlay

    # Add labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(comparison, "Original", (60, 35), font, 0.7, (0, 0, 0), 2)
    cv2.putText(comparison, "Grad-CAM++", (width + 50, 35), font, 0.7, (0, 0, 0), 2)

    # Add prediction info at bottom
    probs = probabilities[0].cpu().numpy()
    pred_text = (
        f"Prediction: {CLASS_NAMES[prediction]} ({probs[prediction] * 100:.1f}%)"
    )
    cv2.putText(comparison, pred_text, (10, height + 55), font, 0.5, (0, 0, 0), 1)

    return comparison


def explain_prediction(
    image_path: str,
    model_path: str = DEFAULT_MODEL_PATH,
    output_path: Optional[str] = None,
    method: str = "gradcam++",
    target_class: Optional[int] = None,
    show_comparison: bool = True,
) -> Tuple[np.ndarray, int, dict]:
    """
    Generate Grad-CAM++ explanation for a retinal OCT image.

    This is the main function to use for explainability.

    Args:
        image_path: Path to the input OCT image
        model_path: Path to the trained model weights
        output_path: Path to save the visualization (optional)
        method: CAM method ('gradcam', 'gradcam++', 'scorecam', 'layercam')
        target_class: Target class for CAM (None = use prediction)
        show_comparison: If True, create side-by-side comparison

    Returns:
        Tuple of (visualization_image, predicted_class, info_dict)
    """
    # Load model and image
    model = load_model(model_path)
    img = load_image(image_path)
    input_tensor, rgb_img = preprocess_image(img)

    # Get prediction
    predicted_class, probabilities = predict(model, input_tensor)

    # Generate CAM
    target = target_class if target_class is not None else predicted_class
    grayscale_cam = generate_gradcam(model, input_tensor, target, method)

    # Create visualization
    heatmap_overlay = apply_heatmap(rgb_img, grayscale_cam)

    if show_comparison:
        visualization = create_comparison_image(
            rgb_img, heatmap_overlay, predicted_class, probabilities
        )
    else:
        visualization = heatmap_overlay

    # Save if output path provided
    if output_path:
        cv2.imwrite(output_path, cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR))
        print(f"Saved visualization to: {output_path}")

    # Prepare info dict
    info = {
        "predicted_class": predicted_class,
        "predicted_label": CLASS_NAMES[predicted_class],
        "probabilities": {
            CLASS_NAMES[i]: float(probabilities[0][i]) for i in range(len(CLASS_NAMES))
        },
        "method": method,
        "target_class": CLASS_NAMES[target],
    }

    return visualization, predicted_class, info


def main():
    """Command-line interface for Grad-CAM++ visualization."""
    parser = argparse.ArgumentParser(
        description="Generate Grad-CAM++ explanations for retinal OCT classification"
    )
    parser.add_argument(
        "--image", "-i", type=str, required=True, help="Path to input OCT image"
    )
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        default=DEFAULT_MODEL_PATH,
        help="Path to trained model weights",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="gradcam_output.jpg",
        help="Path to save output visualization",
    )
    parser.add_argument(
        "--method",
        type=str,
        default="gradcam++",
        choices=["gradcam", "gradcam++", "scorecam", "layercam", "eigencam"],
        help="CAM method to use",
    )
    parser.add_argument(
        "--target-class",
        type=int,
        default=None,
        help="Target class index (0=CNV, 1=DME, 2=DRUSEN, 3=NORMAL)",
    )

    args = parser.parse_args()

    # Validate input
    if not Path(args.image).exists():
        print(f"Error: Image not found: {args.image}")
        return 1

    if not Path(args.model).exists():
        print(f"Error: Model not found: {args.model}")
        return 1

    # Generate explanation
    print(f"Loading model from: {args.model}")
    print(f"Processing image: {args.image}")
    print(f"Using method: {args.method}")

    visualization, prediction, info = explain_prediction(
        image_path=args.image,
        model_path=args.model,
        output_path=args.output,
        method=args.method,
        target_class=args.target_class,
    )

    # Print results
    print("\n" + "=" * 50)
    print("PREDICTION RESULTS")
    print("=" * 50)
    print(f"Predicted Class: {info['predicted_label']}")
    print(f"\nClass Probabilities:")
    for label, prob in info["probabilities"].items():
        bar = "█" * int(prob * 20)
        print(f"  {label:8s}: {prob * 100:5.1f}% {bar}")
    print("=" * 50)

    return 0


if __name__ == "__main__":
    exit(main())
