"""
Test Accuracy Script for Retinal OCT Classification

Tests the model on 100 images from each class in data/test/ and reports accuracy.

Usage:
    python test_accuracy.py
    python test_accuracy.py --samples 50  # test 50 per class instead of 100
    python test_accuracy.py --data-dir path/to/test  # custom test directory
"""

import argparse
import random
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

from grad_cam import (
    CLASS_NAMES,
    DEFAULT_MODEL_PATH,
    get_device,
    get_transform,
    load_image,
    load_model,
    predict,
    preprocess_image,
)


def get_test_images(
    data_dir: str, samples_per_class: int = 100, seed: int = 42
) -> Dict[str, List[Path]]:
    """
    Get test images from each class folder.

    Args:
        data_dir: Path to test data directory containing class subfolders
        samples_per_class: Number of images to sample from each class
        seed: Random seed for reproducibility

    Returns:
        Dictionary mapping class name to list of image paths
    """
    random.seed(seed)
    data_path = Path(data_dir)

    if not data_path.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    test_images = {}

    for class_name in CLASS_NAMES:
        class_dir = data_path / class_name
        if not class_dir.exists():
            print(f"Warning: Class directory not found: {class_dir}")
            test_images[class_name] = []
            continue

        # Get all image files
        image_files = (
            list(class_dir.glob("*.jpeg"))
            + list(class_dir.glob("*.jpg"))
            + list(class_dir.glob("*.png"))
        )

        # Sample images
        if len(image_files) <= samples_per_class:
            sampled = image_files
        else:
            sampled = random.sample(image_files, samples_per_class)

        test_images[class_name] = sorted(sampled)
        print(
            f"  {class_name}: {len(sampled)} images (from {len(image_files)} available)"
        )

    return test_images


def run_test(
    model, test_images: Dict[str, List[Path]], verbose: bool = False
) -> Tuple[Dict[str, Dict], Dict[str, List]]:
    """
    Run predictions on test images and calculate accuracy.

    Args:
        model: Loaded PyTorch model
        test_images: Dictionary of class name to image paths
        verbose: Print each prediction

    Returns:
        Tuple of (results_dict, confusion_data)
    """
    results = {}
    confusion = defaultdict(list)  # actual_class -> list of predicted classes

    for actual_class, image_paths in test_images.items():
        if not image_paths:
            continue

        actual_idx = CLASS_NAMES.index(actual_class)
        correct = 0
        predictions = []

        for img_path in image_paths:
            try:
                # Load and preprocess image
                img = load_image(str(img_path))
                input_tensor, _ = preprocess_image(img)

                # Predict
                predicted_idx, probabilities = predict(model, input_tensor)
                predicted_class = CLASS_NAMES[predicted_idx]

                predictions.append(predicted_idx)
                confusion[actual_class].append(predicted_class)

                if predicted_idx == actual_idx:
                    correct += 1

                if verbose:
                    status = "✓" if predicted_idx == actual_idx else "✗"
                    conf = probabilities[0][predicted_idx].item() * 100
                    print(
                        f"  {status} {img_path.name}: {predicted_class} ({conf:.1f}%)"
                    )

            except Exception as e:
                print(f"  Error processing {img_path.name}: {e}")
                continue

        total = len(image_paths)
        accuracy = (correct / total * 100) if total > 0 else 0

        results[actual_class] = {
            "correct": correct,
            "total": total,
            "accuracy": accuracy,
            "predictions": predictions,
        }

    return results, dict(confusion)


def print_results(results: Dict[str, Dict], confusion: Dict[str, List]):
    """Print formatted results and confusion matrix."""
    print("\n" + "=" * 60)
    print("ACCURACY RESULTS")
    print("=" * 60)

    total_correct = 0
    total_samples = 0

    # Per-class accuracy
    print(f"\n{'Class':<10} {'Correct':<10} {'Total':<10} {'Accuracy':<10}")
    print("-" * 40)

    for class_name in CLASS_NAMES:
        if class_name in results:
            r = results[class_name]
            total_correct += r["correct"]
            total_samples += r["total"]
            print(
                f"{class_name:<10} {r['correct']:<10} {r['total']:<10} {r['accuracy']:.2f}%"
            )

    # Overall accuracy
    overall_accuracy = (total_correct / total_samples * 100) if total_samples > 0 else 0
    print("-" * 40)
    print(
        f"{'OVERALL':<10} {total_correct:<10} {total_samples:<10} {overall_accuracy:.2f}%"
    )

    # Confusion matrix
    print("\n" + "=" * 60)
    print("CONFUSION MATRIX")
    print("=" * 60)
    header = "Actual \\ Pred"
    print(f"\n{header:<12}", end="")
    for name in CLASS_NAMES:
        print(f"{name:<10}", end="")
    print()
    print("-" * 52)

    for actual in CLASS_NAMES:
        print(f"{actual:<12}", end="")
        if actual in confusion:
            pred_counts = {name: 0 for name in CLASS_NAMES}
            for pred in confusion[actual]:
                pred_counts[pred] += 1
            for name in CLASS_NAMES:
                count = pred_counts[name]
                if actual == name and count > 0:
                    print(f"\033[92m{count:<10}\033[0m", end="")  # Green for correct
                elif count > 0:
                    print(f"\033[91m{count:<10}\033[0m", end="")  # Red for incorrect
                else:
                    print(f"{count:<10}", end="")
        else:
            for _ in CLASS_NAMES:
                print(f"{'N/A':<10}", end="")
        print()

    print("=" * 60)

    return overall_accuracy


def main():
    parser = argparse.ArgumentParser(
        description="Test model accuracy on retinal OCT images"
    )
    parser.add_argument(
        "--data-dir",
        "-d",
        type=str,
        default="data/test",
        help="Path to test data directory",
    )
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        default=DEFAULT_MODEL_PATH,
        help="Path to model weights",
    )
    parser.add_argument(
        "--samples",
        "-n",
        type=int,
        default=100,
        help="Number of samples per class (default: 100)",
    )
    parser.add_argument(
        "--seed", "-s", type=int, default=42, help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Print each prediction"
    )

    args = parser.parse_args()

    # Validate paths
    if not Path(args.model).exists():
        print(f"Error: Model not found: {args.model}")
        return 1

    if not Path(args.data_dir).exists():
        print(f"Error: Data directory not found: {args.data_dir}")
        return 1

    print("=" * 60)
    print("RETINAL OCT CLASSIFICATION - ACCURACY TEST")
    print("=" * 60)
    print(f"\nDevice: {get_device()}")
    print(f"Model: {args.model}")
    print(f"Test data: {args.data_dir}")
    print(f"Samples per class: {args.samples}")
    print(f"Random seed: {args.seed}")

    # Load model
    print("\nLoading model...")
    start_time = time.time()
    model = load_model(args.model)
    print(f"Model loaded in {time.time() - start_time:.2f}s")

    # Get test images
    print(f"\nSampling {args.samples} images per class...")
    test_images = get_test_images(args.data_dir, args.samples, args.seed)

    total_images = sum(len(imgs) for imgs in test_images.values())
    if total_images == 0:
        print("Error: No test images found!")
        return 1

    # Run test
    print(f"\nRunning predictions on {total_images} images...")
    start_time = time.time()
    results, confusion = run_test(model, test_images, args.verbose)
    elapsed = time.time() - start_time
    print(f"Completed in {elapsed:.2f}s ({total_images / elapsed:.1f} images/sec)")

    # Print results
    overall_accuracy = print_results(results, confusion)

    return 0 if overall_accuracy >= 90 else 1


if __name__ == "__main__":
    sys.exit(main())
