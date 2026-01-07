# AGENTS.md - Retinal OCT Damage Detection

> Guidelines for AI agents working in this repository.

---

## WQF7002 AI Techniques - Group Assignment 2025/2026

### Project Goal
Build an end-to-end AI application that aligns with a real-world problem and contributes to one or more **Sustainable Development Goals (SDGs)**.

### Project Requirements

| # | Requirement | Description |
|---|-------------|-------------|
| 1 | **Problem Identification** | Select a practical AI application tied to SDG(s), explain societal impact |
| 2 | **Data Preparation** | Public dataset (Kaggle/HuggingFace), EDA, cleaning, train/test split |
| 3 | **Model Training** | Use pre-trained model (BERT, ResNet, Whisper, etc.), justify choice, fine-tune |
| 4 | **Evaluation** | Test with metrics (accuracy, F1, BLEU), document refinements |
| 5 | **User Interface** | Interactive app using Gradio or Streamlit |
| 6 | **Discussion** | Performance analysis, ethics (bias/privacy), future improvements |
| Bonus | **Deployment** | Deploy on Hugging Face Spaces |

### Assessment Rubric (40% Total)

| Criteria | Weight |
|----------|--------|
| Problem Relevance & SDG Alignment | 6% |
| Data Preparation & EDA | 6% |
| Model Selection & Training | 8% |
| Evaluation & Results | 6% |
| User Interface / Demo | 6% |
| Ethical & Future Discussion | 4% |
| Presentation Delivery | 4% |

### AI Usage Policy
AI tools allowed for research and brainstorming. Must be acknowledged. No direct copy/paste.

---

## Project Context

**Goal:** Detect retinal damages from OCT (Optical Coherence Tomography) images to assist healthcare workers as a triage tool, supporting SDG 3 (Good Health and Well-being).

**Problem:** Many institutions still conduct OCT assessment manually, and there's a global shortage of ophthalmologists, especially in rural areas. Early detection can prevent blindness.

### Classification Classes (4 categories)
| Class | Description |
|-------|-------------|
| **CNV** | Choroidal Neovascularization - abnormal blood vessels in choroid layer |
| **DME** | Diabetic Macular Edema - fluid buildup from diabetes complications |
| **DRUSEN** | Yellowish lipid/protein deposits under retina (early AMD sign) |
| **NORMAL** | Healthy retina |

### Model Performance (Reference)
- **Accuracy:** 98.66% (after training)
- **Architecture:** VGG-16 with batch normalization (transfer learning)
- **Input Size:** 224x224 pixels (ImageNet standard)

---

## Build & Run Commands

### Environment Setup
```bash
python -m venv venv
source venv/bin/activate  # macOS/Linux
pip install -r requirements.txt
```

### Linting (Recommended)
```bash
pip install ruff mypy
ruff check .
ruff format .
mypy *.py --ignore-missing-imports
```

### Testing
```bash
pip install pytest pytest-cov
pytest                                    # all tests
pytest tests/test_model.py::test_predict  # single test
pytest --cov=. --cov-report=term-missing  # with coverage
```

---

## Project Structure

```
retinal-oct-model/
├── main.py                              # Reference code (Streamlit example)
├── grad_cam.py                          # Grad-CAM++ XAI visualization module
├── VGG16_OCT_Retina_trained_model.pt   # Trained model weights
└── AGENTS.md                            # This file
```

**Note:** `main.py` is reference code only. Main application TBD.

### Running Grad-CAM++ Visualization
```bash
python grad_cam.py --image path/to/oct_image.jpg --output result.jpg
python grad_cam.py --image path/to/oct_image.jpg --method gradcam++  # default
python grad_cam.py --image path/to/oct_image.jpg --method layercam   # alternative
```

---

## Code Style

### Import Order (PEP 8)
```python
import time                              # 1. Standard library
import torch                             # 2. Third-party
from torchvision import models, transforms
from PIL import Image
from utils import helper                 # 3. Local modules
```

### Naming Conventions
| Element | Style | Example |
|---------|-------|---------|
| Functions | snake_case | `load_model()`, `predict()` |
| Classes | PascalCase | `InformationBox` |
| Constants | UPPER_SNAKE | `MODEL_PATH` |
| Variables | snake_case | `uploaded_file` |

### Type Hints
```python
def load_image(file_path: str) -> Image.Image:
    return Image.open(file_path).convert("RGB")

def predict(img: Image.Image) -> int:
    return int(predicted_class)
```

### Formatting
- 4 spaces indentation
- 88-100 char line length
- Double quotes for strings
- 2 blank lines between top-level definitions

---

## PyTorch Patterns

### Device Handling
```python
def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

### Model Loading
```python
def load_model(model_path: str) -> nn.Module:
    model = models.vgg16_bn()
    model.classifier[-1] = nn.Linear(4096, 4)
    model.load_state_dict(torch.load(model_path, map_location=get_device()))
    model.to(get_device())
    model.eval()
    return model
```

### Inference
```python
with torch.no_grad():
    predictions = model(input_tensor)
```

---

## Agent Instructions

### Adding Features
1. Follow existing code patterns
2. Add type hints to all functions
3. Use proper device handling (CPU/GPU)

### Fixing Bugs
1. Minimal changes only
2. Don't refactor while fixing

### Testing Changes
```bash
python -m pytest tests/ -v
```

---

## Future Improvements (Planned)

### Model Enhancements
- VGG-19, ResNet, Inception, EfficientNet
- Transformer-based models
- Class weights for imbalance handling

### XAI Dashboard
- Grad-CAM for feature visualization
- Counterfactual explanations
- Builds trust with clinicians and patients

---

## Dataset Reference

- **Source:** Kaggle Retinal OCT Images
- **Total:** 84,495 images
- **Split:** Training, validation, testing sets
- **Note:** Class imbalance exists but not yet addressed

---

## Dependencies

```
torch>=2.0.0
torchvision>=0.15.0
pillow>=9.0.0
grad-cam>=1.5.0
opencv-python>=4.8.0
```
