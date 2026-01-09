---
title: Retinal OCT Classification
emoji: 👁
colorFrom: green
colorTo: teal
sdk: docker
app_file: streamlit_app.py
pinned: false
license: mit
tags:
  - medical
  - computer-vision
  - pytorch
  - healthcare
  - sdg3
---

# Retinal OCT Classification for Damage Detection

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/get-started/locally/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Project Overview

This project implements an end-to-end deep learning application for detecting retinal damages from Optical Coherence Tomography (OCT) images. The system is designed as a clinical decision support tool to assist healthcare workers in the early identification of retinal pathologies, addressing the global shortage of ophthalmologists and supporting **Sustainable Development Goal 3: Good Health and Well-being**.

By automating the triage process, this application enables faster screening in resource-limited settings and rural areas, where manual OCT assessment may be delayed due to a lack of specialized personnel.

## Screenshots

[Placeholder: Insert Streamlit Dashboard Screenshot]
[Placeholder: Insert Grad-CAM++ Visualization Screenshot]

## Features

- **Professional Clinical Interface**: A Streamlit-based web application with a medical-grade dashboard.
- **High-Accuracy Classification**: Leveraging a fine-tuned VGG-16 architecture with 98.66% accuracy.
- **Explainable AI (XAI)**: Integrated Grad-CAM++ visualization to highlight pathological regions in OCT scans.
- **Patient Report Generation**: LLM-powered (Llama 3.3 70B) patient-friendly explanations with PDF export.
- **Multi-Class Support**: Detects four distinct categories (CNV, DME, DRUSEN, NORMAL).
- **Analysis History**: Track previous analyses with thumbnails and results.
- **Performance Analytics**: Real-time metrics including precision, recall, and confusion matrix visualization.

## Installation

Ensure you have Python 3.10 or higher installed. It is recommended to use a virtual environment.

1. Clone the repository:
   ```bash
   git clone https://github.com/[username]/retinal-oct-model.git
   cd retinal-oct-model
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Model Weights:
   The trained model weights (`VGG16_OCT_Retina_trained_model.pt`) are approximately 500MB and are not included in the repository. Please ensure the file is placed in the root directory before running the application.

## Usage

### Clinical Web Application
To launch the interactive Streamlit dashboard:
```bash
streamlit run streamlit_app.py
```

### XAI Visualization (Command Line)
To generate Grad-CAM++ heatmaps for a specific OCT image:
```bash
python grad_cam.py --image path/to/image.jpeg --output result.jpg --method gradcam++
```

### Accuracy Testing
To verify model performance on the test dataset:
```bash
python test_accuracy.py --data-dir data/test --samples 100
```

## Model Details

- **Architecture**: VGG-16 with Batch Normalization (Transfer Learning)
- **Training Epochs**: Fine-tuned on ImageNet pre-trained weights
- **Input Resolution**: 224 x 224 pixels
- **Performance**: 98.66% Accuracy
- **Device Support**: Automatic detection of CUDA (GPU) or CPU execution

## Dataset Information

The model was trained and validated using the **Kaggle Retinal OCT Images** dataset.
- **Total Images**: 84,495
- **Format**: JPEG, Grayscale (processed as RGB)
- **Content**: Horizontal b-scans through the fovea

### Classification Classes

| Class | Full Name | Clinical Description |
|-------|-----------|----------------------|
| **CNV** | Choroidal Neovascularization | Abnormal blood vessel growth in the choroid layer |
| **DME** | Diabetic Macular Edema | Fluid buildup in the macula due to diabetes |
| **DRUSEN** | Drusen | Yellowish lipid deposits under the retina (early AMD) |
| **NORMAL** | Normal Retina | Healthy retina with well-defined layers |

## Explainable AI (XAI)

To build trust with clinical practitioners, the system utilizes **Grad-CAM++ (Gradient-weighted Class Activation Mapping)**. This technique produces a heatmap that identifies exactly which regions of the OCT image influenced the model's prediction. 

The heatmaps highlight:
- Hyperreflective materials in CNV
- Cystic spaces in DME
- RPE elevations in DRUSEN

## Project Structure

```

retinal-oct-model/
├── streamlit_app.py         # Main clinical web application (Streamlit)
├── grad_cam.py              # XAI visualization module (Grad-CAM++)
├── requirements.txt         # Project dependencies
├── .env.example             # Environment variables template
├── data/                    # Sample test images
├── journal/                 # Reference papers for citations
├── AGENTS.md                # Guidelines for AI development
└── README.md                # Project documentation
```

## Team & Credits

This project was developed as part of the **WQF7002 AI Techniques** course (2025/2026).

- **Course**: WQF7002 AI Techniques
- **Institution**: University of Malaya
- **Semester**: 2025/2026 Group Assignment

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Deployment

### Hugging Face Spaces

This project is configured for deployment on Hugging Face Spaces using Docker:

1. Create a new Space on [huggingface.co/spaces](https://huggingface.co/spaces)
2. Select **Docker** as the SDK
3. Upload the following files:
   - `streamlit_app.py` (main application)
   - `grad_cam.py` (XAI module)
   - `requirements.txt`
   - `Dockerfile`
   - `README.md`
   - `VGG16_OCT_Retina_trained_model.pt` (model weights)
   - `data/` folder (sample images)
4. Set up environment secrets (Settings → Secrets):
   - Add `GROQ_API_KEY` for patient report generation
5. The Space will automatically build and deploy

#### Using Git LFS for Large Files

For model weights (>10MB), use Git LFS:
```bash
git lfs install
git lfs track "*.pt"
git add .gitattributes
git add VGG16_OCT_Retina_trained_model.pt
git commit -m "Add model weights with LFS"
git push
```

#### Environment Variables

For local development, copy `.env.example` to `.env` and add your API key:
```bash
cp .env.example .env
# Edit .env and add: GROQ_API_KEY=your_api_key_here
```

## References

- Choi, K. J., et al. (2021). Deep learning models for screening of high myopia using optical coherence tomography. *Scientific Reports*, 11, 21663. https://doi.org/10.1038/s41598-021-00622-x

- Valerio, A. G., et al. (2025). From segmentation to explanation: Generating textual reports from MRI with LLMs. *Computer Methods and Programs in Biomedicine*, 270, 108922. https://doi.org/10.1016/j.cmpb.2025.108922
