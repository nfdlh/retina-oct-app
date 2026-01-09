"""
Retinal OCT Analysis - Professional Medical Web Application

A clean, professional medical interface with top navigation
for retinal OCT image analysis with explainable AI.

Requirements:
    pip install streamlit streamlit-antd-components plotly

Usage:
    streamlit run streamlit_app.py
"""

import io
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

load_dotenv()

import numpy as np
import streamlit as st
import streamlit_antd_components as sac
import plotly.express as px
import plotly.graph_objects as go
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


# =============================================================================
# Page Configuration
# =============================================================================
st.set_page_config(
    page_title="Retinal OCT Analysis",
    page_icon="eye",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# =============================================================================
# Custom CSS - Polished Medical Theme
# =============================================================================
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

/* Hide Streamlit toolbar/header */
header[data-testid="stHeader"] { display: none !important; }
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }

:root {
    --primary: #22c55e;
    --primary-light: #4ade80;
    --primary-dark: #16a34a;
    --bg-primary: #f9fafb;
    --bg-card: #ffffff;
    --text-primary: #111827;
    --text-secondary: #6b7280;
    --text-muted: #9ca3af;
    --border: #e5e7eb;
}

html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    background-color: var(--bg-primary);
    color: var(--text-primary);
}

.stApp {
    background-color: var(--bg-primary);
}

h1, h2, h3, h4, h5, h6 {
    font-family: 'Inter', sans-serif;
    font-weight: 600;
    color: var(--text-primary);
}

* { scroll-behavior: smooth; }

.block-container {
    padding-top: 1rem;
    padding-bottom: 2rem;
    padding-left: 3rem;
    padding-right: 3rem;
    max-width: 1400px;
    margin: 0 auto;
}

/* Fix navigation tabs visibility */
.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
    justify-content: center;
    background: transparent !important;
}

.stTabs [data-baseweb="tab-panel"] {
    background: transparent !important;
}

.stTabs {
    background: transparent !important;
}

.stTabs [data-baseweb="tab"] {
    color: var(--text-primary) !important;
    font-weight: 500;
    padding: 10px 20px;
    background: transparent !important;
}

.stTabs [aria-selected="true"] {
    color: var(--primary) !important;
    font-weight: 600;
}

/* Antd tabs (sac.tabs) - fix white background */
[class*="ant-tabs"] {
    background: transparent !important;
}

[class*="ant-tabs-nav"] {
    background: transparent !important;
}

[class*="ant-tabs-nav"]::before {
    border: none !important;
}

[class*="ant-tabs-content"] {
    background: transparent !important;
}

[class*="ant-tabs-tab"] {
    background: transparent !important;
    color: var(--text-secondary) !important;
    font-size: 14px !important;
    font-weight: 500 !important;
    padding: 8px 16px !important;
    margin: 0 4px !important;
    border-radius: 8px !important;
    transition: all 0.2s ease !important;
}

[class*="ant-tabs-tab"]:hover {
    color: var(--primary) !important;
    background: rgba(68, 186, 130, 0.1) !important;
}

[class*="ant-tabs-tab-active"] {
    background: var(--primary) !important;
    color: white !important;
}

[class*="ant-tabs-tab-active"] [class*="ant-tabs-tab-btn"] {
    color: white !important;
}

[class*="ant-tabs-ink-bar"] {
    display: none !important;
}

/* File uploader - minimalist design */
[data-testid="stFileUploader"] {
    background: transparent !important;
}

[data-testid="stFileUploader"] > label {
    font-size: 13px !important;
    font-weight: 500 !important;
    color: #374151 !important;
    margin-bottom: 8px !important;
}

[data-testid="stFileUploader"] > div > section {
    background: #fafafa !important;
    border: 1px dashed #d1d5db !important;
    border-radius: 12px !important;
    padding: 32px 24px !important;
    cursor: pointer !important;
    transition: all 0.2s ease !important;
}

[data-testid="stFileUploader"] > div > section:hover {
    border-color: #22c55e !important;
    background: #f0fdf4 !important;
}

[data-testid="stFileUploaderDropzoneInstructions"] {
    text-align: center !important;
}

[data-testid="stFileUploaderDropzoneInstructions"] svg {
    width: 40px !important;
    height: 40px !important;
    margin-bottom: 12px !important;
    color: #9ca3af !important;
}

[data-testid="stFileUploaderDropzoneInstructions"] > div > span {
    font-size: 14px !important;
    font-weight: 500 !important;
    color: #374151 !important;
}

[data-testid="stFileUploaderDropzoneInstructions"] > div > small {
    font-size: 12px !important;
    color: #9ca3af !important;
}

[data-testid="stFileUploader"] button[data-testid="stBaseButton-secondary"] {
    background: #22c55e !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 8px 20px !important;
    font-weight: 500 !important;
    font-size: 13px !important;
    margin-top: 12px !important;
}

[data-testid="stFileUploader"] button[data-testid="stBaseButton-secondary"]:hover {
    background: #16a34a !important;
}

/* Navigation tabs - blend with background */
[class*="ant-tabs"] {
    background: transparent !important;
}

[class*="ant-tabs-nav"] {
    background: transparent !important;
}

[class*="ant-tabs-nav"]::before {
    border: none !important;
}

[class*="ant-tabs-tab"] {
    color: #6b7280 !important;
    font-size: 14px !important;
    font-weight: 500 !important;
    background: transparent !important;
    padding: 8px 16px !important;
    border-radius: 8px !important;
    transition: all 0.2s ease !important;
}

[class*="ant-tabs-tab"]:hover {
    color: #22c55e !important;
}

[class*="ant-tabs-tab-active"] {
    color: #22c55e !important;
    background: #f0fdf4 !important;
}

[class*="ant-tabs-tab-active"] [class*="ant-tabs-tab-btn"] {
    color: #22c55e !important;
}

[class*="ant-tabs-ink-bar"] {
    display: none !important;
}

/* Header */
.top-header {
    background: var(--bg-card);
    padding: 16px 24px;
    border-radius: 16px;
    margin: 0 0 16px 0;
    display: flex;
    justify-content: space-between;
    align-items: center;
    box-shadow: var(--shadow-sm);
    border: 1px solid var(--border);
}

.header-nav {
    display: flex;
    align-items: center;
    gap: 8px;
}

.logo-section {
    display: flex;
    align-items: center;
    gap: 16px;
}

.logo-icon {
    width: 48px;
    height: 48px;
    background: var(--primary);
    border-radius: 12px;
    display: flex;
    align-items: center;
    justify-content: center;
    box-shadow: 0 4px 12px rgba(68, 186, 130, 0.3);
}

.logo-icon svg {
    width: 28px;
    height: 28px;
    fill: white;
}

.logo-text { color: var(--text-primary); }

.logo-title {
    font-family: 'Inter', sans-serif;
    font-size: 22px;
    font-weight: 700;
    margin: 0;
    letter-spacing: -0.5px;
    color: var(--text-primary);
}

.logo-subtitle {
    font-size: 13px;
    color: var(--text-secondary);
    margin: 2px 0 0 0;
}

.header-right {
    display: flex;
    align-items: center;
    gap: 20px;
}

.user-section {
    display: flex;
    align-items: center;
    gap: 12px;
    color: var(--text-primary);
}

.user-avatar {
    width: 40px;
    height: 40px;
    background: #e5e5e5;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    color: var(--text-secondary);
    font-weight: 600;
}

/* Cards */
.info-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 24px;
    margin-bottom: 20px;
    box-shadow: var(--shadow-sm);
    transition: all 0.2s ease;
}

.info-card:hover {
    box-shadow: var(--shadow-md);
    border-color: var(--border);
}

.info-card-header {
    font-family: 'Inter', sans-serif;
    font-size: 18px;
    font-weight: 600;
    color: var(--text-primary);
    margin: 0 0 16px 0;
    display: flex;
    align-items: center;
    gap: 10px;
}

/* Metrics */
.metric-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 24px 16px;
    text-align: center;
    box-shadow: var(--shadow-sm);
    transition: all 0.2s ease;
}

.metric-card:hover {
    transform: translateY(-2px);
    box-shadow: var(--shadow-md);
    border-color: var(--primary);
}

.metric-value {
    font-family: 'Inter', sans-serif;
    font-size: 32px;
    font-weight: 700;
    color: var(--primary);
    margin: 0;
}

.metric-label {
    font-size: 13px;
    color: var(--text-secondary);
    margin: 8px 0 0 0;
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

.section-header {
    font-family: 'Inter', sans-serif;
    font-size: 20px;
    font-weight: 600;
    color: var(--text-primary);
    margin: 32px 0 16px 0;
    padding-bottom: 0;
    border-bottom: none;
    display: flex;
    align-items: center;
    gap: 8px;
}

/* Status Labels */
.status-high {
    background: var(--cnv-bg);
    color: var(--cnv-text);
    padding: 4px 12px;
    border-radius: 100px;
    font-size: 12px;
    font-weight: 600;
    border: 1px solid var(--cnv-border);
    display: inline-block;
}

.status-moderate {
    background: var(--drusen-bg);
    color: var(--drusen-text);
    padding: 4px 12px;
    border-radius: 100px;
    font-size: 12px;
    font-weight: 600;
    border: 1px solid var(--drusen-border);
    display: inline-block;
}

.status-none {
    background: var(--normal-bg);
    color: var(--normal-text);
    padding: 4px 12px;
    border-radius: 100px;
    font-size: 12px;
    font-weight: 600;
    border: 1px solid var(--normal-border);
    display: inline-block;
}

/* Results */
.result-box {
    background: #ffffff;
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 32px;
    text-align: center;
    margin: 16px 0;
    box-shadow: var(--shadow-md);
}

.result-title {
    font-family: 'Inter', sans-serif;
    font-size: 24px;
    font-weight: 700;
    color: var(--text-primary);
    margin: 0 0 12px 0;
}

.result-confidence {
    font-family: 'Inter', sans-serif;
    font-size: 56px;
    font-weight: 800;
    color: var(--primary);
    margin: 16px 0;
    letter-spacing: -1px;
}

/* Condition Table */
.condition-table {
    width: 100%;
    border-collapse: separate;
    border-spacing: 0;
    border-radius: 12px;
    overflow: hidden;
    border: 1px solid var(--border);
    background: var(--bg-card);
}

.condition-table th {
    background: var(--bg-elevated);
    padding: 16px;
    text-align: left;
    font-weight: 600;
    color: var(--text-primary);
    border-bottom: 1px solid var(--border);
    font-size: 13px;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

.condition-table td {
    padding: 16px;
    border-bottom: 1px solid var(--border);
    color: var(--text-secondary);
    font-size: 14px;
}

.condition-table tr:last-child td {
    border-bottom: none;
}

.clinical-notice {
    background: #fffbeb;
    border: 1px solid #fde68a;
    border-left: 4px solid #f59e0b;
    border-radius: 8px;
    padding: 16px;
    margin: 24px 0;
    font-size: 14px;
    color: #92400e;
}

/* Buttons */
.stButton > button {
    border-radius: 8px !important;
    font-weight: 500 !important;
    padding: 10px 24px !important;
    transition: all 0.2s ease !important;
    border: none !important;
    height: auto !important;
}

.stButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: var(--shadow-md) !important;
}

.stButton > button[kind="primary"] {
    background: var(--primary) !important;
    color: white !important;
}

.stButton > button[kind="secondary"] {
    background: white !important;
    color: var(--text-primary) !important;
    border: 1px solid var(--border) !important;
}

/* Sidebar & Navigation */
[data-testid="stSidebar"] {
    background-color: #ffffff;
    border-right: 1px solid var(--border);
}

.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
    background-color: transparent;
    border-bottom: 1px solid var(--border);
    padding-bottom: 0;
}

.stTabs [data-baseweb="tab"] {
    border-radius: 8px 8px 0 0;
    padding: 12px 24px;
    font-weight: 500;
    color: var(--text-secondary);
    border: none;
    background: transparent;
}

.stTabs [aria-selected="true"] {
    background: transparent !important;
    color: var(--primary) !important;
    border-bottom: 2px solid var(--primary) !important;
    border-radius: 0 !important;
}

/* File Uploader */
.stFileUploader > div > div {
    background-color: white;
    border: 1px dashed var(--border);
    border-radius: 12px;
}

/* Expanders */
div[data-testid="stExpander"] {
    border: 1px solid var(--border);
    border-radius: 12px;
    background: var(--bg-card);
    box-shadow: none;
}

.streamlit-expanderHeader {
    font-weight: 600;
    color: var(--text-primary);
    background: transparent;
}

.logo-section {
    display: flex;
    align-items: center;
    gap: 16px;
}

.logo-icon {
    width: 48px;
    height: 48px;
    background: var(--primary);
    border-radius: 12px;
    display: flex;
    align-items: center;
    justify-content: center;
    box-shadow: 0 4px 12px rgba(68, 186, 130, 0.3);
}

.logo-icon svg {
    width: 28px;
    height: 28px;
    fill: white;
}

.logo-text { color: var(--text-primary); }

.logo-title {
    font-family: 'Inter', sans-serif;
    font-size: 22px;
    font-weight: 700;
    margin: 0;
    letter-spacing: -0.5px;
    color: var(--text-primary);
}

.logo-subtitle {
    font-size: 13px;
    color: var(--text-secondary);
    margin: 2px 0 0 0;
}

.header-right {
    display: flex;
    align-items: center;
    gap: 20px;
}

.user-section {
    display: flex;
    align-items: center;
    gap: 12px;
    color: var(--text-primary);
}

.user-avatar {
    width: 40px;
    height: 40px;
    background: #e5e5e5;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    color: var(--text-secondary);
    font-weight: 600;
}

/* Cards */
.info-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 24px;
    margin-bottom: 20px;
    box-shadow: var(--shadow-sm);
    transition: all 0.2s ease;
}

.info-card:hover {
    box-shadow: var(--shadow-md);
    border-color: var(--border);
}

.info-card-header {
    font-family: 'Inter', sans-serif;
    font-size: 18px;
    font-weight: 600;
    color: var(--text-primary);
    margin: 0 0 16px 0;
    display: flex;
    align-items: center;
    gap: 10px;
}

/* Metrics */
.metric-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 24px 16px;
    text-align: center;
    box-shadow: var(--shadow-sm);
    transition: all 0.2s ease;
}

.metric-card:hover {
    transform: translateY(-2px);
    box-shadow: var(--shadow-md);
    border-color: var(--primary);
}

.metric-value {
    font-family: 'Inter', sans-serif;
    font-size: 32px;
    font-weight: 700;
    color: var(--primary);
    margin: 0;
}

.metric-label {
    font-size: 13px;
    color: var(--text-secondary);
    margin: 8px 0 0 0;
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

.section-header {
    font-family: 'Inter', sans-serif;
    font-size: 20px;
    font-weight: 600;
    color: var(--text-primary);
    margin: 32px 0 16px 0;
    padding-bottom: 0;
    border-bottom: none;
    display: flex;
    align-items: center;
    gap: 8px;
}

/* Status Labels */
.status-high {
    background: var(--cnv-bg);
    color: var(--cnv-text);
    padding: 4px 12px;
    border-radius: 100px;
    font-size: 12px;
    font-weight: 600;
    border: 1px solid var(--cnv-border);
    display: inline-block;
}

.status-moderate {
    background: var(--drusen-bg);
    color: var(--drusen-text);
    padding: 4px 12px;
    border-radius: 100px;
    font-size: 12px;
    font-weight: 600;
    border: 1px solid var(--drusen-border);
    display: inline-block;
}

.status-none {
    background: var(--normal-bg);
    color: var(--normal-text);
    padding: 4px 12px;
    border-radius: 100px;
    font-size: 12px;
    font-weight: 600;
    border: 1px solid var(--normal-border);
    display: inline-block;
}

/* Results */
.result-box {
    background: #ffffff;
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 32px;
    text-align: center;
    margin: 16px 0;
    box-shadow: var(--shadow-md);
}

.result-title {
    font-family: 'Inter', sans-serif;
    font-size: 24px;
    font-weight: 700;
    color: var(--text-primary);
    margin: 0 0 12px 0;
}

.result-confidence {
    font-family: 'Inter', sans-serif;
    font-size: 56px;
    font-weight: 800;
    color: var(--primary);
    margin: 16px 0;
    letter-spacing: -1px;
}

/* Condition Table */
.condition-table {
    width: 100%;
    border-collapse: separate;
    border-spacing: 0;
    border-radius: 12px;
    overflow: hidden;
    border: 1px solid var(--border);
    background: var(--bg-card);
}

.condition-table th {
    background: var(--bg-elevated);
    padding: 16px;
    text-align: left;
    font-weight: 600;
    color: var(--text-primary);
    border-bottom: 1px solid var(--border);
    font-size: 13px;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

.condition-table td {
    padding: 16px;
    border-bottom: 1px solid var(--border);
    color: var(--text-secondary);
    font-size: 14px;
}

.condition-table tr:last-child td {
    border-bottom: none;
}

.clinical-notice {
    background: #fffbeb;
    border: 1px solid #fde68a;
    border-left: 4px solid #f59e0b;
    border-radius: 8px;
    padding: 16px;
    margin: 24px 0;
    font-size: 14px;
    color: #92400e;
}

/* Buttons */
.stButton > button {
    border-radius: 8px !important;
    font-weight: 500 !important;
    padding: 10px 24px !important;
    transition: all 0.2s ease !important;
    border: none !important;
    height: auto !important;
}

.stButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: var(--shadow-md) !important;
}

.stButton > button[kind="primary"] {
    background: var(--primary) !important;
    color: white !important;
}

.stButton > button[kind="secondary"] {
    background: white !important;
    color: var(--text-primary) !important;
    border: 1px solid var(--border) !important;
}

/* Sidebar & Navigation */
[data-testid="stSidebar"] {
    background-color: #ffffff;
    border-right: 1px solid var(--border);
}

.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
    background-color: transparent;
    border-bottom: 1px solid var(--border);
    padding-bottom: 0;
}

.stTabs [data-baseweb="tab"] {
    border-radius: 8px 8px 0 0;
    padding: 12px 24px;
    font-weight: 500;
    color: var(--text-secondary);
    border: none;
    background: transparent;
}

.stTabs [aria-selected="true"] {
    background: transparent !important;
    color: var(--primary) !important;
    border-bottom: 2px solid var(--primary) !important;
    border-radius: 0 !important;
}

/* File Uploader */
.stFileUploader > div > div {
    background-color: white;
    border: 1px dashed var(--border);
    border-radius: 12px;
}

/* Expanders */
div[data-testid="stExpander"] {
    border: 1px solid var(--border);
    border-radius: 12px;
    background: var(--bg-card);
    box-shadow: none;
}

.streamlit-expanderHeader {
    font-weight: 600;
    color: var(--text-primary);
    background: transparent;
}
</style>
""",
    unsafe_allow_html=True,
)


# =============================================================================
# Class Information
# =============================================================================
CLASS_DETAILS = {
    "CNV": {
        "full_name": "Choroidal Neovascularization",
        "description": "Growth of abnormal blood vessels from the choroid layer through Bruch's membrane.",
        "clinical_significance": "High severity. Requires immediate attention. Treatment: anti-VEGF injections.",
        "oct_features": "Hyperreflective material above RPE, subretinal fluid, RPE elevation.",
        "color": "red",
        "priority": "High",
        "status_class": "status-high",
    },
    "DME": {
        "full_name": "Diabetic Macular Edema",
        "description": "Fluid accumulation in the macula due to damaged blood vessels from diabetes.",
        "clinical_significance": "High severity. Requires treatment. Treatment: anti-VEGF, laser therapy.",
        "oct_features": "Intraretinal cystic spaces, increased retinal thickness, loss of foveal contour.",
        "color": "red",
        "priority": "High",
        "status_class": "status-high",
    },
    "DRUSEN": {
        "full_name": "Drusen",
        "description": "Yellowish deposits between RPE and Bruch's membrane. Early sign of AMD.",
        "clinical_significance": "Moderate severity. Regular monitoring required.",
        "oct_features": "Dome-shaped RPE elevations, variable reflectivity.",
        "color": "orange",
        "priority": "Moderate",
        "status_class": "status-moderate",
    },
    "NORMAL": {
        "full_name": "Normal Retina",
        "description": "Well-organized retinal layers with no abnormalities.",
        "clinical_significance": "No issues detected. Continue regular examinations.",
        "oct_features": "Well-defined layers, normal foveal contour, intact RPE.",
        "color": "green",
        "priority": "None",
        "status_class": "status-none",
    },
}


# =============================================================================
# Model Loading
# =============================================================================
@st.cache_resource
def get_model():
    return load_model(DEFAULT_MODEL_PATH)


# =============================================================================
# Sample Images
# =============================================================================
SAMPLE_IMAGES = {
    "CNV": "data/CNV.jpeg",
    "DME": "data/DME.jpeg",
    "DRUSEN": "data/DRUSEN.jpeg",
    "NORMAL": "data/NORMAL.jpeg",
}


# =============================================================================
# Helper Functions
# =============================================================================
def get_image_download_buffer(image_array: np.ndarray) -> io.BytesIO:
    """Convert numpy array to downloadable buffer."""
    img = Image.fromarray(image_array)
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    buffer.seek(0)
    return buffer


def create_thumbnail(image: Image.Image, size: tuple = (64, 64)) -> str:
    """Create a base64-encoded thumbnail from PIL Image."""
    import base64

    thumbnail = image.copy()
    thumbnail.thumbnail(size, Image.Resampling.LANCZOS)
    buffer = io.BytesIO()
    thumbnail.save(buffer, format="JPEG", quality=85)
    buffer.seek(0)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def add_to_history(result: dict, image_name: str, image: Optional[Image.Image] = None):
    """Add analysis result to session history."""
    if "analysis_history" not in st.session_state:
        st.session_state.analysis_history = []

    thumbnail_base64 = None
    if image is not None:
        thumbnail_base64 = create_thumbnail(image)

    history_entry = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "image_name": image_name,
        "predicted_class": result["predicted_class"],
        "confidence": result["confidence"],
        "thumbnail": thumbnail_base64,
    }
    st.session_state.analysis_history.insert(0, history_entry)
    st.session_state.analysis_history = st.session_state.analysis_history[:10]


def generate_patient_report(
    predicted_class: str,
    confidence: float,
    class_details: dict,
) -> str:
    """Generate a patient-friendly report using Groq LLM."""
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        return "⚠️ GROQ_API_KEY not found. Please set the environment variable to generate reports."

    try:
        from groq import Groq

        client = Groq(api_key=api_key)

        condition_info = class_details[predicted_class]
        prompt = f"""You are a compassionate medical assistant explaining OCT scan results to a patient. 
Write a clear, reassuring, and easy-to-understand explanation.

**Scan Results:**
- Detected Condition: {predicted_class} ({condition_info["full_name"]})
- Confidence Level: {confidence * 100:.1f}%
- Clinical Description: {condition_info["description"]}
- Medical Significance: {condition_info["clinical_significance"]}

**Instructions:**
1. Start with a brief, calming introduction
2. Explain what was found in simple terms (avoid medical jargon)
3. Describe what this means for the patient's eye health
4. Provide general lifestyle recommendations if applicable
5. Emphasize the importance of following up with their eye doctor
6. End with an encouraging note

Keep the response under 300 words. Use a warm, supportive tone.
Do NOT use markdown headers or bullet points - write in flowing paragraphs."""

        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=500,
        )

        return response.choices[0].message.content

    except ImportError:
        return "⚠️ Groq package not installed. Run: pip install groq"
    except Exception as e:
        return f"⚠️ Error generating report: {str(e)}"


def generate_pdf_report(
    result: dict,
    class_details: dict,
    patient_report: str,
    original_image: Optional[Image.Image] = None,
    heatmap_image: Optional[np.ndarray] = None,
) -> Optional[bytes]:
    """Generate a PDF report with images and LLM explanation."""
    try:
        from fpdf import FPDF
        import tempfile

        def sanitize_text(text: str) -> str:
            replacements = {
                "–": "-",
                "—": "-",
                "'": "'",
                "'": "'",
                """: '"',
                """: '"',
                "…": "...",
                "•": "-",
                "·": "-",
                "\u2018": "'",
                "\u2019": "'",
                "\u201c": '"',
                "\u201d": '"',
                "\u2013": "-",
                "\u2014": "-",
                "\u2026": "...",
            }
            for old, new in replacements.items():
                text = text.replace(old, new)
            return text.encode("latin-1", errors="replace").decode("latin-1")

        pdf = FPDF()
        pdf.set_auto_page_break(auto=True, margin=15)
        pdf.add_page()

        pdf.set_font("Helvetica", "B", 20)
        pdf.set_text_color(22, 101, 52)
        pdf.cell(0, 15, "Retinal OCT Analysis Report", ln=True, align="C")

        pdf.set_font("Helvetica", "", 10)
        pdf.set_text_color(100, 100, 100)
        pdf.cell(
            0,
            8,
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            ln=True,
            align="C",
        )
        pdf.ln(10)

        pdf.set_draw_color(200, 200, 200)
        pdf.line(10, pdf.get_y(), 200, pdf.get_y())
        pdf.ln(10)

        if original_image is not None or heatmap_image is not None:
            start_x = pdf.get_x()

            if original_image is not None:
                with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                    original_image.save(tmp.name, format="PNG")
                    pdf.image(tmp.name, x=15, w=85)
                    pdf.set_xy(15, pdf.get_y() + 3)
                    pdf.set_font("Helvetica", "I", 9)
                    pdf.set_text_color(100, 100, 100)
                    pdf.cell(85, 5, "Original OCT Scan", align="C")

            if heatmap_image is not None:
                with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                    heatmap_pil = Image.fromarray(heatmap_image)
                    heatmap_pil.save(tmp.name, format="PNG")
                    pdf.image(tmp.name, x=110, y=pdf.get_y() - 68, w=85)
                    pdf.set_xy(110, pdf.get_y())
                    pdf.set_font("Helvetica", "I", 9)
                    pdf.cell(85, 5, "Attention Heatmap (Grad-CAM++)", align="C")

            pdf.ln(15)

        predicted_class = result["predicted_class"]
        confidence = result["confidence"]
        info = class_details[predicted_class]

        pdf.set_font("Helvetica", "B", 14)
        pdf.set_text_color(30, 30, 30)
        pdf.cell(0, 10, "Classification Result", ln=True)

        pdf.set_font("Helvetica", "", 11)
        pdf.set_text_color(60, 60, 60)

        pdf.set_font("Helvetica", "B", 11)
        pdf.cell(50, 8, "Detected Condition:")
        pdf.set_font("Helvetica", "", 11)
        pdf.cell(0, 8, f"{predicted_class} ({info['full_name']})", ln=True)

        pdf.set_font("Helvetica", "B", 11)
        pdf.cell(50, 8, "Confidence:")
        pdf.set_font("Helvetica", "", 11)
        pdf.cell(0, 8, f"{confidence * 100:.1f}%", ln=True)

        pdf.set_font("Helvetica", "B", 11)
        pdf.cell(50, 8, "Priority Level:")
        pdf.set_font("Helvetica", "", 11)
        pdf.cell(0, 8, info["priority"], ln=True)

        pdf.ln(5)

        pdf.set_font("Helvetica", "B", 11)
        pdf.cell(0, 8, "Clinical Significance:", ln=True)
        pdf.set_font("Helvetica", "", 10)
        pdf.multi_cell(0, 6, sanitize_text(info["clinical_significance"]))

        pdf.ln(10)

        if patient_report and not patient_report.startswith("⚠️"):
            pdf.set_font("Helvetica", "B", 14)
            pdf.set_text_color(30, 30, 30)
            pdf.cell(0, 10, "Understanding Your Results", ln=True)

            pdf.set_font("Helvetica", "", 10)
            pdf.set_text_color(60, 60, 60)
            pdf.multi_cell(0, 6, sanitize_text(patient_report))
            pdf.ln(10)

        pdf.set_draw_color(200, 200, 200)
        pdf.line(10, pdf.get_y(), 200, pdf.get_y())
        pdf.ln(5)

        pdf.set_font("Helvetica", "I", 9)
        pdf.set_text_color(150, 100, 50)
        pdf.multi_cell(
            0,
            5,
            "Disclaimer: This report is generated by an AI system for clinical decision support only. Results should be verified by qualified ophthalmologists. This is not a medical diagnosis.",
        )

        return bytes(pdf.output())

    except ImportError as e:
        st.error(f"PDF library not installed. Run: pip install fpdf2. Error: {str(e)}")
        return None
    except Exception as e:
        st.error(f"Error generating PDF: {str(e)}")
        return None


# =============================================================================
# Session State Initialization (before header)
# =============================================================================
if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = False
if "analysis_result" not in st.session_state:
    st.session_state.analysis_result = None
if "analysis_history" not in st.session_state:
    st.session_state.analysis_history = []
if "patient_report" not in st.session_state:
    st.session_state.patient_report = None
if "generating_report" not in st.session_state:
    st.session_state.generating_report = False


# =============================================================================
# Top Header with Integrated Navigation
# =============================================================================
st.markdown(
    """
    <div style="
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 14px 20px;
        margin-bottom: 16px;
        background: linear-gradient(135deg, #166534 0%, #15803d 100%);
        border-radius: 12px;
        border: 1px solid #166534;
    ">
        <div style="display: flex; align-items: center; gap: 12px;">
            <div style="
                width: 38px;
                height: 38px;
                background: rgba(255, 255, 255, 0.2);
                border-radius: 10px;
                display: flex;
                align-items: center;
                justify-content: center;
                box-shadow: 0 2px 8px rgba(0, 0, 0, 0.15);
            ">
                <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="white">
                    <path d="M12 4.5C7 4.5 2.73 7.61 1 12c1.73 4.39 6 7.5 11 7.5s9.27-3.11 11-7.5c-1.73-4.39-6-7.5-11-7.5zM12 17c-2.76 0-5-2.24-5-5s2.24-5 5-5 5 2.24 5 5-2.24 5-5 5zm0-8c-1.66 0-3 1.34-3 3s1.34 3 3 3 3-1.34 3-3-1.34-3-3-3z"/>
                </svg>
            </div>
            <div>
                <p style="margin: 0; font-size: 15px; font-weight: 700; color: #ffffff;">Retinal OCT Analysis</p>
                <p style="margin: 0; font-size: 11px; color: #bbf7d0;">Clinical Decision Support</p>
            </div>
        </div>
        <div style="display: flex; align-items: center; gap: 10px;">
            <div style="text-align: right;">
                <p style="margin: 0; font-size: 10px; color: #bbf7d0; text-transform: uppercase; letter-spacing: 0.5px;">Welcome back,</p>
                <p style="margin: 0; font-size: 13px; font-weight: 600; color: #ffffff;">Dr. Clinician</p>
            </div>
            <div style="width: 34px; height: 34px; background: rgba(255, 255, 255, 0.2); border-radius: 50%; display: flex; align-items: center; justify-content: center;">
                <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="white">
                    <path d="M12 12c2.21 0 4-1.79 4-4s-1.79-4-4-4-4 1.79-4 4 1.79 4 4 4zm0 2c-2.67 0-8 1.34-8 4v2h16v-2c0-2.66-5.33-4-8-4z"/>
                </svg>
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

selected = sac.tabs(
    [
        sac.TabsItem("Home", icon="house-door"),
        sac.TabsItem("Analyze", icon="search"),
        sac.TabsItem("History", icon="clock-history"),
        sac.TabsItem("Model Info", icon="cpu"),
        sac.TabsItem("About", icon="info-circle"),
    ],
    index=0,
    format_func="title",
    align="center",
)

st.markdown("---")


# =============================================================================
# Theme Styles
# =============================================================================
if st.session_state.dark_mode:
    st.markdown(
        """
        <style>
            :root {
                --primary: #2dd4bf;
                --primary-light: #5eead4;
                --primary-dark: #14b8a6;
                --bg-primary: #0f172a;
                --bg-card: #1e293b;
                --bg-elevated: #334155;
                --text-primary: #f1f5f9;
                --text-secondary: #94a3b8;
                --text-muted: #64748b;
                --border: #334155;
                --border-light: #475569;
            }
            .stApp { background-color: #0f172a; color: #f1f5f9; }
            .stMarkdown, .stText, p, span, label { color: #f1f5f9 !important; }
            .info-card, .metric-card, .nav-container { 
                background: linear-gradient(180deg, #1e293b 0%, #0f172a 100%) !important; 
                border-color: #334155 !important; 
            }
            .info-card-header, .section-header, .result-title { 
                color: #f1f5f9 !important; 
            }
            .metric-value { color: #2dd4bf !important; }
            .metric-label { color: #94a3b8 !important; }
            .top-header { 
                background: linear-gradient(135deg, #042f2e 0%, #0f766e 50%, #44ba82 100%) !important;
                box-shadow: 0 4px 24px rgba(68, 186, 130, 0.3) !important;
            }
            .result-box {
                background: linear-gradient(135deg, #134e4a 0%, #0f766e 100%) !important;
                border-color: #2dd4bf !important;
            }
            .result-confidence { color: #2dd4bf !important; }
            .condition-table th { 
                background: #334155 !important; 
                color: #f1f5f9 !important;
            }
            .condition-table td { 
                color: #cbd5e1 !important;
                border-color: #737373 !important;
            }
            .condition-table tr:hover td { background: #334155 !important; }
            .clinical-notice {
                background: linear-gradient(135deg, #422006 0%, #78350f 100%) !important;
                border-color: #f59e0b !important;
                color: #fef3c7 !important;
            }
            [data-testid="stSidebar"] {
                background: linear-gradient(180deg, #1e293b 0%, #0f172a 100%) !important;
                border-color: #334155 !important;
            }
            [data-testid="stSidebar"] h1,
            [data-testid="stSidebar"] h2,
            [data-testid="stSidebar"] h3,
            [data-testid="stSidebar"] p,
            [data-testid="stSidebar"] label { color: #f1f5f9 !important; }
            div[data-testid="stExpander"] {
                background: #1e293b !important;
                border-color: #334155 !important;
            }
            .streamlit-expanderHeader { color: #f1f5f9 !important; }
            .stTabs [aria-selected="true"] {
                background: #14b8a6 !important;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


# =============================================================================
# Page: Home (Clinician-focused)
# =============================================================================
def render_home():
    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown(
            """
        <div class="info-card">
            <h3 class="info-card-header">
                <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="#44ba82">
                    <path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm1 15h-2v-6h2v6zm0-8h-2V7h2v2z"/>
                </svg>
                Clinical Decision Support System
            </h3>
            <p style="color: #737373; line-height: 1.7; margin: 0;">
                Analyze OCT images to assist in detecting retinal pathologies. 
                Upload a scan, receive instant classification with attention visualization 
                highlighting the regions of clinical interest.
            </p>
        </div>
        """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            '<h3 class="section-header">Quick Actions</h3>', unsafe_allow_html=True
        )

        if st.button("Start New Analysis", type="primary", use_container_width=True):
            st.session_state.nav_to_analyze = True
            st.rerun()

        if st.button("View Analysis History", use_container_width=True):
            st.session_state.nav_to_history = True
            st.rerun()

    col3, col4 = st.columns([1, 1])

    with col3:
        st.markdown(
            '<h3 class="section-header">Detectable Conditions</h3>',
            unsafe_allow_html=True,
        )

        st.markdown(
            """
        <table class="condition-table">
            <thead>
                <tr>
                    <th>Code</th>
                    <th>Condition</th>
                    <th>Clinical Action</th>
                    <th>Priority</th>
                </tr>
            </thead>
            <tbody>
                <tr>
                    <td><strong>CNV</strong></td>
                    <td>Choroidal Neovascularization</td>
                    <td>Refer for anti-VEGF treatment</td>
                    <td><span class="status-high">High</span></td>
                </tr>
                <tr>
                    <td><strong>DME</strong></td>
                    <td>Diabetic Macular Edema</td>
                    <td>Refer for anti-VEGF / laser therapy</td>
                    <td><span class="status-high">High</span></td>
                </tr>
                <tr>
                    <td><strong>DRUSEN</strong></td>
                    <td>Drusen (Early AMD)</td>
                    <td>Schedule follow-up monitoring</td>
                    <td><span class="status-moderate">Moderate</span></td>
                </tr>
                <tr>
                    <td><strong>NORMAL</strong></td>
                    <td>Healthy Retina</td>
                    <td>Routine follow-up</td>
                    <td><span class="status-none">None</span></td>
                </tr>
            </tbody>
        </table>
        """,
            unsafe_allow_html=True,
        )

        st.markdown(
            """
        <div class="clinical-notice">
            <strong>Clinical Disclaimer:</strong> This tool assists clinical 
            decision-making and does not replace professional medical judgment. All results 
            should be verified by qualified ophthalmologists.
        </div>
        """,
            unsafe_allow_html=True,
        )

    with col4:
        st.markdown(
            """
        <div class="info-card" style="background: linear-gradient(135deg, #eff6ff 0%, #dbeafe 100%); border-color: #bfdbfe;">
            <h3 class="info-card-header" style="color: #1e40af;">
                <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="#1e40af">
                    <path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm-2 15l-5-5 1.41-1.41L10 14.17l7.59-7.59L19 8l-9 9z"/>
                </svg>
                How It Works
            </h3>
            <ol style="color: #1e40af; font-size: 13px; margin: 0; padding-left: 20px;">
                <li>Upload an OCT scan image</li>
                <li>AI analyzes the retinal layers</li>
                <li>View classification and attention map</li>
                <li>Review clinical recommendations</li>
            </ol>
        </div>
        """,
            unsafe_allow_html=True,
        )


# =============================================================================
# Page: History
# =============================================================================
def render_history():
    st.markdown(
        '<h3 class="section-header">Analysis History</h3>',
        unsafe_allow_html=True,
    )

    if st.session_state.analysis_history:
        for entry in st.session_state.analysis_history:
            priority_info = CLASS_DETAILS[entry["predicted_class"]]

            col_thumb, col_info = st.columns([1, 6])

            with col_thumb:
                if entry.get("thumbnail"):
                    import base64

                    img_bytes = base64.b64decode(entry["thumbnail"])
                    st.image(img_bytes, width=56)
                else:
                    st.markdown(
                        '<div style="width: 56px; height: 56px; background: #e5e7eb; border-radius: 8px;"></div>',
                        unsafe_allow_html=True,
                    )

            with col_info:
                st.markdown(
                    f"""
                    <div style="padding: 4px 0;">
                        <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 4px;">
                            <div style="display: flex; align-items: center; gap: 8px;">
                                <strong>{entry["image_name"]}</strong>
                                <span class="{priority_info["status_class"]}">{entry["predicted_class"]}</span>
                            </div>
                            <div style="color: #64748b; font-size: 12px;">
                                {entry["confidence"] * 100:.1f}% | {entry["timestamp"]}
                            </div>
                        </div>
                        <div style="font-size: 12px; color: #6b7280; margin-top: 4px;">
                            {CLASS_DETAILS[entry["predicted_class"]]["description"][:80]}...
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            st.markdown(
                "<hr style='margin: 8px 0; border: none; border-top: 1px solid #e5e7eb;'>",
                unsafe_allow_html=True,
            )

        if st.button("Clear History", use_container_width=True):
            st.session_state.analysis_history = []
            st.rerun()
    else:
        st.info("No analysis history yet. Analyze some images to see them here.")

        st.markdown(
            """
        <div class="info-card">
            <h3 class="info-card-header">How to use</h3>
            <ol style="color: #737373; padding-left: 20px; margin: 0;">
                <li>Go to the <strong>Analyze</strong> tab</li>
                <li>Upload an OCT image or select a sample</li>
                <li>Click "Analyze Image"</li>
                <li>Your results will appear here</li>
            </ol>
        </div>
        """,
            unsafe_allow_html=True,
        )


# =============================================================================
# Page: Model Info
# =============================================================================
def render_model_info():
    st.markdown(
        '<h3 class="section-header">Model Performance</h3>', unsafe_allow_html=True
    )

    cols = st.columns(4)
    metrics = [
        ("98.66%", "Accuracy"),
        ("98.69%", "Precision"),
        ("98.66%", "Recall"),
        ("98.66%", "F1 Score"),
    ]

    for i, (value, label) in enumerate(metrics):
        with cols[i]:
            st.markdown(
                f"""
            <div class="metric-card">
                <p class="metric-value">{value}</p>
                <p class="metric-label">{label}</p>
            </div>
            """,
                unsafe_allow_html=True,
            )

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(
            '<h3 class="section-header">Per-Class Performance</h3>',
            unsafe_allow_html=True,
        )

        fig = go.Figure()
        classes = ["CNV", "DME", "DRUSEN", "NORMAL"]
        precision = [0.96, 0.99, 1.00, 1.00]
        recall = [0.99, 0.99, 0.97, 1.00]

        fig.add_trace(
            go.Bar(
                name="Precision",
                x=classes,
                y=precision,
                marker_color="#22c55e",
                text=[f"{v:.0%}" for v in precision],
                textposition="outside",
                textfont=dict(size=10, color="#374151"),
            )
        )
        fig.add_trace(
            go.Bar(
                name="Recall",
                x=classes,
                y=recall,
                marker_color="#3b82f6",
                text=[f"{v:.0%}" for v in recall],
                textposition="outside",
                textfont=dict(size=10, color="#374151"),
            )
        )
        fig.update_layout(
            barmode="group",
            yaxis_range=[0.9, 1.05],
            template="simple_white",
            paper_bgcolor="#ffffff",
            plot_bgcolor="#ffffff",
            height=320,
            margin=dict(t=40, b=50, l=50, r=30),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                x=0.5,
                xanchor="center",
                font=dict(color="#374151", size=11),
            ),
            font=dict(family="Inter, sans-serif", color="#374151", size=11),
            xaxis=dict(showgrid=False),
            yaxis=dict(showgrid=True, gridcolor="#f3f4f6", tickformat=".0%"),
            bargap=0.25,
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown(
            '<h3 class="section-header">Confusion Matrix</h3>', unsafe_allow_html=True
        )

        confusion = [[98, 1, 1, 0], [2, 95, 2, 1], [0, 1, 97, 2], [0, 0, 0, 100]]

        fig = go.Figure(
            data=go.Heatmap(
                z=confusion,
                x=CLASS_NAMES,
                y=CLASS_NAMES,
                colorscale=[[0, "#dcfce7"], [0.5, "#4ade80"], [1, "#16a34a"]],
                showscale=False,
                text=[[str(val) for val in row] for row in confusion],
                texttemplate="%{text}",
                textfont=dict(size=14, color="#1f2937"),
                hovertemplate="Actual: %{y}<br>Predicted: %{x}<br>Count: %{z}<extra></extra>",
            )
        )
        fig.update_layout(
            template="simple_white",
            paper_bgcolor="#ffffff",
            plot_bgcolor="#ffffff",
            height=320,
            margin=dict(t=30, b=60, l=60, r=30),
            font=dict(family="Inter, sans-serif", color="#374151", size=12),
            xaxis=dict(title="Predicted", tickfont=dict(size=11), side="bottom"),
            yaxis=dict(title="Actual", tickfont=dict(size=11), autorange="reversed"),
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown(
        '<h3 class="section-header">Model Architecture</h3>',
        unsafe_allow_html=True,
    )

    st.markdown(
        """
    <div class="info-card">
        <table style="width: 100%; color: #737373;">
            <tr><td style="padding: 8px 0;"><strong>Architecture</strong></td><td>VGG-16 with Batch Normalization</td></tr>
            <tr><td style="padding: 8px 0;"><strong>Training</strong></td><td>Transfer learning from ImageNet</td></tr>
            <tr><td style="padding: 8px 0;"><strong>Input Size</strong></td><td>224 x 224 pixels</td></tr>
            <tr><td style="padding: 8px 0;"><strong>Output Classes</strong></td><td>4 (CNV, DME, DRUSEN, NORMAL)</td></tr>
            <tr><td style="padding: 8px 0;"><strong>Dataset</strong></td><td>84,495 OCT images (Kaggle)</td></tr>
            <tr><td style="padding: 8px 0;"><strong>XAI Method</strong></td><td>Grad-CAM++ for attention visualization</td></tr>
        </table>
    </div>
    """,
        unsafe_allow_html=True,
    )

    st.markdown(
        '<h3 class="section-header">Condition Reference</h3>',
        unsafe_allow_html=True,
    )

    for name in CLASS_NAMES:
        info = CLASS_DETAILS[name]
        with st.expander(f"{name} - {info['full_name']}"):
            c1, c2 = st.columns([2, 1])
            with c1:
                st.write(f"**Description:** {info['description']}")
                st.write(f"**Clinical Significance:** {info['clinical_significance']}")
                st.write(f"**OCT Features:** {info['oct_features']}")
            with c2:
                st.markdown(f"**Priority Level:**")
                st.markdown(
                    f"<span class='{info['status_class']}'>{info['priority']}</span>",
                    unsafe_allow_html=True,
                )


# =============================================================================
# Page: Analyze
# =============================================================================
def render_analyze():
    if "last_uploaded_file" not in st.session_state:
        st.session_state.last_uploaded_file = None

    header_col1, header_col2 = st.columns([1, 1])
    with header_col1:
        st.markdown("#### Upload OCT Image")
    with header_col2:
        st.markdown("#### Analysis Result")

    col1, col2 = st.columns([1, 1])

    image = None
    image_name = "uploaded_image"

    with col1:
        uploaded_file = st.file_uploader(
            "Upload OCT Image",
            type=["jpg", "jpeg", "png"],
            label_visibility="collapsed",
        )
        enable_gradcam = st.toggle("Enable Grad-CAM++ Visualization", value=True)

        if uploaded_file:
            if st.session_state.last_uploaded_file != uploaded_file.name:
                st.session_state.analysis_result = None
                st.session_state.last_uploaded_file = uploaded_file.name
            image = Image.open(uploaded_file).convert("RGB")
            image_name = uploaded_file.name
        else:
            if st.session_state.last_uploaded_file is not None:
                st.session_state.analysis_result = None
                st.session_state.last_uploaded_file = None

    with col2:
        if "analysis_result" in st.session_state and st.session_state.analysis_result:
            result = st.session_state.analysis_result
            info = CLASS_DETAILS[result["predicted_class"]]

            priority_colors = {
                "High": {"color": "#dc2626", "bg": "#fee2e2"},
                "Moderate": {"color": "#d97706", "bg": "#fef3c7"},
                "None": {"color": "#16a34a", "bg": "#dcfce7"},
            }
            p_color = priority_colors.get(info["priority"], priority_colors["None"])

            st.markdown(
                f"""
            <div style="background: #ffffff; border: 1px solid #e5e7eb; border-radius: 12px; padding: 16px 20px; text-align: center;">
                <span style="background: {p_color["bg"]}; color: {p_color["color"]}; padding: 3px 10px; border-radius: 20px; font-size: 10px; font-weight: 600; text-transform: uppercase;">{info["priority"]} Risk</span>
                <p style="font-size: 28px; font-weight: 700; margin: 10px 0 6px 0; color: #111827;">{result["predicted_class"]}</p>
                <p style="font-size: 32px; font-weight: 700; margin: 0; color: {p_color["color"]};">{result["confidence"] * 100:.1f}%</p>
            </div>
            """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                """
            <div style="background: #ffffff; border: 1px dashed #d1d5db; border-radius: 12px; padding: 20px; text-align: center;">
                <p style="color: #9ca3af; margin: 0; font-size: 13px;">Results appear here</p>
            </div>
            """,
                unsafe_allow_html=True,
            )

    img_col1, img_col2 = st.columns([1, 1])

    with img_col1:
        if image:
            st.image(image, caption=f"Original: {image_name}", use_container_width=True)

            if st.button("Analyze Image", type="primary", use_container_width=True):
                with st.spinner("Analyzing..."):
                    model = get_model()
                    input_tensor, rgb_img = preprocess_image(image)
                    predicted_idx, probabilities = predict(model, input_tensor)
                    predicted_class = CLASS_NAMES[predicted_idx]
                    confidence = float(probabilities[0][predicted_idx])

                    heatmap_overlay = None
                    if enable_gradcam:
                        grayscale_cam = generate_gradcam(
                            model, input_tensor, predicted_idx, method="gradcam++"
                        )
                        heatmap_overlay = apply_heatmap(
                            rgb_img, grayscale_cam, alpha=0.5
                        )
                        heatmap_pil = Image.fromarray(heatmap_overlay)
                        heatmap_overlay = np.array(heatmap_pil.resize(image.size))

                    st.session_state.analysis_result = {
                        "predicted_class": predicted_class,
                        "confidence": confidence,
                        "probabilities": {
                            CLASS_NAMES[i]: float(probabilities[0][i]) for i in range(4)
                        },
                        "heatmap": heatmap_overlay,
                        "image_name": image_name,
                        "gradcam_enabled": enable_gradcam,
                        "original_image": image,
                    }
                    st.session_state.patient_report = None
                    add_to_history(st.session_state.analysis_result, image_name, image)
                    st.rerun()

    with img_col2:
        if "analysis_result" in st.session_state and st.session_state.analysis_result:
            result = st.session_state.analysis_result
            info = CLASS_DETAILS[result["predicted_class"]]

            if result.get("gradcam_enabled") and result.get("heatmap") is not None:
                st.image(
                    result["heatmap"],
                    caption="Attention Map",
                    use_container_width=True,
                )

            prob = result["probabilities"]
            for class_name, value in prob.items():
                is_predicted = class_name == result["predicted_class"]
                bar_color = "#22c55e" if is_predicted else "#e5e7eb"
                text_color = "#166534" if is_predicted else "#6b7280"
                pct = value * 100

                st.markdown(
                    f"""
                    <div style="margin-bottom: 6px;">
                        <div style="display: flex; justify-content: space-between; margin-bottom: 2px;">
                            <span style="font-size: 12px; font-weight: {"600" if is_predicted else "400"}; color: {text_color};">{class_name}</span>
                            <span style="font-size: 12px; font-weight: 600; color: {text_color};">{pct:.1f}%</span>
                        </div>
                        <div style="background: #f3f4f6; border-radius: 3px; height: 6px; overflow: hidden;">
                            <div style="background: {bar_color}; width: {pct}%; height: 100%; border-radius: 3px;"></div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            with st.expander("Clinical Details", expanded=False):
                st.write(f"**Description:** {info['description']}")
                st.write(f"**Significance:** {info['clinical_significance']}")

            st.markdown(
                """
                <style>
                    .action-buttons button {
                        height: 48px !important;
                        min-height: 48px !important;
                        padding-top: 0 !important;
                        padding-bottom: 0 !important;
                        line-height: 48px !important;
                    }
                    .action-buttons button p {
                        margin: 0 !important;
                        line-height: 48px !important;
                    }
                    .action-buttons [data-testid="stDownloadButton"] button {
                        height: 48px !important;
                        min-height: 48px !important;
                    }
                </style>
                """,
                unsafe_allow_html=True,
            )

            st.markdown('<div class="action-buttons">', unsafe_allow_html=True)
            btn_col1, btn_col2, btn_col3 = st.columns(3)
            with btn_col1:
                if result.get("heatmap") is not None:
                    heatmap_buffer = get_image_download_buffer(result["heatmap"])
                    st.download_button(
                        "⬇ Heatmap",
                        data=heatmap_buffer,
                        file_name=f"gradcam_{result.get('image_name', 'result')}.png",
                        mime="image/png",
                        use_container_width=True,
                    )
                else:
                    st.button("⬇ Heatmap", disabled=True, use_container_width=True)
            with btn_col2:
                if st.button("📄 Report", use_container_width=True):
                    st.session_state.generating_report = True
                    st.rerun()
            with btn_col3:
                if st.button("✕ Clear", use_container_width=True):
                    st.session_state.analysis_result = None
                    st.session_state.patient_report = None
                    st.rerun()
            st.markdown("</div>", unsafe_allow_html=True)

            if st.session_state.get("generating_report"):
                with st.spinner("Generating patient-friendly report..."):
                    report = generate_patient_report(
                        result["predicted_class"],
                        result["confidence"],
                        CLASS_DETAILS,
                    )
                    st.session_state.patient_report = report
                    st.session_state.generating_report = False
                    st.rerun()

            if st.session_state.get("patient_report"):
                st.markdown("---")
                st.markdown("#### 📄 Patient Report")
                st.markdown(
                    f"""
                    <div style="background: #f0fdf4; border: 1px solid #bbf7d0; border-radius: 12px; padding: 20px; margin: 8px 0;">
                        <p style="color: #166534; line-height: 1.7; margin: 0; font-size: 14px;">
                            {st.session_state.patient_report}
                        </p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                pdf_bytes = generate_pdf_report(
                    result=result,
                    class_details=CLASS_DETAILS,
                    patient_report=st.session_state.patient_report,
                    original_image=result.get("original_image"),
                    heatmap_image=result.get("heatmap"),
                )

                st.markdown('<div class="action-buttons">', unsafe_allow_html=True)
                dl_col1, dl_col2 = st.columns(2)
                with dl_col1:
                    st.download_button(
                        "⬇ Download TXT",
                        data=st.session_state.patient_report,
                        file_name=f"patient_report_{result.get('image_name', 'scan')}.txt",
                        mime="text/plain",
                        use_container_width=True,
                    )
                with dl_col2:
                    if pdf_bytes:
                        st.download_button(
                            "⬇ Download PDF",
                            data=pdf_bytes,
                            file_name=f"patient_report_{result.get('image_name', 'scan')}.pdf",
                            mime="application/pdf",
                            use_container_width=True,
                        )
                    else:
                        st.button(
                            "PDF Unavailable", disabled=True, use_container_width=True
                        )
                st.markdown("</div>", unsafe_allow_html=True)

    st.markdown(
        """
    <div class="clinical-notice">
        <strong>Notice:</strong> Results are for clinical decision support only. 
        Always verify findings with comprehensive clinical examination.
    </div>
    """,
        unsafe_allow_html=True,
    )


# =============================================================================
# Page: About
# =============================================================================
def render_about():
    col1, col2 = st.columns(2)

    with col1:
        st.markdown(
            """
        <div class="info-card">
            <h3 class="info-card-header">
                <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="#44ba82">
                    <path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm-1 17.93c-3.95-.49-7-3.85-7-7.93 0-.62.08-1.21.21-1.79L9 15v1c0 1.1.9 2 2 2v1.93zm6.9-2.54c-.26-.81-1-1.39-1.9-1.39h-1v-3c0-.55-.45-1-1-1H8v-2h2c.55 0 1-.45 1-1V7h2c1.1 0 2-.9 2-2v-.41c2.93 1.19 5 4.06 5 7.41 0 2.08-.8 3.97-2.1 5.39z"/>
                </svg>
                Project Overview
            </h3>
            <p style="color: #737373; margin-bottom: 16px;">
                <strong>WQF7002 AI Techniques</strong><br>
                Group Assignment 2025/2026
            </p>
            <p style="color: #737373;">
                This project aims to detect retinal damages from OCT images to assist 
                healthcare workers as a triage tool, supporting SDG 3 (Good Health and Well-being).
            </p>
        </div>
        """,
            unsafe_allow_html=True,
        )

        st.markdown(
            """
        <div class="info-card">
            <h3 class="info-card-header">Problem Statement</h3>
            <ul style="color: #737373; margin: 0; padding-left: 20px;">
                <li>Many institutions conduct OCT assessment manually</li>
                <li>Global shortage of ophthalmologists in rural areas</li>
                <li>Early detection can prevent blindness</li>
                <li>AI can serve as an effective triage tool</li>
            </ul>
        </div>
        """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            """
        <div class="info-card">
            <h3 class="info-card-header">Technical Details</h3>
            <table style="width: 100%; color: #737373;">
                <tr><td><strong>Model</strong></td><td>VGG-16 with Batch Normalization</td></tr>
                <tr><td><strong>Dataset</strong></td><td>84,495 OCT images (Kaggle)</td></tr>
                <tr><td><strong>Input Size</strong></td><td>224 x 224 pixels</td></tr>
                <tr><td><strong>XAI Method</strong></td><td>Grad-CAM++</td></tr>
                <tr><td><strong>Accuracy</strong></td><td>98.66%</td></tr>
            </table>
        </div>
        """,
            unsafe_allow_html=True,
        )

        st.markdown(
            """
        <div class="info-card" style="background: linear-gradient(135deg, #eff6ff 0%, #dbeafe 100%); border-color: #bfdbfe;">
            <h3 class="info-card-header" style="color: #1e40af;">
                SDG 3: Good Health and Well-being
            </h3>
            <p style="color: #1e40af; margin: 0;">
                This project contributes to early detection of retinal diseases, 
                addressing the global shortage of ophthalmologists and enabling 
                faster screening in resource-limited settings.
            </p>
        </div>
        """,
            unsafe_allow_html=True,
        )


# =============================================================================
# Main Routing
# =============================================================================
if "nav_to_analyze" in st.session_state and st.session_state.nav_to_analyze:
    st.session_state.nav_to_analyze = False
    selected = "Analyze"

if "nav_to_history" in st.session_state and st.session_state.nav_to_history:
    st.session_state.nav_to_history = False
    selected = "History"

if selected == "Home":
    render_home()
elif selected == "Analyze":
    render_analyze()
elif selected == "History":
    render_history()
elif selected == "Model Info":
    render_model_info()
elif selected == "About":
    render_about()
else:
    render_home()
