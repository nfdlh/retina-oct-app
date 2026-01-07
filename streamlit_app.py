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
from datetime import datetime
from pathlib import Path

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

:root {
    --primary: #0d9488;
    --primary-light: #14b8a6;
    --primary-dark: #0f766e;
    --accent: #6366f1;
    --accent-light: #818cf8;
    --bg-primary: #fafaf9;
    --bg-card: #ffffff;
    --bg-elevated: #f5f5f4;
    --text-primary: #1c1917;
    --text-secondary: #57534e;
    --text-muted: #a8a29e;
    --border: #e7e5e4;
    --border-light: #f5f5f4;
    --shadow-sm: 0 1px 2px rgba(0,0,0,0.04);
    --shadow-md: 0 4px 12px rgba(0,0,0,0.06);
    --shadow-lg: 0 8px 24px rgba(0,0,0,0.08);
    --cnv-bg: #fff1f2;
    --cnv-border: #fecdd3;
    --cnv-text: #be123c;
    --dme-bg: #fef3c7;
    --dme-border: #fde68a;
    --dme-text: #b45309;
    --drusen-bg: #e0f2fe;
    --drusen-border: #7dd3fc;
    --drusen-text: #0369a1;
    --normal-bg: #d1fae5;
    --normal-border: #6ee7b7;
    --normal-text: #047857;
}

html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
}

h1, h2, h3, h4, h5, h6 {
    font-family: 'Inter', sans-serif;
    font-weight: 600;
    color: var(--text-primary);
}

* { scroll-behavior: smooth; }

.block-container {
    padding-top: 0.5rem;
    padding-bottom: 1rem;
    max-width: 1200px;
}

.top-header {
    background: linear-gradient(135deg, #134e4a 0%, #0f766e 50%, #0d9488 100%);
    padding: 20px 32px;
    border-radius: 0 0 24px 24px;
    margin: -0.5rem -1rem 24px -1rem;
    display: flex;
    justify-content: space-between;
    align-items: center;
    box-shadow: 0 4px 24px rgba(13, 148, 136, 0.2);
}

.logo-section {
    display: flex;
    align-items: center;
    gap: 16px;
}

.logo-icon {
    width: 52px;
    height: 52px;
    background: rgba(255,255,255,0.15);
    border-radius: 14px;
    display: flex;
    align-items: center;
    justify-content: center;
    backdrop-filter: blur(10px);
    border: 1px solid rgba(255,255,255,0.2);
}

.logo-icon svg {
    width: 30px;
    height: 30px;
    fill: white;
}

.logo-text { color: white; }

.logo-title {
    font-family: 'Inter', sans-serif;
    font-size: 24px;
    font-weight: 700;
    margin: 0;
    letter-spacing: -0.5px;
    color: white;
}

.logo-subtitle {
    font-size: 13px;
    opacity: 0.85;
    margin: 4px 0 0 0;
    letter-spacing: 0.3px;
    color: #ccfbf1;
}

.header-right {
    display: flex;
    align-items: center;
    gap: 20px;
}

.user-section {
    display: flex;
    align-items: center;
    gap: 14px;
    color: white;
}

.user-avatar {
    width: 44px;
    height: 44px;
    background: linear-gradient(135deg, #2dd4bf 0%, #14b8a6 100%);
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    box-shadow: 0 2px 12px rgba(45, 212, 191, 0.4);
    border: 2px solid rgba(255,255,255,0.4);
}

.user-info { text-align: right; }

.user-greeting {
    font-size: 11px;
    opacity: 0.8;
    margin: 0;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

.user-name {
    font-family: 'Inter', sans-serif;
    font-size: 15px;
    font-weight: 600;
    margin: 2px 0 0 0;
}

.nav-container {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 6px;
    margin-bottom: 28px;
    box-shadow: var(--shadow-sm);
}

.info-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 24px;
    margin-bottom: 16px;
    box-shadow: var(--shadow-sm);
    transition: all 0.2s ease;
}

.info-card:hover {
    box-shadow: var(--shadow-md);
    border-color: var(--primary-light);
}

.info-card-header {
    font-family: 'Inter', sans-serif;
    font-size: 16px;
    font-weight: 600;
    color: var(--text-primary);
    margin: 0 0 16px 0;
    display: flex;
    align-items: center;
    gap: 10px;
}

.metric-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 24px 20px;
    text-align: center;
    box-shadow: var(--shadow-sm);
    transition: all 0.3s ease;
}

.metric-card:hover {
    transform: translateY(-4px);
    box-shadow: var(--shadow-lg);
    border-color: var(--primary);
}

.metric-value {
    font-family: 'Inter', sans-serif;
    font-size: 36px;
    font-weight: 700;
    color: var(--primary-dark);
    margin: 0;
}

.metric-label {
    font-size: 13px;
    color: var(--text-secondary);
    margin: 8px 0 0 0;
    font-weight: 500;
}

.section-header {
    font-family: 'Inter', sans-serif;
    font-size: 18px;
    font-weight: 600;
    color: var(--text-primary);
    margin: 28px 0 16px 0;
    padding-bottom: 12px;
    border-bottom: 2px solid var(--border);
    display: flex;
    align-items: center;
    gap: 8px;
}

.status-high {
    background: var(--cnv-bg);
    color: var(--cnv-text);
    padding: 6px 14px;
    border-radius: 20px;
    font-size: 12px;
    font-weight: 600;
    border: 1px solid var(--cnv-border);
    display: inline-block;
}

.status-moderate {
    background: var(--drusen-bg);
    color: var(--drusen-text);
    padding: 6px 14px;
    border-radius: 20px;
    font-size: 12px;
    font-weight: 600;
    border: 1px solid var(--drusen-border);
    display: inline-block;
}

.status-none {
    background: var(--normal-bg);
    color: var(--normal-text);
    padding: 6px 14px;
    border-radius: 20px;
    font-size: 12px;
    font-weight: 600;
    border: 1px solid var(--normal-border);
    display: inline-block;
}

.result-box {
    background: linear-gradient(135deg, #f0fdfa 0%, #ccfbf1 100%);
    border: 1px solid #5eead4;
    border-radius: 20px;
    padding: 32px;
    text-align: center;
    margin: 16px 0;
    box-shadow: 0 4px 20px rgba(13, 148, 136, 0.1);
}

.result-title {
    font-family: 'Inter', sans-serif;
    font-size: 24px;
    font-weight: 700;
    color: var(--primary-dark);
    margin: 0 0 8px 0;
}

.result-confidence {
    font-family: 'Inter', sans-serif;
    font-size: 48px;
    font-weight: 800;
    color: var(--primary);
    margin: 12px 0;
}

.result-box-cnv {
    background: linear-gradient(135deg, #fff1f2 0%, #ffe4e6 100%);
    border-color: #fda4af;
}
.result-box-cnv .result-title,
.result-box-cnv .result-confidence { color: var(--cnv-text); }

.result-box-dme {
    background: linear-gradient(135deg, #fffbeb 0%, #fef3c7 100%);
    border-color: #fcd34d;
}
.result-box-dme .result-title,
.result-box-dme .result-confidence { color: var(--dme-text); }

.result-box-drusen {
    background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%);
    border-color: #7dd3fc;
}
.result-box-drusen .result-title,
.result-box-drusen .result-confidence { color: var(--drusen-text); }

.result-box-normal {
    background: linear-gradient(135deg, #ecfdf5 0%, #d1fae5 100%);
    border-color: #6ee7b7;
}
.result-box-normal .result-title,
.result-box-normal .result-confidence { color: var(--normal-text); }

.condition-table {
    width: 100%;
    border-collapse: separate;
    border-spacing: 0;
    border-radius: 12px;
    overflow: hidden;
    box-shadow: var(--shadow-sm);
}

.condition-table th {
    background: var(--bg-elevated);
    padding: 14px 16px;
    text-align: left;
    font-weight: 600;
    color: var(--text-primary);
    border-bottom: 2px solid var(--border);
    font-size: 13px;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

.condition-table td {
    padding: 14px 16px;
    border-bottom: 1px solid var(--border-light);
    color: var(--text-secondary);
    font-size: 14px;
}

.condition-table tr:hover td {
    background: var(--bg-elevated);
}

.condition-table tr:last-child td {
    border-bottom: none;
}

.clinical-notice {
    background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%);
    border: 1px solid #fbbf24;
    border-left: 4px solid #f59e0b;
    border-radius: 12px;
    padding: 18px 22px;
    margin: 20px 0;
    font-size: 14px;
    color: #92400e;
    box-shadow: var(--shadow-sm);
}

.stButton > button {
    border-radius: 12px !important;
    font-weight: 600 !important;
    padding: 12px 24px !important;
    transition: all 0.2s ease !important;
    border: none !important;
}

.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: var(--shadow-md) !important;
}

.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, var(--primary-dark) 0%, var(--primary) 100%) !important;
    color: white !important;
}

.stButton > button[kind="secondary"] {
    background: var(--bg-elevated) !important;
    color: var(--text-primary) !important;
    border: 1px solid var(--border) !important;
}

#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #f5f5f4 0%, #fafaf9 100%);
    border-right: 1px solid var(--border);
}

[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
    color: var(--text-primary);
}

.stFileUploader {
    border-radius: 12px;
}

.streamlit-expanderHeader {
    font-weight: 600;
    border-radius: 12px;
    color: var(--text-primary);
}

div[data-testid="stExpander"] {
    border: 1px solid var(--border);
    border-radius: 12px;
    background: var(--bg-card);
}

.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
}

.stTabs [data-baseweb="tab"] {
    border-radius: 10px;
    padding: 10px 20px;
    font-weight: 500;
}

.stTabs [aria-selected="true"] {
    background: var(--primary) !important;
    color: white !important;
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


def add_to_history(result: dict, image_name: str):
    """Add analysis result to session history."""
    if "analysis_history" not in st.session_state:
        st.session_state.analysis_history = []

    history_entry = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "image_name": image_name,
        "predicted_class": result["predicted_class"],
        "confidence": result["confidence"],
    }
    st.session_state.analysis_history.insert(0, history_entry)
    # Keep only last 10 entries
    st.session_state.analysis_history = st.session_state.analysis_history[:10]


# =============================================================================
# Session State Initialization (before header)
# =============================================================================
if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = False
if "analysis_result" not in st.session_state:
    st.session_state.analysis_result = None
if "analysis_history" not in st.session_state:
    st.session_state.analysis_history = []


# =============================================================================
# Top Header with User Info
# =============================================================================
st.markdown(
    """
<div class="top-header">
    <div class="logo-section">
        <div class="logo-icon">
            <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24">
                <path d="M12 4.5C7 4.5 2.73 7.61 1 12c1.73 4.39 6 7.5 11 7.5s9.27-3.11 11-7.5c-1.73-4.39-6-7.5-11-7.5zM12 17c-2.76 0-5-2.24-5-5s2.24-5 5-5 5 2.24 5 5-2.24 5-5 5zm0-8c-1.66 0-3 1.34-3 3s1.34 3 3 3 3-1.34 3-3-1.34-3-3-3z"/>
            </svg>
        </div>
        <div class="logo-text">
            <p class="logo-title">Retinal OCT Analysis</p>
            <p class="logo-subtitle">Clinical Decision Support System</p>
        </div>
    </div>
    <div class="header-right">
        <div class="user-section">
            <div class="user-info">
                <p class="user-greeting">Welcome back,</p>
                <p class="user-name">Dr. Clinician</p>
            </div>
            <div class="user-avatar">
                <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="white">
                    <path d="M12 12c2.21 0 4-1.79 4-4s-1.79-4-4-4-4 1.79-4 4 1.79 4 4 4zm0 2c-2.67 0-8 1.34-8 4v2h16v-2c0-2.66-5.33-4-8-4z"/>
                </svg>
            </div>
        </div>
    </div>
</div>
""",
    unsafe_allow_html=True,
)


# =============================================================================
# Sidebar - Settings
# =============================================================================
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    dark_mode = st.toggle(
        "Dark Mode", value=st.session_state.dark_mode, key="dark_toggle"
    )
    if dark_mode != st.session_state.dark_mode:
        st.session_state.dark_mode = dark_mode
        st.rerun()
    st.markdown("---")
    st.markdown("**Retinal OCT Analysis**")
    st.markdown("Clinical Decision Support System")


# =============================================================================
# Top Navigation
# =============================================================================
st.markdown('<div class="nav-container">', unsafe_allow_html=True)

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

st.markdown("</div>", unsafe_allow_html=True)

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
                background: linear-gradient(135deg, #042f2e 0%, #0f766e 50%, #0d9488 100%) !important;
                box-shadow: 0 4px 24px rgba(13, 148, 136, 0.3) !important;
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
                border-color: #57534e !important;
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
    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown(
            """
        <div class="info-card">
            <h3 class="info-card-header">
                <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="#0d9488">
                    <path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm1 15h-2v-6h2v6zm0-8h-2V7h2v2z"/>
                </svg>
                Clinical Decision Support System
            </h3>
            <p style="color: #57534e; line-height: 1.7; margin: 0;">
                Analyze OCT images to assist in detecting retinal pathologies. 
                Upload a scan, receive instant classification with attention visualization 
                highlighting the regions of clinical interest.
            </p>
        </div>
        """,
            unsafe_allow_html=True,
        )

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

    with col2:
        st.markdown(
            '<h3 class="section-header">Quick Actions</h3>', unsafe_allow_html=True
        )

        if st.button("Start New Analysis", type="primary", use_container_width=True):
            st.session_state.nav_to_analyze = True
            st.rerun()

        st.markdown("<br>", unsafe_allow_html=True)

        if st.button("View Analysis History", use_container_width=True):
            st.session_state.nav_to_history = True
            st.rerun()

        st.markdown("<br>", unsafe_allow_html=True)

        st.markdown(
            """
        <div class="info-card" style="background: linear-gradient(135deg, #e0f2fe 0%, #cffafe 100%); border-color: #67e8f9;">
            <h3 class="info-card-header" style="color: #0e7490;">
                <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="#0e7490">
                    <path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm-2 15l-5-5 1.41-1.41L10 14.17l7.59-7.59L19 8l-9 9z"/>
                </svg>
                How It Works
            </h3>
            <ol style="color: #0e7490; font-size: 13px; margin: 0; padding-left: 20px;">
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
            st.markdown(
                f"""
                <div class="info-card" style="padding: 12px 16px; margin-bottom: 8px;">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <div>
                            <strong>{entry["image_name"]}</strong>
                            <span class="{priority_info["status_class"]}" style="margin-left: 8px;">
                                {entry["predicted_class"]}
                            </span>
                        </div>
                        <div style="text-align: right; color: #64748b; font-size: 12px;">
                            {entry["confidence"] * 100:.1f}% | {entry["timestamp"]}
                        </div>
                    </div>
                </div>
                """,
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
            <ol style="color: #57534e; padding-left: 20px; margin: 0;">
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
            go.Bar(name="Precision", x=classes, y=precision, marker_color="#0f766e")
        )
        fig.add_trace(
            go.Bar(name="Recall", x=classes, y=recall, marker_color="#2dd4bf")
        )
        fig.update_layout(
            barmode="group",
            yaxis_range=[0.9, 1.01],
            template="plotly_white",
            height=350,
            margin=dict(t=20, b=40),
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
            font=dict(family="Source Sans Pro, sans-serif"),
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown(
            '<h3 class="section-header">Confusion Matrix</h3>', unsafe_allow_html=True
        )

        confusion = [[98, 1, 1, 0], [2, 95, 2, 1], [0, 1, 97, 2], [0, 0, 0, 100]]

        fig = px.imshow(
            confusion,
            labels=dict(x="Predicted", y="Actual", color="Count"),
            x=CLASS_NAMES,
            y=CLASS_NAMES,
            color_continuous_scale=[[0, "#f0fdfa"], [0.5, "#2dd4bf"], [1, "#0f766e"]],
            text_auto=True,
        )
        fig.update_layout(
            template="plotly_white",
            height=350,
            margin=dict(t=20, b=40),
            font=dict(family="Source Sans Pro, sans-serif"),
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown(
        '<h3 class="section-header">Model Architecture</h3>',
        unsafe_allow_html=True,
    )

    st.markdown(
        """
    <div class="info-card">
        <table style="width: 100%; color: #57534e;">
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
    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown(
            """
        <div class="info-card">
            <h3 class="info-card-header">
                <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="#0d9488">
                    <path d="M19.35 10.04C18.67 6.59 15.64 4 12 4 9.11 4 6.6 5.64 5.35 8.04 2.34 8.36 0 10.91 0 14c0 3.31 2.69 6 6 6h13c2.76 0 5-2.24 5-5 0-2.64-2.05-4.78-4.65-4.96zM14 13v4h-4v-4H7l5-5 5 5h-3z"/>
                </svg>
                Upload OCT Image
            </h3>
        </div>
        """,
            unsafe_allow_html=True,
        )

        uploaded_file = st.file_uploader(
            "Select OCT image file", type=["jpg", "jpeg", "png"]
        )

        # Sample images section
        st.markdown("**Or try a sample image:**")
        sample_cols = st.columns(4)
        selected_sample = None

        for idx, (class_name, path) in enumerate(SAMPLE_IMAGES.items()):
            with sample_cols[idx]:
                if Path(path).exists():
                    if st.button(
                        class_name, key=f"sample_{class_name}", use_container_width=True
                    ):
                        selected_sample = path
                        st.session_state.selected_sample = path
                        st.session_state.sample_name = class_name

        with st.expander("Analysis Settings"):
            cam_method = st.selectbox(
                "XAI Visualization Method",
                ["gradcam++", "gradcam", "layercam", "eigencam"],
            )
            overlay_alpha = st.slider("Heatmap Opacity", 0.2, 0.8, 0.5, 0.1)

        # Determine which image to use
        image = None
        image_name = "uploaded_image"

        if uploaded_file:
            image = Image.open(uploaded_file).convert("RGB")
            image_name = uploaded_file.name
        elif "selected_sample" in st.session_state and st.session_state.selected_sample:
            sample_path = st.session_state.selected_sample
            if Path(sample_path).exists():
                image = Image.open(sample_path).convert("RGB")
                image_name = st.session_state.get("sample_name", "sample")

        if image:
            st.image(
                image, caption=f"OCT Image: {image_name}", use_container_width=True
            )

            if st.button("Analyze Image", type="primary", use_container_width=True):
                with st.spinner("Analyzing image..."):
                    model = get_model()
                    input_tensor, rgb_img = preprocess_image(image)
                    predicted_idx, probabilities = predict(model, input_tensor)
                    predicted_class = CLASS_NAMES[predicted_idx]
                    confidence = float(probabilities[0][predicted_idx])

                    grayscale_cam = generate_gradcam(
                        model, input_tensor, predicted_idx, method=cam_method
                    )
                    heatmap_overlay = apply_heatmap(
                        rgb_img, grayscale_cam, alpha=overlay_alpha
                    )

                    st.session_state.analysis_result = {
                        "predicted_class": predicted_class,
                        "confidence": confidence,
                        "probabilities": {
                            CLASS_NAMES[i]: float(probabilities[0][i]) for i in range(4)
                        },
                        "heatmap": heatmap_overlay,
                        "image_name": image_name,
                    }
                    add_to_history(st.session_state.analysis_result, image_name)
                    st.rerun()

        st.markdown(
            """
        <div class="clinical-notice">
            <strong>Notice:</strong> Results are for clinical decision support only. 
            Always verify findings with comprehensive clinical examination.
        </div>
        """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            """
        <div class="info-card">
            <h3 class="info-card-header">
                <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="#0d9488">
                    <path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z"/>
                </svg>
                Analysis Results
            </h3>
        </div>
        """,
            unsafe_allow_html=True,
        )

        if "analysis_result" in st.session_state and st.session_state.analysis_result:
            result = st.session_state.analysis_result
            info = CLASS_DETAILS[result["predicted_class"]]

            st.markdown(
                f"""
            <div class="result-box">
                <p class="result-title">{info["full_name"]}</p>
                <span class="{info["status_class"]}">{info["priority"]} Priority</span>
                <p class="result-confidence">{result["confidence"] * 100:.1f}%</p>
            </div>
            """,
                unsafe_allow_html=True,
            )

            st.markdown("**Attention Visualization**")
            st.image(
                result["heatmap"],
                caption="Red = High attention | Blue = Low attention",
                use_container_width=True,
            )

            st.markdown("**Class Probabilities**")
            prob = result["probabilities"]
            colors = [
                "#0f766e" if k == result["predicted_class"] else "#d6d3d1" for k in prob
            ]

            fig = go.Figure(
                go.Bar(
                    x=list(prob.values()),
                    y=list(prob.keys()),
                    orientation="h",
                    marker_color=colors,
                    text=[f"{v * 100:.1f}%" for v in prob.values()],
                    textposition="inside",
                    textfont=dict(color="white"),
                )
            )
            fig.update_layout(
                xaxis_range=[0, 1],
                template="plotly_white",
                height=160,
                margin=dict(l=0, r=0, t=0, b=0),
                font=dict(family="Source Sans Pro, sans-serif"),
            )
            st.plotly_chart(fig, use_container_width=True)

            with st.expander("Clinical Details", expanded=True):
                st.write(f"**Description:** {info['description']}")
                st.write(f"**Significance:** {info['clinical_significance']}")
                st.write(f"**OCT Features:** {info['oct_features']}")

            btn_col1, btn_col2 = st.columns(2)
            with btn_col1:
                heatmap_buffer = get_image_download_buffer(result["heatmap"])
                st.download_button(
                    "Download Heatmap",
                    data=heatmap_buffer,
                    file_name=f"gradcam_{result.get('image_name', 'result')}.png",
                    mime="image/png",
                    use_container_width=True,
                )
            with btn_col2:
                if st.button("Clear Results", use_container_width=True):
                    st.session_state.analysis_result = None
                    st.session_state.selected_sample = None
                    st.rerun()
        else:
            st.info("Upload an OCT image and click 'Analyze Image' to see results.")

            st.markdown("""
            **Expected Output:**
            - Disease classification (CNV, DME, DRUSEN, or NORMAL)
            - Confidence score percentage
            - Grad-CAM++ attention visualization
            - Detailed clinical information
            """)


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
                <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="#0d9488">
                    <path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm-1 17.93c-3.95-.49-7-3.85-7-7.93 0-.62.08-1.21.21-1.79L9 15v1c0 1.1.9 2 2 2v1.93zm6.9-2.54c-.26-.81-1-1.39-1.9-1.39h-1v-3c0-.55-.45-1-1-1H8v-2h2c.55 0 1-.45 1-1V7h2c1.1 0 2-.9 2-2v-.41c2.93 1.19 5 4.06 5 7.41 0 2.08-.8 3.97-2.1 5.39z"/>
                </svg>
                Project Overview
            </h3>
            <p style="color: #57534e; margin-bottom: 16px;">
                <strong>WQF7002 AI Techniques</strong><br>
                Group Assignment 2025/2026
            </p>
            <p style="color: #57534e;">
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
            <ul style="color: #57534e; margin: 0; padding-left: 20px;">
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
            <table style="width: 100%; color: #57534e;">
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
        <div class="info-card" style="background: linear-gradient(135deg, #e0f2fe 0%, #cffafe 100%); border-color: #67e8f9;">
            <h3 class="info-card-header" style="color: #0e7490;">
                SDG 3: Good Health and Well-being
            </h3>
            <p style="color: #0e7490; margin: 0;">
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
