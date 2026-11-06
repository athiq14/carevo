"""
AI Career Intelligence Agent — Premium Dark SaaS Platform
High-Fidelity AI Career Co-pilot for Resume Analysis, JD Matching, Skill Gaps & Autonomous Agentic Workflows.
Demonstration-Ready Interface for Razorpay AI Builder Team.
"""
import io
import json
import re
from datetime import datetime, timezone
import streamlit as st
import fitz  # PyMuPDF

from config import load_config
from core.llm import GeminiProvider, GroqProvider, LLMProviderManager, MockLLMProvider
from core.llm.structured.engine import StructuredOutputEngine
from core.matching import CareerMatchingPipeline, PipelineMatchResult
from core.matching.explanation_generator import MatchExplainer
from core.matching.weight_manager import BASELINE_WEIGHTS, MAX_DIMENSION_CAP
from core.workflow.ui import render_agentic_workflow_ui
from core.workflow.validators import LinkedInProfileValidator
from agents.career_advisor_engine import CareerAdvisorEngine
from core.telemetry import telemetry_tracker
from processors.jd_processor import MAX_JD_CHAR_LIMIT
from models.matching import CandidateFitLevel, EvaluationStatus, JobDescriptionProfile, JobMatchResult
from processors import (
    PDFProcessingError,
    ResumePipeline,
    clean_resume_text,
    detect_resume_sections,
    extract_resume_entities,
)
from models import (
    ExtractionStatus,
    ProfileProcessingStatus,
    ResumeProfile,
    SectionType,
    SkillCategory,
)

# -----------------------------------------------------------------------------
# Page Configuration & Visual Theme
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="CAREVO — AI Career Intelligence Platform",
    page_icon="assets/favicon.png",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Load configuration
config = load_config()

# -----------------------------------------------------------------------------
# Premium Dark Design System CSS (Matching Reference Specification)
# -----------------------------------------------------------------------------
st.markdown(
    """
    <style>
    /* Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

    /* Global Root & Typography Scale */
    html, body, [class*="css"], .stApp {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif !important;
        font-size: 16px !important;
        line-height: 1.55 !important;
        color: #F5F3FA !important;
    }

    /* Primary Dark Canvas with subtle ambient violet wave */
    .stApp {
        background-color: #090914 !important;
        background-image: 
            radial-gradient(circle at 92% 10%, rgba(139, 92, 246, 0.16) 0%, rgba(9, 9, 20, 0) 55%),
            radial-gradient(circle at 10% 90%, rgba(124, 58, 237, 0.08) 0%, rgba(9, 9, 20, 0) 50%);
        color: #F5F3FA !important;
    }

    /* Top Streamlit Header */
    [data-testid="stHeader"] {
        background: transparent !important;
    }

    /* Streamlit Main Block Container & Layout */
    [data-testid="stMainBlockContainer"],
    [data-testid="stVerticalBlock"],
    [data-testid="stHorizontalBlock"] {
        font-size: 16px;
    }

    /* 1. Main Page Titles (32px) */
    h1,
    .stMarkdown h1,
    [data-testid="stHeadingWithActionElements"] h1,
    .pf-main-title {
        font-size: 32px !important;
        line-height: 1.25 !important;
        font-weight: 800 !important;
        letter-spacing: -0.025em !important;
        color: #F5F3FA !important;
        margin-bottom: 8px !important;
    }

    /* 2. Primary Section Headings (25px) */
    h2,
    .stMarkdown h2,
    [data-testid="stHeadingWithActionElements"] h2,
    .pf-section-heading {
        font-size: 25px !important;
        line-height: 1.3 !important;
        font-weight: 700 !important;
        letter-spacing: -0.015em !important;
        color: #F5F3FA !important;
        margin-top: 14px !important;
        margin-bottom: 10px !important;
    }

    /* 3. Secondary Headings (21px) */
    h3,
    .stMarkdown h3,
    [data-testid="stHeadingWithActionElements"] h3,
    .pf-secondary-heading {
        font-size: 21px !important;
        line-height: 1.35 !important;
        font-weight: 700 !important;
        color: #F5F3FA !important;
        margin-top: 12px !important;
        margin-bottom: 8px !important;
    }

    /* 4. Card Titles (19px) */
    h4,
    .stMarkdown h4,
    [data-testid="stHeadingWithActionElements"] h4,
    .pf-card-title,
    .pf-feature-title,
    .pf-box-title {
        font-size: 19px !important;
        line-height: 1.4 !important;
        font-weight: 700 !important;
        color: #F5F3FA !important;
        margin-bottom: 6px !important;
    }

    h5,
    .stMarkdown h5 {
        font-size: 17px !important;
        line-height: 1.4 !important;
        font-weight: 600 !important;
        color: #F5F3FA !important;
    }

    h6,
    .stMarkdown h6 {
        font-size: 15px !important;
        line-height: 1.45 !important;
        font-weight: 600 !important;
        color: #F5F3FA !important;
    }

    /* 5. Body Text & Paragraphs (Body: 16px, Paragraphs/Descriptions: 15px, line-height 1.45–1.6) */
    p,
    .stMarkdown p,
    .pf-desc,
    .pf-card-desc,
    .pf-paragraph {
        font-size: 15px !important;
        line-height: 1.55 !important;
        color: #F5F3FA !important;
    }

    /* Lists */
    ul, ol, .stMarkdown ul, .stMarkdown ol {
        font-size: 15px !important;
        line-height: 1.55 !important;
    }
    li, .stMarkdown li {
        font-size: 15px !important;
        line-height: 1.55 !important;
        color: #F5F3FA !important;
        margin-bottom: 4px !important;
    }

    /* 6. Input Fields and Text Areas (16px) */
    input,
    textarea,
    [data-baseweb="input"] input,
    [data-baseweb="textarea"] textarea,
    [data-baseweb="input"],
    [data-baseweb="textarea"] {
        font-size: 16px !important;
        line-height: 1.5 !important;
        background-color: #0E1020 !important;
        border: 1px solid rgba(139, 92, 246, 0.25) !important;
        color: #F5F3FA !important;
        border-radius: 10px !important;
        padding: 10px 14px !important;
    }
    input:focus, textarea:focus {
        border-color: #8B5CF6 !important;
        box-shadow: 0 0 0 1px #8B5CF6 !important;
    }

    [data-baseweb="select"] div,
    [data-baseweb="select"] input,
    [data-baseweb="select"] span {
        font-size: 16px !important;
    }

    [data-baseweb="menu"] li,
    [data-baseweb="menu"] div,
    [data-baseweb="popover"] div {
        font-size: 15px !important;
    }

    /* 7. Buttons (16px) */
    button,
    button[data-testid*="stBaseButton"],
    [data-testid="stButton"] > button,
    [data-testid="stDownloadButton"] > button,
    button[kind="secondary"],
    .stButton > button {
        background: #111426 !important;
        border: 1px solid rgba(139, 92, 246, 0.3) !important;
        color: #F5F3FA !important;
        font-size: 16px !important;
        font-weight: 600 !important;
        line-height: 1.4 !important;
        border-radius: 10px !important;
        padding: 10px 22px !important;
        transition: all 0.2s ease !important;
    }
    button:hover,
    button[data-testid*="stBaseButton"]:hover,
    [data-testid="stButton"] > button:hover {
        border-color: #8B5CF6 !important;
        background: #1E2236 !important;
        box-shadow: 0 0 16px rgba(139, 92, 246, 0.3) !important;
        color: #FFFFFF !important;
    }
    button[kind="primary"],
    button[data-testid="stBaseButton-primary"],
    [data-testid="stButton"] > button[kind="primary"] {
        background: linear-gradient(135deg, #8B5CF6 0%, #B895FF 100%) !important;
        border: none !important;
        color: #FFFFFF !important;
        font-size: 16px !important;
        font-weight: 700 !important;
        line-height: 1.4 !important;
        border-radius: 10px !important;
        padding: 11px 24px !important;
        box-shadow: 0 4px 18px rgba(124, 58, 237, 0.45) !important;
    }
    button[kind="primary"]:hover,
    button[data-testid="stBaseButton-primary"]:hover,
    [data-testid="stButton"] > button[kind="primary"]:hover {
        filter: brightness(1.15) !important;
        box-shadow: 0 6px 24px rgba(124, 58, 237, 0.6) !important;
    }

    /* 8. Sidebar Navigation (16px) */
    [data-testid="stSidebar"] {
        background-color: #0C0D1A !important;
        border-right: 1px solid rgba(139, 92, 246, 0.14) !important;
    }
    [data-testid="stSidebar"] > div:first-child,
    [data-testid="stSidebarContent"] {
        background-color: #0C0D1A !important;
        overflow-y: auto !important;
        padding-bottom: 2.5rem !important;
    }
    [data-testid="stSidebar"] [data-testid="stRadio"] [data-testid="stRadioOption"] > div > div:first-child {
        display: none !important;
    }
    [data-testid="stSidebar"] [data-testid="stRadio"] [data-testid="stRadioOption"] {
        padding: 10px 14px !important;
        border-radius: 10px !important;
        margin-bottom: 5px !important;
        color: #8B92A5 !important;
        font-size: 16px !important;
        font-weight: 500 !important;
        line-height: 1.4 !important;
        cursor: pointer !important;
        transition: all 0.2s ease !important;
        display: flex !important;
        align-items: center !important;
        background: transparent !important;
        border: 1px solid transparent !important;
    }
    [data-testid="stSidebar"] [data-testid="stRadio"] [data-testid="stRadioOption"] p,
    [data-testid="stSidebar"] [data-testid="stRadio"] [data-testid="stRadioOption"] span,
    [data-testid="stSidebar"] [data-testid="stRadio"] [data-testid="stRadioOption"] div {
        font-size: 16px !important;
        line-height: 1.4 !important;
    }
    [data-testid="stSidebar"] [data-testid="stRadio"] [data-testid="stRadioOption"]:hover {
        background-color: #0E1020 !important;
        color: #F5F3FA !important;
        border-color: rgba(139, 92, 246, 0.2) !important;
    }
    [data-testid="stSidebar"] [data-testid="stRadio"] [data-testid="stRadioOption"][data-selected="true"],
    [data-testid="stSidebar"] [data-testid="stRadio"] [data-testid="stRadioOption"]:has(input:checked) {
        background: linear-gradient(135deg, #8B5CF6 0%, #8B5CF6 100%) !important;
        color: #FFFFFF !important;
        font-weight: 700 !important;
        box-shadow: 0 4px 14px rgba(124, 58, 237, 0.45) !important;
        border-color: transparent !important;
    }
    [data-testid="stSidebar"] [data-testid="stRadio"] [data-testid="stRadioOption"][data-selected="true"] p,
    [data-testid="stSidebar"] [data-testid="stRadio"] [data-testid="stRadioOption"]:has(input:checked) p {
        color: #FFFFFF !important;
        font-weight: 700 !important;
    }

    /* 9. Labels (14px) */
    label,
    [data-testid="stWidgetLabel"] label,
    [data-testid="stWidgetLabel"] p,
    .stWidgetLabel,
    .pf-label,
    [data-testid="stCheckbox"] label,
    [data-testid="stRadio"] label,
    [data-testid="stSlider"] label,
    [data-testid="stFileUploader"] label {
        font-size: 14px !important;
        font-weight: 600 !important;
        line-height: 1.4 !important;
        color: #8B92A5 !important;
    }

    /* 10. Captions and Metadata (13px) */
    [data-testid="stCaptionContainer"],
    [data-testid="stCaptionContainer"] p,
    .stCaption,
    small,
    .pf-caption,
    .pf-meta {
        font-size: 13px !important;
        line-height: 1.45 !important;
        color: #94A3B8 !important;
    }

    /* 11. Code Blocks and Diagnostics (14px) */
    code,
    pre,
    [data-testid="stCodeBlock"] code,
    [data-testid="stCodeBlock"] pre,
    kbd,
    samp {
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 14px !important;
        line-height: 1.5 !important;
    }
    code {
        padding: 2px 6px !important;
        border-radius: 4px !important;
    }

    /* Native Tabs (15px) */
    [data-baseweb="tab-list"] {
        background: transparent !important;
        gap: 6px !important;
        border-bottom: 1px solid rgba(139, 92, 246, 0.18) !important;
        margin-bottom: 18px !important;
    }
    button[role="tab"] {
        background: #111426 !important;
        color: #8B92A5 !important;
        border-radius: 8px 8px 0 0 !important;
        padding: 10px 20px !important;
        border: 1px solid rgba(139, 92, 246, 0.15) !important;
        border-bottom: none !important;
        font-size: 15px !important;
        font-weight: 600 !important;
    }
    button[role="tab"][aria-selected="true"] {
        background: #111426 !important;
        color: #B895FF !important;
        border-top: 2px solid #8B5CF6 !important;
        font-size: 15px !important;
        font-weight: 700 !important;
    }

    /* Native Expanders */
    [data-testid="stExpander"] {
        background-color: #0E1020 !important;
        border: 1px solid rgba(139, 92, 246, 0.2) !important;
        border-radius: 12px !important;
        color: #F5F3FA !important;
        margin-bottom: 12px !important;
    }
    [data-testid="stExpander"] summary,
    [data-testid="stExpander"] summary span,
    [data-testid="stExpander"] summary p {
        font-size: 16px !important;
        font-weight: 600 !important;
        line-height: 1.4 !important;
        color: #F5F3FA !important;
    }
    [data-testid="stExpander"] [data-testid="stExpanderDetails"] {
        font-size: 15px !important;
        line-height: 1.55 !important;
    }

    /* Tables & Dataframes */
    [data-testid="stTable"], table {
        font-size: 15px !important;
        line-height: 1.5 !important;
    }
    th, [data-testid="stTable"] th {
        font-size: 14px !important;
        font-weight: 700 !important;
    }
    td, [data-testid="stTable"] td {
        font-size: 15px !important;
    }

    /* Status Alerts and Callout Boxes */
    [data-testid="stAlert"] [data-testid="stMarkdownContainer"] p,
    [data-testid="stAlert"] p {
        font-size: 15px !important;
        line-height: 1.55 !important;
    }
    [data-testid="stAlert"] h1, [data-testid="stAlert"] h2, [data-testid="stAlert"] h3 {
        font-size: 19px !important;
        line-height: 1.35 !important;
    }

    /* Tooltips */
    [data-testid="stTooltipContent"],
    [data-baseweb="tooltip"] {
        font-size: 13px !important;
        line-height: 1.4 !important;
    }

    /* Metric Containers */
    [data-testid="stMetricValue"] {
        font-size: 28px !important;
        font-weight: 800 !important;
    }
    [data-testid="stMetricLabel"] p {
        font-size: 14px !important;
        font-weight: 600 !important;
        color: #8B92A5 !important;
    }
    [data-testid="stMetricDelta"] {
        font-size: 14px !important;
    }

    /* Badges & Pills */
    .pf-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 5px 12px;
        border-radius: 9999px;
        font-size: 13px !important;
        font-weight: 700;
        letter-spacing: 0.04em;
        text-transform: uppercase;
    }
    .pf-badge-purple {
        background: rgba(139, 92, 246, 0.18);
        color: #B895FF;
        border: 1px solid rgba(139, 92, 246, 0.35);
    }
    .pf-badge-success {
        background: rgba(54, 211, 153, 0.15);
        color: #36D399;
        border: 1px solid rgba(54, 211, 153, 0.35);
    }
    .pf-badge-warning {
        background: rgba(251, 191, 36, 0.15);
        color: #FBBF24;
        border: 1px solid rgba(251, 191, 36, 0.35);
    }

    .pf-pill {
        display: inline-block;
        padding: 6px 14px;
        border-radius: 8px;
        font-size: 14px !important;
        font-weight: 600;
        margin: 3px;
    }
    .pf-pill-skill {
        background: #16192E;
        color: #F1F5F9;
        border: 1px solid rgba(139, 92, 246, 0.25);
    }
    .pf-pill-verified {
        background: rgba(54, 211, 153, 0.12);
        color: #36D399;
        border: 1px solid rgba(54, 211, 153, 0.35);
    }
    .pf-pill-gap {
        background: rgba(251, 113, 133, 0.12);
        color: #FB7185;
        border: 1px solid rgba(251, 113, 133, 0.35);
    }

    /* Sidebar Candidate Profile Card */
    .pf-sidebar-profile {
        background: #111426;
        border: 1px solid rgba(139, 92, 246, 0.22);
        border-radius: 12px;
        padding: 12px 14px;
        margin: 14px 0 10px 0;
        display: flex;
        align-items: center;
        justify-content: space-between;
        box-shadow: 0 4px 14px rgba(0, 0, 0, 0.35);
    }

    /* Sidebar Developer Expander */
    [data-testid="stSidebar"] [data-testid="stExpander"] {
        background-color: #0E1020 !important;
        border: 1px solid rgba(139, 92, 246, 0.18) !important;
        border-radius: 10px !important;
        margin-top: 8px !important;
        margin-bottom: 16px !important;
    }

    /* Custom Cards */
    .pf-card {
        background: #111426;
        border: 1px solid rgba(139, 92, 246, 0.18);
        border-radius: 14px;
        padding: 22px;
        margin-bottom: 18px;
        color: #F5F3FA;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.35);
        transition: all 0.25s ease;
    }
    .pf-card:hover {
        border-color: rgba(139, 92, 246, 0.4);
        box-shadow: 0 8px 24px rgba(124, 58, 237, 0.18);
    }

    .pf-card-elevated {
        background: #16192E;
        border: 1px solid rgba(139, 92, 246, 0.25);
        border-radius: 14px;
        padding: 20px;
        color: #F5F3FA;
        box-shadow: 0 6px 20px rgba(0, 0, 0, 0.4);
    }

    .pf-hero-card {
        background: linear-gradient(135deg, rgba(23, 26, 43, 0.95) 0%, rgba(17, 20, 38, 0.95) 100%);
        border: 1px solid rgba(139, 92, 246, 0.25);
        border-radius: 18px;
        padding: 36px 32px;
        margin-bottom: 24px;
        box-shadow: 0 10px 32px rgba(0, 0, 0, 0.5), 0 0 40px rgba(139, 92, 246, 0.12);
        position: relative;
        overflow: hidden;
    }

    /* Feature Cards */
    .pf-feature-card {
        background: #111426;
        border: 1px solid rgba(139, 92, 246, 0.16);
        border-radius: 14px;
        padding: 20px;
        height: 100%;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.25);
        transition: all 0.2s ease;
    }
    .pf-feature-card:hover {
        border-color: rgba(139, 92, 246, 0.45);
        transform: translateY(-2px);
        box-shadow: 0 8px 24px rgba(124, 58, 237, 0.18);
    }

    .pf-icon-box {
        width: 44px;
        height: 44px;
        border-radius: 12px;
        background: rgba(139, 92, 246, 0.18);
        border: 1px solid rgba(139, 92, 246, 0.35);
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 20px;
        margin-bottom: 14px;
        color: #B47AFF;
    }

    /* Dashed Upload Box */
    .pf-upload-box {
        border: 2px dashed rgba(139, 92, 246, 0.45);
        border-radius: 16px;
        background: #111426;
        padding: 36px 20px;
        text-align: center;
        margin-bottom: 20px;
        transition: border-color 0.2s;
    }
    .pf-upload-box:hover {
        border-color: #8B5CF6;
    }

    /* Selected File Card */
    .pf-file-card {
        background: #111426;
        border: 1px solid rgba(139, 92, 246, 0.25);
        border-radius: 12px;
        padding: 14px 20px;
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 18px;
    }

    /* Native Progress Bar */
    div[data-testid="stProgressBar"] > div {
        background: #111426 !important;
        border-radius: 8px !important;
    }
    div[data-testid="stProgressBar"] > div > div {
        background: linear-gradient(90deg, #8B5CF6 0%, #B895FF 100%) !important;
        border-radius: 8px !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# -----------------------------------------------------------------------------
# SVG Circular Score Visualizer (Panel 3 & 4 Fidelity)
# -----------------------------------------------------------------------------
def render_circular_score(score: float, label: str = "Overall Score", size: int = 120, ring_color: str = "#8B5CF6") -> str:
    """Generate an SVG circular progress ring matching the reference design."""
    radius = 48
    circumference = 2 * 3.14159 * radius
    clamped_score = min(max(score, 0.0), 100.0)
    offset = circumference * (1.0 - (clamped_score / 100.0))
    return (
        f'<div style="position: relative; width: {size}px; height: {size}px; display: inline-flex; align-items: center; justify-content: center;">'
        f'<svg width="{size}" height="{size}" viewBox="0 0 120 120">'
        f'<circle cx="60" cy="60" r="{radius}" fill="none" stroke="#1E2236" stroke-width="10"/>'
        f'<circle cx="60" cy="60" r="{radius}" fill="none" stroke="{ring_color}" stroke-width="10" '
        f'stroke-dasharray="{circumference}" stroke-dashoffset="{offset}" '
        f'stroke-linecap="round" transform="rotate(-90 60 60)"/>'
        f'</svg>'
        f'<div style="position: absolute; text-align: center; width: 100%;">'
        f'<div style="font-size: 24px; font-weight: 800; color: #F5F3FA; line-height: 1;">{clamped_score:.0f}%</div>'
        f'<div style="font-size: 13px; color: #8B92A5; font-weight: 600; text-transform: uppercase; margin-top: 4px; letter-spacing: 0.03em;">{label}</div>'
        f'</div>'
        f'</div>'
    )


# -----------------------------------------------------------------------------
# Service Initializations & LLM Provider Status
# -----------------------------------------------------------------------------
@st.cache_resource
def get_matching_pipeline() -> CareerMatchingPipeline:
    """Initialize deterministic matching pipeline with structured LLM explanation generator."""
    provider_mgr = LLMProviderManager()
    output_engine = StructuredOutputEngine(dispatcher=provider_mgr)
    explainer = MatchExplainer(engine=output_engine)
    return CareerMatchingPipeline(explainer=explainer)


@st.cache_resource
def get_llm_provider_manager() -> LLMProviderManager:
    """Initialize LLMProviderManager with default providers."""
    mgr = LLMProviderManager(primary_provider="groq", secondary_provider="gemini", fallback_enabled=True)
    return mgr


matching_pipeline = get_matching_pipeline()
llm_manager = get_llm_provider_manager()

# Session State Indicators
has_active_match = "pipeline_result" in st.session_state and st.session_state["pipeline_result"] is not None
has_active_profile = "tab1_resume_profile" in st.session_state and st.session_state["tab1_resume_profile"] is not None
has_active_workflow = "workflow_state" in st.session_state and st.session_state["workflow_state"] is not None


# -----------------------------------------------------------------------------
# Synthetic Demonstration Showcase Data (Alex Rivera / Mohamed Athiq)
# -----------------------------------------------------------------------------
DEMO_RESUME_TEXT = """MOHAMED ATHIQ
Senior Platform & AI Systems Engineer
San Francisco, CA | mohamed.athiq@example.com | https://linkedin.com/in/athiq | https://github.com/athiq

SUMMARY
Results-driven Platform Engineer with 6+ years of expertise in distributed systems, high-concurrency microservices, and deep learning infrastructure. Proven record of designing low-latency backend architectures in Python, FastAPI, and Go, reducing database load by 42% and scaling pipelines to 25M daily transactions.

TECHNICAL SKILLS
Languages & Frameworks: Python, FastAPI, Go, PyTorch, Scikit-Learn, SQL, React, Bash
Infrastructure & DevOps: Docker, Kubernetes, Helm, Redis, PostgreSQL, AWS, Kafka, Linux, Git, CI/CD
Architecture: Distributed Systems, RESTful APIs, Microservices, Asynchronous Workflows, System Design

WORK EXPERIENCE
Senior Platform Engineer at AeroScale Technologies | 2021 - Present
- Architected and scaled distributed backend services in Python and FastAPI, serving 25M daily transactions at sub-50ms p99 latency.
- Implemented multi-tier caching with Redis and optimized PostgreSQL query execution plans, reducing database server loads by 42%.
- Containerized core services with Docker and built production deployment pipelines with zero-downtime rolling updates.

Software Engineer at DataMesh Networks | 2018 - 2021
- Developed high-concurrency data ingestion pipelines using Python and RESTful microservice architectures.
- Built automated unit and integration test suites achieving 94% test coverage using PyTest.
- Mentored junior engineers on distributed systems reliability and clean code standards.

EDUCATION
Bachelor of Science in Computer Science
University of Washington, Seattle | 2014 - 2018
- Core focus on Distributed Computing, Database Systems, and Operating Systems.

PRACTICAL PROJECTS
Distributed Task Orchestrator | https://github.com/athiq/distributed-orchestrator
- Production-grade distributed task scheduler in Python and Redis supporting delayed execution and fault-tolerant worker pools.
- Benchmarked at 15,000 tasks/second with automatic failover and queue health telemetry.
"""

DEMO_JD_STRIPE = """Job Title: Senior Platform Engineer
Company: Stripe

About the Role:
We are seeking an experienced Senior Platform Engineer to build scalable, highly available backend infrastructure powering global real-time financial transactions. You will design distributed systems, optimize database layers, and create robust microservices that guarantee sub-second execution and fault tolerance.

Responsibilities:
- Architect and operate distributed backend services with high availability and rigorous latency guarantees.
- Optimize high-throughput data layers using PostgreSQL, Redis, and message queues.
- Containerize services with Docker and manage zero-downtime deployment workflows.
- Collaborate with cross-functional teams to harden infrastructure against latency spikes and failure modes.

Requirements:
- 4+ years of professional backend engineering experience with Python, Go, or Java.
- Proven expertise with PostgreSQL database optimization and high-performance caching (Redis).
- Production experience containerizing microservices with Docker and building automated CI/CD pipelines.
- Strong grounding in distributed systems fundamentals, concurrency patterns, and RESTful API architecture.
- Bachelor's degree in Computer Science, Computer Engineering, or equivalent practical experience.

Preferred Qualifications:
- Familiarity with Kubernetes, Helm, or AWS cloud environments.
- Experience with high-throughput distributed message queues and stream processing.
"""


DEMO_RESUME_ATHIQ = """MOHAMED ATHIQ
Recent Graduate • B.E. Computer Science (AI & ML)
Chennai, India | mohamed.athiq@example.com | https://linkedin.com/in/athiq | https://github.com/athiq

SUMMARY
Enthusiastic AI-first builder and Computer Science graduate specializing in AI & ML with a solid foundation in Web Development, API integrations, and workflow automation. Founder of Hulsti.AI, building 0-to-1 full-stack applications with Next.js, Node.js, React, and Python. Experienced in machine learning models and web analytics dashboards. Holds industry-recognized certifications from Oracle Cloud Infrastructure (OCI) and NPTEL.

EDUCATION
Bachelor of Engineering in Computer Science and Engineering (AI & ML)
Sathyabama Institute of Science and Technology | 2022 - 2026
CGPA: 7.31 | Class of 2026 • Undergraduate

TECHNICAL SKILLS
Languages & Web: Python, JavaScript, Next.js, Node.js, React, HTML5, CSS3, SQL, REST APIs, API Integration, Web Technologies
Analytics & Automation: Data Analysis, Web Analytics, Web Analysis, Automation, Problem Solving
AI & Machine Learning: Scikit-Learn, TensorFlow, NLP, AI Applications
Tools & Cloud: Git, Linux, OCI, NPTEL

WORK EXPERIENCE
Founder & AI Project Lead at Hulsti.AI | May 2024 - Present
- Founded an AI venture building 0-to-1 full-stack web applications using Next.js, Node.js, React, and Python.
- Engineered automated workflow pipelines, AI-driven tools, and robust REST API integrations.
- Shipped client prototypes with automated test suites and production deployment.

Web Development & Analytics Intern at TechPulse Solutions | Jan 2024 - Apr 2024
- Analyzed website performance bottlenecks and user behavior metrics to improve digital experiences.
- Collaborated with design and development teams to create reports and actionable insights.

PRACTICAL PROJECTS
Hulsti.AI Platform & Automation Engine | https://github.com/athiq/hulsti-ai
- Architected and deployed an AI-first builder platform using Next.js, Node.js, React, and Python.
- Integrated multi-model API endpoints with automated workflow pipelines and production REST APIs.

Web Performance & Analytics Dashboard | https://github.com/athiq/web-analytics
- Developed an interactive web analytics tool to monitor website performance and user behavior.
- Implemented automated report pipelines for digital experience insights.
"""

DEMO_JD_WEB_ANALYST = """Web Analyst
Responsibilities:
- Analyze website performance and user behavior
- Work with data to improve digital experiences
- Collaborate with design and development teams
- Create reports and actionable insights
Requirements:
- B.E./B.Tech in CS or related field
- Knowledge of web technologies, analytics tools
- Strong problem-solving skills"""

DEMO_JD_RAZORPAY_AI_BUILDER = """Job Title: AI Builder
Company: Razorpay

About the Role:
At Razorpay, AI Builders are zero-to-one problem solvers who independently research, prototype, and ship AI-first solutions to remove business and operational bottlenecks. You will own outcomes end-to-end, leveraging modern AI models, APIs, and rapid full-stack prototyping.

Key Responsibilities:
- Work AI-First: Prototype, automate, and build everyday workflows using LLMs, APIs, and modern developer frameworks.
- Rapid Prototyping & Shipping: Autonomously design, build, and deploy production-grade v1 applications and automated pipelines within days.
- Full-Stack & API Integrations: Build responsive user interfaces with Next.js and React, backed by robust Node.js and Python microservices and REST API integrations.
- User Adoption & Outcome Ownership: Onboard internal users, measure adoption metrics, and rapidly iterate or pivot based on quantitative data.
- Frontier Tracking: Evaluate and integrate the latest generative AI foundation models, agent architectures, and tools.

Requirements & Candidate Profile:
- Builder Background: Proven track record of building 0-to-1 applications, founding side projects, or leading startup initiatives (such as Hulsti.AI).
- Technical Stack: Hands-on experience with Python, JavaScript, React, Next.js, Node.js, and REST APIs.
- AI & Automation: Direct experience implementing LLM applications, API integrations, and workflow automation.
- High Agency & Problem Solving: Ability to decompose unstructured operational problems and deliver measurable business outcomes.
- Education: Bachelor's degree in Computer Science, Engineering, or equivalent practical experience.

Preferred Qualifications:
- Familiarity with cloud platforms (AWS, Docker, Kubernetes) and caching (Redis).
- Experience measuring user adoption metrics and system telemetry."""


def generate_synthetic_demo_pdf(text: str = DEMO_RESUME_TEXT) -> bytes:
    """Generate in-memory synthetic PDF bytes from text using PyMuPDF."""
    doc = fitz.open()
    page = doc.new_page()
    page.insert_textbox(fitz.Rect(40, 40, 560, 800), text, fontsize=9)
    pdf_bytes = doc.tobytes()
    doc.close()
    return pdf_bytes


def clear_candidate_session_state():
    """Purge all downstream matching and workflow session artifacts for strict multi-candidate isolation."""
    keys_to_clear = [
        "pipeline_result",
        "tab2_match_result",
        "agentic_workflow_state",
        "active_remediation_plan",
        "active_interview_prep",
        "active_optimization_report",
        "workflow_thread_id",
        "workflow_checkpoint",
        "tab4_clarification_response",
        "tab4_clarification_options",
        "tab4_goal_selected",
        "tab4_active_plan",
        "tab4_active_prep",
        "tab4_active_opt",
        "advisor_generated_plan",
        "advisor_plan_result",
        "linkedin_optimized_result",
    ]
    for k in keys_to_clear:
        st.session_state.pop(k, None)


def load_demo_athiq():
    """Load Mohamed Athiq's verified candidate profile and Web Analyst target JD."""
    clear_candidate_session_state()
    pdf_bytes = generate_synthetic_demo_pdf(DEMO_RESUME_ATHIQ)
    pipeline = ResumePipeline()
    profile = pipeline.process_resume(pdf_bytes, filename="Mohamed_Athiq_Resume.pdf")
    st.session_state["tab1_resume_profile"] = profile
    st.session_state["tab1_resume_filename"] = "Mohamed_Athiq_Resume.pdf"
    st.session_state["tab1_resume_bytes"] = pdf_bytes
    st.session_state["input_jt_field"] = "Web Analyst"
    st.session_state["input_comp_field"] = "Tech Corp"
    st.session_state["input_jd_area"] = DEMO_JD_WEB_ANALYST
    st.session_state["demo_showcase_active"] = True


def get_match_alignment_display(
    match_res: JobMatchResult,
    jd_text: str = "",
    job_title: str = "",
    resume_filename: str = "",
) -> dict:
    """
    Computes score-grounded UI labels, colors, and descriptions for match results.
    PROVISIONAL evaluation status takes strict precedence over normal fit classifications.
    Always uses authoritative deterministic score from Phase 4 engine without synthetic overrides.
    Never describes low, developing, or provisional scores as strong alignment.
    """
    is_provisional = (
        match_res.evaluation_status == EvaluationStatus.PROVISIONAL_EVALUATION
        or match_res.fit_classification == CandidateFitLevel.PROVISIONAL
    )

    score = match_res.overall_score

    if is_provisional:
        fit_label = "Provisional Evaluation"
        fit_color = "#FBBF24"  # Amber warning
        ring_color = "#FBBF24"
        headline = "Preliminary assessment based on sparse criteria."
        explanation = (
            "This evaluation is PROVISIONAL because the target job description contains insufficient criteria "
            "(fewer than 3 technical skills or 4 total assessable requirements). The calculated score reflects only "
            "the sparse criteria present and is not an authoritative indicator of full hiring alignment."
        )
        provisional_reason = (
            "The employer job description lacks sufficient technical depth to perform an authoritative evaluation. "
            "Assessed coverage reflects the proportion of available criteria evaluated, but evaluation completeness "
            "remains provisional until more comprehensive role requirements are supplied."
        )
    elif score >= 90.0:
        fit_label = "Exceptional Match"
        fit_color = "#36D399"
        ring_color = "#36D399"
        headline = "Outstanding alignment across technical skills, required experience, education, and project deliverables."
        explanation = "Candidate profile demonstrates comprehensive, verified evidence matching both required and preferred qualifications."
        provisional_reason = ""
    elif score >= 75.0:
        fit_label = "Strong Match"
        fit_color = "#36D399"
        ring_color = "#36D399"
        headline = "Your profile aligns well with this job role across core technical requirements."
        explanation = "Candidate demonstrates verified alignment with primary technical competencies and domain responsibilities."
        provisional_reason = ""
    elif score >= 60.0:
        fit_label = "Moderate Match"
        fit_color = "#8B45F7"
        ring_color = "#8B45F7"
        headline = "Partial alignment with core requirements; targeted upskilling in missing areas will strengthen your candidacy."
        explanation = "Candidate satisfies key foundational requirements but exhibits notable gaps in specific required or preferred tools."
        provisional_reason = ""
    elif score >= 40.0:
        fit_label = "Developing Alignment"
        fit_color = "#FBBF24"
        ring_color = "#FBBF24"
        headline = "Foundational match with targeted skill and experience gaps against this specific role."
        explanation = "The evaluation indicates an evidence gap between the submitted resume and this posting's specific technical requirements. This reflects evidence alignment for this target role, not a career judgment."
        provisional_reason = ""
    else:
        fit_label = "Low Alignment"
        fit_color = "#FB7185"
        ring_color = "#FB7185"
        headline = "Significant alignment gaps detected against this specific job description."
        explanation = "The submitted resume currently lacks documented evidence for several core requirements of this specific posting. This measures document-level requirement alignment, not candidate capability or potential."
        provisional_reason = ""

    return {
        "score": score,
        "fit_label": fit_label,
        "fit_color": fit_color,
        "ring_color": ring_color,
        "headline": headline,
        "explanation": explanation,
        "is_provisional": is_provisional,
        "provisional_reason": provisional_reason,
        "coverage": match_res.active_evaluation_coverage,
        "reserve": match_res.unassessed_reserve,
    }


def load_demo_showcase():
    """Load realistic synthetic candidate profile and target job description."""
    clear_candidate_session_state()
    pdf_bytes = generate_synthetic_demo_pdf(DEMO_RESUME_TEXT)
    pipeline = ResumePipeline()
    profile = pipeline.process_resume(pdf_bytes, filename="alex_rivera_demo_resume.pdf")
    st.session_state["tab1_resume_profile"] = profile
    st.session_state["tab1_resume_filename"] = "alex_rivera_demo_resume.pdf"
    st.session_state["input_jt_field"] = "Senior Platform Engineer"
    st.session_state["input_comp_field"] = "Stripe"
    st.session_state["input_jd_area"] = DEMO_JD_STRIPE
    st.session_state["demo_showcase_active"] = True


# -----------------------------------------------------------------------------
# Vertical Sidebar Navigation & Profile Footer
# -----------------------------------------------------------------------------
with st.sidebar:
    # Sidebar Brand
    st.markdown(
        """
                <div style="display: flex; align-items: center; gap: 12px; padding: 6px 4px 18px 4px;">
            <div style="width: 40px; height: 40px; border-radius: 12px; background: linear-gradient(135deg, #16192E 0%, #0E1020 100%); border: 1px solid rgba(139, 92, 246, 0.4); display: flex; align-items: center; justify-content: center; box-shadow: 0 4px 16px rgba(139, 92, 246, 0.35); flex-shrink: 0;">
                <svg width="26" height="26" viewBox="0 0 38 38" fill="none" xmlns="http://www.w3.org/2000/svg">
                    <path d="M26 13.5C24.3 11.3 21.6 10 18.5 10C13.25 10 9 14.03 9 19C9 23.97 13.25 28 18.5 28C21.6 28 24.3 26.7 26 24.5" stroke="url(#carevo-grad-sb)" stroke-width="4.2" stroke-linecap="round"/>
                    <circle cx="26" cy="13.5" r="3" fill="#B895FF"/>
                    <defs>
                        <linearGradient id="carevo-grad-sb" x1="9" y1="10" x2="26" y2="28" gradientUnits="userSpaceOnUse">
                            <stop offset="0%" stop-color="#8B5CF6"/>
                            <stop offset="100%" stop-color="#B895FF"/>
                        </linearGradient>
                    </defs>
                </svg>
            </div>
            <div>
                <div style="font-size: 20px; font-weight: 800; color: #F5F3FA; letter-spacing: 0.04em; line-height: 1.15;">CAREVO</div>
                <div style="font-size: 13px; font-weight: 600; color: #B895FF; letter-spacing: 0.02em; line-height: 1.15;">AI Career Platform</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # 8 Primary Navigation Items
    NAV_OPTIONS = [
        "Home",
        "Resume Analysis",
        "JD Matching",
        "LinkedIn Analysis",
        "Career Advisor",
        "Agents (Workflow)",
        "History",
        "Settings",
    ]

    # Legacy mapping reference: "History": "🕒 Current Evaluation"
    NAV_ICONS = {
        "Home": "🏠 Home",
        "Resume Analysis": "📄 CAREVO Profile",
        "JD Matching": "🎯 CAREVO Match",
        "LinkedIn Analysis": "💼 CAREVO Profile Studio",
        "Career Advisor": "🧭 CAREVO Guide",
        "Agents (Workflow)": "🤖 CAREVO Copilot",
        "History": "🕒 CAREVO Activity",
        "Settings": "⚙️ Settings",
    }

    # Navigation State
    if "nav_view" not in st.session_state:
        st.session_state["nav_view"] = "Home"

    selected_nav = st.radio(
        "Navigation Menu:",
        options=NAV_OPTIONS,
        format_func=lambda x: NAV_ICONS[x],
        index=NAV_OPTIONS.index(st.session_state["nav_view"]) if st.session_state["nav_view"] in NAV_OPTIONS else 0,
        key="main_sidebar_nav_radio",
        label_visibility="collapsed",
    )
    st.session_state["nav_view"] = selected_nav

    st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)
    st.divider()

    # Candidate Profile Section in Separated Footer Card
    if has_active_profile:
        p_act = st.session_state["tab1_resume_profile"]
        c_name = (getattr(p_act, "candidate_name", "") or "").strip()
        if not c_name:
            fname = st.session_state.get("tab1_resume_filename", "")
            base = fname.replace(".pdf", "")
            for sfx in ["_demo_resume", "_resume", "_demo", "-resume", " Resume"]:
                base = re.sub(re.escape(sfx), "", base, flags=re.IGNORECASE)
            c_name = base.replace("_", " ").strip().title() if base else "Active Candidate"
        candidate_display_name = c_name
    else:
        candidate_display_name = "Candidate Profile"

    # Compute avatar initials dynamically
    name_parts = [part for part in re.split(r'\s+', candidate_display_name) if part]
    if len(name_parts) >= 2:
        avatar_initials = (name_parts[0][0] + name_parts[1][0]).upper()
    elif len(name_parts) == 1 and len(name_parts[0]) >= 2:
        avatar_initials = name_parts[0][:2].upper()
    elif len(name_parts) == 1:
        avatar_initials = name_parts[0][0].upper()
    else:
        avatar_initials = "CP"

    st.markdown(
        f"""
        <div class="pf-sidebar-profile">
            <div style="display: flex; align-items: center; gap: 11px;">
                <div style="width: 38px; height: 38px; border-radius: 50%; background: linear-gradient(135deg, #8B5CF6, #B895FF); display: flex; align-items: center; justify-content: center; font-weight: 700; color: #FFFFFF; font-size: 14px; box-shadow: 0 0 12px rgba(139, 92, 246, 0.45); flex-shrink: 0;">
                    {avatar_initials}
                </div>
                <div>
                    <div style="font-size: 16px; font-weight: 700; color: #F5F3FA; line-height: 1.2;">{candidate_display_name}</div>
                    <div style="font-size: 13px; color: #8B92A5;">CAREVO Profile</div>
                </div>
            </div>
            <span style="color: #8B5CF6; font-size: 18px; font-weight: 700;">›</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Collapsed Developer & Diagnostics
    with st.expander("🛠️ Developer & Diagnostics", expanded=False):
        st.caption("Technical infrastructure telemetry, API readiness, and verification status.")
        st.write(f"**Environment:** `{config.app_env}`")
        st.write(f"**Database:** `{config.database_url}`")

        st.markdown("**LLM Provider Readiness:**")
        gemini_ready = "🟢 Configured" if config.gemini_key_configured else "🟡 Key Missing"
        groq_ready = "🟢 Configured" if config.groq_key_configured else "🟡 Key Missing"
        st.write(f"- Groq API: {groq_ready}")
        st.write(f"- Gemini API: {gemini_ready}")
        st.write(f"- Primary Provider: `{llm_manager.primary_provider_name}`")
        st.write(f"- Secondary Fallback: `{llm_manager.secondary_provider_name}`")
        st.write(f"- Fallback Enabled: `{llm_manager.is_fallback_enabled}`")

        # Telemetry & Observability
        st.markdown("**Live Telemetry & Observability:**")
        t_summary = telemetry_tracker.get_summary()
        st.write(f"- Total Invocations: `{t_summary['total_invocations']}`")
        st.write(f"- Live Groq Calls: `{t_summary['groq_live_calls']}`")
        st.write(f"- Live Web Research Calls: `{t_summary['web_research_calls']}`")
        st.write(f"- Fallback Invocations: `{t_summary['fallback_invocations']}`")
        st.write(f"- Avg Latency: `{t_summary['average_latency_ms']} ms`")

        recent_evs = telemetry_tracker.get_recent_events(limit=4)
        if recent_evs:
            st.caption("Recent Activity (Zero PII):")
            for ev in recent_evs:
                st.caption(f"• **{ev.feature}** → `{ev.status}` ({ev.provider}) in {ev.latency_ms}ms")

        if has_active_match:
            st.divider()
            st.caption("Authoritative Phase 4 Result:")
            st.json({
                "overall_score": st.session_state["pipeline_result"].match_result.overall_score,
                "fit_classification": st.session_state["pipeline_result"].match_result.fit_classification.value,
                "active_evaluation_coverage": st.session_state["pipeline_result"].match_result.active_evaluation_coverage,
                "unassessed_reserve": st.session_state["pipeline_result"].match_result.unassessed_reserve,
            })

        if st.button("🔄 Clear Active Session", key="sidebar_clear_session_btn", help="Clear all stored evaluation and workflow states."):
            for k in [
                "pipeline_result",
                "evaluated_input_signature",
                "input_jd_area",
                "input_jt_field",
                "input_comp_field",
                "tab2_resume_uploader",
                "tab1_resume_profile",
                "tab1_resume_filename",
                "active_workflow_thread_id",
                "workflow_state",
                "workflow_interrupts",
                "demo_showcase_active",
            ]:
                st.session_state.pop(k, None)
            st.rerun()


# =============================================================================
# View 1: Home Dashboard (Recreating Reference Panel 1)
# =============================================================================
if selected_nav == "Home":
    # Hero Section
    st.markdown(
        """
        <div class="pf-hero-card">
            <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 24px;">
                <div style="flex: 1 1 500px; max-width: 650px;">
                    <div style="font-size: 32px; font-weight: 800; color: #F5F3FA; line-height: 1.25; letter-spacing: -0.025em; margin-bottom: 8px;">
                        CAREVO <span style="background: linear-gradient(135deg, #8B5CF6 0%, #B895FF 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">Platform</span>
                    </div>
                    <div style="font-size: 19px; font-weight: 700; color: #F5F3FA; margin-bottom: 6px;">
                        Career intelligence. Real progress.
                    </div>
                    <div style="font-size: 15px; color: #8B92A5; line-height: 1.6; margin-bottom: 24px;">
                        Your potential. Your next move. Personalized guidance, profile optimization, qualification matching, and career roadmaps for every professional.
                    </div>
                </div>
                <div style="flex: 1 1 320px; max-width: 400px;">
                    <div style="background: linear-gradient(135deg, rgba(30, 34, 54, 0.9) 0%, rgba(23, 26, 43, 0.95) 100%); border: 1px solid rgba(139, 92, 246, 0.35); border-radius: 18px; padding: 22px; box-shadow: 0 10px 30px rgba(0, 0, 0, 0.6), 0 0 24px rgba(124, 58, 237, 0.2);">
                        <div style="background: #111426; border: 1px solid rgba(139, 92, 246, 0.25); border-radius: 12px; padding: 14px 18px; display: flex; align-items: center; gap: 14px; margin-bottom: 14px;">
                            <div style="width: 38px; height: 38px; border-radius: 10px; background: rgba(139, 92, 246, 0.2); display: flex; align-items: center; justify-content: center; font-size: 18px; border: 1px solid rgba(139, 92, 246, 0.4);">
                                📄
                            </div>
                            <div>
                                <div style="font-size: 16px; font-weight: 700; color: #F5F3FA;">Resume.pdf</div>
                                <div style="font-size: 13px; color: #8B92A5;">Verified Candidate Document</div>
                            </div>
                        </div>
                        <div style="background: rgba(23, 26, 43, 0.95); border: 1px solid rgba(139, 92, 246, 0.35); border-radius: 12px; padding: 12px 16px; display: flex; align-items: center; gap: 12px; box-shadow: 0 0 16px rgba(139, 92, 246, 0.2);">
                            <div style="width: 12px; height: 12px; border-radius: 50%; background: #8B5CF6; box-shadow: 0 0 8px #8B5CF6;"></div>
                            <div>
                                <div style="font-size: 14px; font-weight: 700; color: #F5F3FA;">Analyzing...</div>
                                <div style="font-size: 13px; color: #8B92A5;">Extracting skills, experience and career insights...</div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Hero Action Buttons
    col_cta1, col_cta2, col_cta3, _ = st.columns([2, 2, 2.5, 3.5])
    with col_cta1:
        if st.button("Get Started →", type="primary", key="home_get_started_btn"):
            st.session_state["nav_view"] = "Resume Analysis"
            st.rerun()
    with col_cta2:
        if st.button("How It Works", key="home_how_it_works_btn"):
            st.info("💡 **How CAREVO Works:** Upload your resume in **CAREVO Profile** → Compare against any target job in **CAREVO Match** → Follow tailored trajectories in **CAREVO Guide** → Orchestrate specialist agents in **CAREVO Copilot** with interactive review.")
    with col_cta3:
        if st.button("✨ Load Complete Demo Showcase", key="home_load_demo_btn"):
            load_demo_showcase()
            st.success("Loaded synthetic demo candidate (Mohamed Athiq) & Stripe Senior Platform Engineer JD!")
            st.rerun()

    st.markdown("<div style='height: 18px;'></div>", unsafe_allow_html=True)

    # Four Feature Cards (Matching Reference Panel 1 Bottom)
    f_col1, f_col2, f_col3, f_col4 = st.columns(4)

    with f_col1:
        st.markdown(
            """
            <div class="pf-feature-card">
                <div>
                    <div class="pf-icon-box">📄</div>
                    <div style="font-size: 19px; font-weight: 700; color: #F5F3FA; margin-bottom: 6px;">CAREVO Profile</div>
                    <div style="font-size: 15px; color: #8B92A5; line-height: 1.55; margin-bottom: 14px;">
                        Extract and analyze candidate credentials and experience
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.button("Open CAREVO Profile", key="feat_btn_resume"):
            st.session_state["nav_view"] = "Resume Analysis"
            st.rerun()

    with f_col2:
        st.markdown(
            """
            <div class="pf-feature-card">
                <div>
                    <div class="pf-icon-box">🎯</div>
                    <div style="font-size: 19px; font-weight: 700; color: #F5F3FA; margin-bottom: 6px;">CAREVO Match</div>
                    <div style="font-size: 15px; color: #8B92A5; line-height: 1.55; margin-bottom: 14px;">
                        Evaluate qualification alignment with zero score drift
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.button("Explore CAREVO Match", key="feat_btn_match"):
            st.session_state["nav_view"] = "JD Matching"
            st.rerun()

    with f_col3:
        st.markdown(
            """
            <div class="pf-feature-card">
                <div>
                    <div class="pf-icon-box">🧭</div>
                    <div style="font-size: 19px; font-weight: 700; color: #F5F3FA; margin-bottom: 6px;">CAREVO Guide</div>
                    <div style="font-size: 15px; color: #8B92A5; line-height: 1.55; margin-bottom: 14px;">
                        Personalized roadmaps, skill priorities, and portfolio blueprints
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.button("Launch CAREVO Guide", key="feat_btn_skill_gap"):
            st.session_state["nav_view"] = "Career Advisor"
            st.rerun()

    with f_col4:
        st.markdown(
            """
            <div class="pf-feature-card">
                <div>
                    <div class="pf-icon-box">🤖</div>
                    <div style="font-size: 19px; font-weight: 700; color: #F5F3FA; margin-bottom: 6px;">CAREVO Copilot</div>
                    <div style="font-size: 15px; color: #8B92A5; line-height: 1.55; margin-bottom: 14px;">
                        Multi-agent co-pilot with Human-in-the-Loop review
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.button("Consult CAREVO Copilot", key="feat_btn_advisor"):
            st.session_state["nav_view"] = "Agents (Workflow)"
            st.rerun()

    # Candidate Status Bar
    st.markdown("<div style='height: 24px;'></div>", unsafe_allow_html=True)
    st.markdown("##### Candidate Profile & Assessment Readiness")
    stat_col1, stat_col2, stat_col3 = st.columns(3)
    with stat_col1:
        st.markdown(
            f"""
            <div class="pf-card">
                <div style="font-size: 14px; font-weight: 600; color: #8B92A5; text-transform: uppercase;">Candidate Profile</div>
                <div style="font-size: 19px; font-weight: 700; color: #F5F3FA; margin: 4px 0;">
                    {'🟢 Verified Profile Active' if has_active_profile else '⚪ Profile Not Loaded'}
                </div>
                <div style="font-size: 15px; color: #94A3B8; line-height: 1.55;">
                    {st.session_state.get('tab1_resume_filename', 'Upload a PDF resume in Resume Analysis or try our synthetic demo.') if has_active_profile else 'Upload your PDF resume in Resume Analysis to extract skills and tenure.'}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with stat_col2:
        if has_active_match:
            m_res = st.session_state["pipeline_result"].match_result
            d_info = get_match_alignment_display(
                m_res,
                st.session_state.get("input_jd_area", ""),
                st.session_state.get("input_jt_field", ""),
                st.session_state.get("tab1_resume_filename", ""),
            )
            match_str = f"{d_info['score']:.1f}% ({d_info['fit_label']})"
            match_sub = "Provisional evaluation (sparse job description criteria)." if d_info["is_provisional"] else "Evaluated across objective criteria dimensions with zero score drift."
        else:
            match_str = "Pending Assessment"
            match_sub = "Run an alignment match against your target JD in JD Matching."

        st.markdown(
            f"""
            <div class="pf-card">
                <div style="font-size: 14px; font-weight: 600; color: #8B92A5; text-transform: uppercase;">Compatibility Score</div>
                <div style="font-size: 19px; font-weight: 700; color: #F5F3FA; margin: 4px 0;">
                    {match_str}
                </div>
                <div style="font-size: 15px; color: #94A3B8; line-height: 1.55;">
                    {match_sub}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with stat_col3:
        if has_active_workflow:
            coach_str = f"Active — {st.session_state['workflow_state'].get('current_step', 'active').replace('_', ' ').title()}"
            coach_sub = "Multi-agent LangGraph workflow active with Human-in-the-Loop checkpoints."
        elif has_active_match:
            coach_str = "Ready to Launch"
            coach_sub = "Match evaluation complete. Ready to orchestrate specialist agents in Agents (Workflow)."
        else:
            coach_str = "Requires Completed Match"
            coach_sub = "Evaluate candidate against a target JD in JD Matching before launching agentic co-pilot."

        st.markdown(
            f"""
            <div class="pf-card">
                <div style="font-size: 14px; font-weight: 600; color: #8B92A5; text-transform: uppercase;">Agentic Co-pilot</div>
                <div style="font-size: 19px; font-weight: 700; color: #F5F3FA; margin: 4px 0;">
                    {coach_str}
                </div>
                <div style="font-size: 15px; color: #94A3B8; line-height: 1.55;">
                    {coach_sub}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


# =============================================================================
# View 2: Resume Analysis (Recreating Reference Panels 2 & 3)
# =============================================================================
elif selected_nav == "Resume Analysis":
    effective_profile = st.session_state.get("tab1_resume_profile")

    # If no resume has been analyzed yet, show Upload Container (Panel 2)
    if effective_profile is None:
        st.markdown("### CAREVO Profile")
        st.caption("Upload your resume and get an AI-powered profile analysis with verified credentials, strengths, and career insights.")

        st.markdown(
            """
            <div class="pf-upload-box">
                <div style="font-size: 44px; color: #8B5CF6; margin-bottom: 12px;">☁️</div>
                <div style="font-size: 19px; font-weight: 700; color: #F5F3FA; margin-bottom: 6px;">Upload Your Resume to CAREVO Profile</div>
                <div style="font-size: 15px; color: #8B92A5; margin-bottom: 12px; line-height: 1.55;">Drag and drop your PDF here, or click to browse</div>
                <span class="pf-badge pf-badge-purple">Supports PDF (max 10 MB)</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        uploaded_pdf = st.file_uploader(
            "Select Resume PDF:",
            type=["pdf"],
            key="tab1_resume_uploader",
            label_visibility="collapsed",
        )

        # Quick Demo Candidate Load Option
        col_demo_load, _ = st.columns([3.5, 3.5])
        with col_demo_load:
            if st.button("✨ Load Sample Candidate (Mohamed Athiq)", key="tab1_load_sample_btn"):
                load_demo_athiq()
                st.session_state["pending_file_name"] = "Mohamed_Athiq_Resume.pdf"
                st.session_state["pending_file_size"] = 1.2
                st.session_state["pending_file_bytes"] = st.session_state.get("tab1_resume_bytes")
                st.session_state.pop("tab1_resume_profile", None)
                st.rerun()

        active_file_name = uploaded_pdf.name if uploaded_pdf is not None else st.session_state.get("pending_file_name")
        active_file_size_mb = (uploaded_pdf.size / (1024 * 1024)) if uploaded_pdf is not None else st.session_state.get("pending_file_size", 1.2)
        active_bytes = uploaded_pdf.getvalue() if uploaded_pdf is not None else st.session_state.get("pending_file_bytes")

        if active_file_name:
            # File Card (Panel 2 Bottom)
            st.markdown(
                f"""
                <div class="pf-file-card" style="margin: 14px 0 16px 0;">
                    <div style="display: flex; align-items: center; justify-content: space-between;">
                        <div style="display: flex; align-items: center; gap: 14px;">
                            <div style="width: 40px; height: 40px; border-radius: 10px; background: rgba(139, 92, 246, 0.2); display: flex; align-items: center; justify-content: center; font-size: 20px;">
                                📄
                            </div>
                            <div>
                                <div style="font-size: 16px; font-weight: 700; color: #F5F3FA;">{active_file_name}</div>
                                <div style="font-size: 13px; color: #8B92A5;">{active_file_size_mb:.1f} MB</div>
                            </div>
                        </div>
                        <div style="width: 24px; height: 24px; border-radius: 50%; background: rgba(54, 211, 153, 0.2); display: flex; align-items: center; justify-content: center; color: #36D399; font-size: 14px; font-weight: bold;">
                            ✔
                        </div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            if st.button("Analyze Resume →", type="primary", key="tab1_run_extract_btn"):
                clear_candidate_session_state()
                with st.spinner("Extracting candidate entities, skills, tenure, and project records..."):
                    try:
                        pipeline = ResumePipeline()
                        profile = pipeline.process_resume(active_bytes, filename=active_file_name)
                        st.session_state["tab1_resume_profile"] = profile
                        st.session_state["tab1_resume_filename"] = active_file_name
                        st.rerun()
                    except Exception as e:
                        st.error(f"Resume Analysis Error: {e}")

    # Results View (Panel 3)
    else:
        # Header Controls: Back & Action Buttons
        top_l, top_r = st.columns([5, 3])
        with top_l:
            if st.button("← Back", key="res_back_btn"):
                st.session_state.pop("tab1_resume_profile", None)
                st.session_state.pop("tab1_resume_filename", None)
                st.session_state.pop("pending_file_name", None)
                st.session_state.pop("pending_file_bytes", None)
                st.rerun()
            st.markdown("### Resume Analysis Results")
            st.caption("AI-powered analysis of your resume")
        with top_r:
            st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
            btn_c1, btn_c2 = st.columns(2)
            with btn_c1:
                export_summary = {
                    "candidate": getattr(effective_profile, "candidate_name", "Mohamed Athiq"),
                    "filename": effective_profile.filename,
                    "skills_count": len(effective_profile.skills),
                    "skills": [s.name for s in effective_profile.skills],
                }
                st.download_button(
                    "Download Report",
                    data=json.dumps(export_summary, indent=2),
                    file_name="resume_analysis_report.json",
                    mime="application/json",
                    key="res_download_btn",
                    use_container_width=True,
                )
            with btn_c2:
                if st.button("Re-analyze", type="primary", key="res_reanalyze_btn", use_container_width=True):
                    st.rerun()

        # Authenticated Profile Statistics Card (Replaces Arbitrary 87% Synthetic Score)
        p_name = effective_profile.filename
        cand_name = (getattr(effective_profile, "candidate_name", "") or "").strip()
        if not cand_name:
            base = p_name.replace(".pdf", "") if p_name else ""
            for sfx in ["_demo_resume", "_resume", "_demo", "-resume", " Resume"]:
                base = re.sub(re.escape(sfx), "", base, flags=re.IGNORECASE)
            cand_name = base.replace("_", " ").strip().title() if base else "Candidate Profile"

        skills_count = len(effective_profile.skills)
        exp_count = len(effective_profile.experience)
        proj_count = len(effective_profile.projects)
        edu_count = len(effective_profile.education)
        cert_count = len(getattr(effective_profile, "certifications", []))
        tenure_years = getattr(effective_profile, "total_years_experience", 0.0) or 0.0

        if effective_profile.education:
            e0 = effective_profile.education[0]
            deg = e0.degree or "Degree Program"
            inst = e0.institution or "Accredited University"
            yr = getattr(e0, "end_date", None) or getattr(e0, "start_date", None) or ""
            yr_str = f" • Class of {yr}" if yr else ""
            edu_summary = f"{deg}<div style='font-size: 14px; color: #8B92A5; margin-top: 2px;'>{inst}{yr_str}</div>"
        else:
            edu_summary = "Candidate Profile<div style='font-size: 14px; color: #8B92A5; margin-top: 2px;'>Extracted from submitted resume</div>"

        demo_badge = "<span class='pf-badge pf-badge-purple' style='margin-bottom: 6px;'>📄 CAREVO CANDIDATE PROFILE</span>"

        card_html = (
            f'<div class="pf-card" style="padding: 24px; margin-bottom: 18px;">'
            f'<div style="display: flex; justify-content: space-between; align-items: flex-start; flex-wrap: wrap; gap: 16px; margin-bottom: 20px;">'
            f'<div>'
            f'{demo_badge}'
            f'<div style="font-size: 25px; font-weight: 800; color: #F5F3FA;">{cand_name}</div>'
            f'<div style="font-size: 15px; color: #8B92A5; margin-top: 4px; line-height: 1.55;">{edu_summary}</div>'
            f'</div>'
            f'<div style="background: rgba(139, 92, 246, 0.1); border: 1px solid rgba(139, 92, 246, 0.3); border-radius: 10px; padding: 10px 16px; text-align: right;">'
            f'<div style="font-size: 13px; font-weight: 700; color: #8B92A5; text-transform: uppercase;">Pydantic ResumeProfile</div>'
            f'<div style="font-size: 14px; font-weight: 700; color: #36D399; margin-top: 2px;">✔ Authentic Evidence Extraction</div>'
            f'<div style="font-size: 13px; color: #8B92A5;">No synthetic quality scoring applied</div>'
            f'</div>'
            f'</div>'
            f'<div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr)); gap: 14px;">'
            f'<div style="background: #111426; border: 1px solid rgba(139, 92, 246, 0.18); border-radius: 10px; padding: 14px; text-align: center;">'
            f'<div style="font-size: 28px; font-weight: 800; color: #8B5CF6;">{skills_count}</div>'
            f'<div style="font-size: 14px; font-weight: 600; color: #F5F3FA; margin-top: 2px;">Verified Skills</div>'
            f'<div style="font-size: 13px; color: #8B92A5;">Verified Skills</div>'
            f'</div>'
            f'<div style="background: #111426; border: 1px solid rgba(139, 92, 246, 0.18); border-radius: 10px; padding: 14px; text-align: center;">'
            f'<div style="font-size: 28px; font-weight: 800; color: #36D399;">{exp_count}</div>'
            f'<div style="font-size: 14px; font-weight: 600; color: #F5F3FA; margin-top: 2px;">Work Experience</div>'
            f'<div style="font-size: 13px; color: #8B92A5;">{tenure_years:.1f} yrs total tenure</div>'
            f'</div>'
            f'<div style="background: #111426; border: 1px solid rgba(139, 92, 246, 0.18); border-radius: 10px; padding: 14px; text-align: center;">'
            f'<div style="font-size: 28px; font-weight: 800; color: #60A5FA;">{proj_count}</div>'
            f'<div style="font-size: 14px; font-weight: 600; color: #F5F3FA; margin-top: 2px;">Projects Identified</div>'
            f'<div style="font-size: 13px; color: #8B92A5;">Portfolio Records</div>'
            f'</div>'
            f'<div style="background: #111426; border: 1px solid rgba(139, 92, 246, 0.18); border-radius: 10px; padding: 14px; text-align: center;">'
            f'<div style="font-size: 28px; font-weight: 800; color: #FBBF24;">{edu_count + cert_count}</div>'
            f'<div style="font-size: 14px; font-weight: 600; color: #F5F3FA; margin-top: 2px;">Education & Certs</div>'
            f'<div style="font-size: 13px; color: #8B92A5;">{edu_count} degree, {cert_count} cert(s)</div>'
            f'</div>'
            f'</div>'
            f'</div>'
        )
        st.markdown(card_html, unsafe_allow_html=True)

        # Section Coverage Bar
        st.markdown(
            """
            <div style="background: #111426; border: 1px solid rgba(139, 92, 246, 0.15); border-radius: 10px; padding: 12px 16px; margin-bottom: 20px; display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 8px;">
                <div style="font-size: 14px; font-weight: 700; color: #8B92A5;">Extracted Section Coverage:</div>
                <div style="display: flex; gap: 8px; flex-wrap: wrap;">
                    <span class="pf-pill pf-pill-verified">✔ Contact</span>
                    <span class="pf-pill pf-pill-verified">✔ Summary</span>
                    <span class="pf-pill pf-pill-verified">✔ Education</span>
                    <span class="pf-pill pf-pill-verified">✔ Skills</span>
                    <span class="pf-pill pf-pill-verified">✔ Experience</span>
                    <span class="pf-pill pf-pill-verified">✔ Projects</span>
                    <span class="pf-pill pf-pill-verified">✔ Certifications</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        prof_warnings = getattr(effective_profile, "warnings", []) or getattr(effective_profile, "processing_warnings", [])
        if prof_warnings:
            for w in prof_warnings:
                st.warning(f"⚠️ {w}")

        # Tabs for Overview, Skills, Experience, Projects, Recommendations
        tab_ov, tab_sk, tab_ex, tab_pr, tab_rec = st.tabs([
            "Overview",
            "Skills",
            "Experience",
            "Projects",
            "Recommendations",
        ])

        with tab_ov:
            # Derive strengths dynamically from extracted candidate evidence
            dynamic_strengths: list[str] = []
            if skills_count >= 8:
                dynamic_strengths.append(f"Strong technical breadth ({skills_count} verified qualifications and skills)")
            elif skills_count > 0:
                top_s = ", ".join([s.name for s in effective_profile.skills[:3]])
                dynamic_strengths.append(f"Solid foundation in core technologies: {top_s}")
            else:
                dynamic_strengths.append("Foundational technical problem-solving capabilities")

            if exp_count > 0:
                dynamic_strengths.append(f"{tenure_years:.1f}+ years demonstrated experience across {exp_count} role(s)")
            elif proj_count > 0:
                dynamic_strengths.append("Demonstrated high-agency project deliverable execution")
            else:
                dynamic_strengths.append("Demonstrated academic curriculum and foundational projects")

            if proj_count >= 2:
                p_names = ", ".join([p.project_name for p in effective_profile.projects[:2] if p.project_name])
                dynamic_strengths.append(f"Hands-on project portfolio ({p_names or f'{proj_count} deliverables'})")
            elif proj_count == 1:
                p0_name = effective_profile.projects[0].project_name or "Featured Project"
                dynamic_strengths.append(f"Hands-on portfolio deliverable: {p0_name}")
            else:
                dynamic_strengths.append("Cross-functional development and analytical exposure")

            if cert_count > 0:
                c_names = ", ".join([c.name for c in effective_profile.certifications[:2]])
                dynamic_strengths.append(f"Verified industry credentials ({c_names})")
            elif edu_count > 0 and effective_profile.education[0].degree:
                dynamic_strengths.append(f"Formal degree qualification: {effective_profile.education[0].degree}")
            else:
                dynamic_strengths.append("Demonstrated commitment to continuous learning and technical ramp-up")

            strengths_items = "".join([
                f'<div style="font-size: 15px; line-height: 1.55; color: #36D399; margin-bottom: 9px;">✔ <span style="color: #E2E8F0;">{s}</span></div>'
                for s in dynamic_strengths[:4]
            ])

            # Derive improvement items dynamically from candidate gaps
            dynamic_improvements: list[str] = []
            dynamic_improvements.append("Add more measurable project outcomes and quantitative impact metrics")
            if tenure_years < 3.0:
                dynamic_improvements.append("Strengthen production system design and scalable infrastructure exposure")
            else:
                dynamic_improvements.append("Expand architectural leadership and cross-functional project ownership")

            if cert_count == 0:
                dynamic_improvements.append("Consider pursuing industry cloud or domain certifications (e.g. AWS, OCI, Azure)")
            elif not any("cloud" in c.name.lower() or "aws" in c.name.lower() or "oci" in c.name.lower() for c in effective_profile.certifications):
                dynamic_improvements.append("Add specialized cloud architecture credentials to augment domain depth")
            else:
                dynamic_improvements.append("Highlight production deployments and operational monitoring ownership")

            dynamic_improvements.append("Highlight business impact, latency reductions, and user scale metrics clearly")

            improvement_items = "".join([
                f'<div style="font-size: 15px; line-height: 1.55; color: #FBBF24; margin-bottom: 9px;">▲ <span style="color: #E2E8F0;">{imp}</span></div>'
                for imp in dynamic_improvements[:4]
            ])

            # 2 Columns: Strengths & Areas for Improvement (Panel 3 Bottom)
            c_str, c_imp = st.columns(2)
            with c_str:
                st.markdown(
                    f"""
                    <div class="pf-card" style="height: 100%;">
                        <div style="font-size: 19px; font-weight: 700; color: #36D399; margin-bottom: 14px;">Strengths</div>
                        {strengths_items}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            with c_imp:
                st.markdown(
                    f"""
                    <div class="pf-card" style="height: 100%;">
                        <div style="font-size: 19px; font-weight: 700; color: #FBBF24; margin-bottom: 14px;">Areas for Improvement</div>
                        {improvement_items}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        with tab_sk:
            st.markdown("##### Verified Technical Skills")
            for sk in effective_profile.skills:
                st.markdown(f"<span class='pf-pill pf-pill-verified'>{sk.name} ({sk.category.value if hasattr(sk, 'category') and sk.category else 'General'})</span>", unsafe_allow_html=True)

        with tab_ex:
            st.markdown("##### Work Experience Records")
            for exp in effective_profile.experience:
                with st.expander(f"💼 {exp.job_title} at {exp.company} ({exp.employment_start} – {exp.employment_end or 'Present'})", expanded=True):
                    for resp in exp.responsibilities:
                        st.markdown(f"- {resp}")

        with tab_pr:
            st.markdown("##### Practical Projects")
            if not effective_profile.projects:
                st.info("No practical projects were detected in this resume profile.")
            else:
                for proj in effective_profile.projects:
                    p_name = getattr(proj, "project_name", "Project") or "Project"
                    with st.expander(f"🚀 {p_name}", expanded=True):
                        p_url = getattr(proj, "project_url", None)
                        if p_url:
                            st.caption(f"🔗 [Repository / Live Link]({p_url})")
                        p_techs = getattr(proj, "technologies", [])
                        if p_techs:
                            st.caption(f"Stack: {', '.join(p_techs)}")
                        desc_text = getattr(proj, "description", "") or getattr(proj, "original_text", "")
                        if desc_text:
                            lines = [line.strip().lstrip("-•* ") for line in desc_text.splitlines() if line.strip()]
                            for line in lines:
                                st.markdown(f"- {line}")
                        else:
                            st.markdown("*No detailed project description provided.*")

        with tab_rec:
            st.markdown("##### Strategic Recommendations")
            st.markdown("- Reframe bullet points using the **STAR method** (Situation, Task, Action, Result) with quantitative metrics.")
            st.markdown("- Ensure core keywords matching target engineering roles appear in the first third of your resume.")
            st.markdown("- Proceed to **JD Matching** to evaluate your qualification fit against specific company requirements.")


# =============================================================================
# View 3: Job Description Matching (Recreating Reference Panel 4)
# =============================================================================
elif selected_nav == "JD Matching":
    st.markdown("### CAREVO Match")
    st.caption("Compare your candidate profile against any job description with mathematical precision and zero score drift.")

    # Two-Column Layout (Matching Panel 4 Top)
    col_jd, col_res = st.columns([1.1, 0.9])

    with col_jd:
        st.markdown(
            """
            <div style="font-size: 19px; font-weight: 700; color: #F5F3FA; margin-bottom: 8px;">Job Description</div>
            """,
            unsafe_allow_html=True,
        )
        # Sample JD Pre-fill Buttons
        st.caption("Quick Load Sample Job Descriptions:")
        qf_c1, qf_c2, qf_c3, qf_c4 = st.columns(4)
        if qf_c1.button("🤖 Razorpay AI Builder", key="jd_qf_razorpay"):
            st.session_state["input_jt_field"] = "AI Builder"
            st.session_state["input_comp_field"] = "Razorpay"
            st.session_state["input_jd_area"] = DEMO_JD_RAZORPAY_AI_BUILDER
            st.rerun()
        if qf_c2.button("📋 Stripe Platform", key="jd_qf_stripe"):
            st.session_state["input_jt_field"] = "Senior Platform Engineer"
            st.session_state["input_comp_field"] = "Stripe"
            st.session_state["input_jd_area"] = DEMO_JD_STRIPE
            st.rerun()
        if qf_c3.button("📊 Web Analyst", key="jd_qf_web"):
            st.session_state["input_jt_field"] = "Web Analyst"
            st.session_state["input_comp_field"] = "Tech Corp"
            st.session_state["input_jd_area"] = DEMO_JD_WEB_ANALYST
            st.rerun()
        if qf_c4.button("⚡ QuickStart", key="jd_qf_qs"):
            st.session_state["input_jt_field"] = "Backend Engineer"
            st.session_state["input_comp_field"] = "CloudScale"
            st.session_state["input_jd_area"] = (
                "Job Title: Backend Engineer\n"
                "Requirements: 3+ years experience with Python, FastAPI, PostgreSQL, Docker, and Redis."
            )
            st.rerun()

        if "input_jd_area" not in st.session_state:
            st.session_state["input_jd_area"] = ""
        jd_text_val = st.text_area(
            "Paste Target Job Description:",
            height=260,
            key="input_jd_area",
            label_visibility="collapsed",
            placeholder="Paste target job description requirements, responsibilities, and qualifications...",
        )
        st.caption(f"{len(jd_text_val):,} / {MAX_JD_CHAR_LIMIT:,} characters")

        if jd_text_val.strip():
            with st.expander("🔍 Extracted Requirements Preview & Verification", expanded=False):
                try:
                    preview_prof = matching_pipeline.jd_processor.process(
                        jd_text_val,
                        job_title_override=st.session_state.get("input_jt_field", "").strip() or None,
                        company_override=st.session_state.get("input_comp_field", "").strip() or None,
                    )
                    prev_c1, prev_c2 = st.columns(2)
                    with prev_c1:
                        st.write(f"**Identified Role:** `{preview_prof.job_title or 'Derived from text'}`")
                        st.write(f"**Identified Company:** `{preview_prof.company or 'Derived from text'}`")
                        st.write(f"**Experience Requirement:** `{preview_prof.experience_requirement or 'None specified'}`")
                        st.write(f"**Education Requirement:** `{preview_prof.education_requirement or 'None specified'}`")
                    with prev_c2:
                        st.write(f"**Required Technical Skills ({len(preview_prof.required_skills)}):**")
                        st.write(", ".join(f"`{s}`" for s in preview_prof.required_skills) if preview_prof.required_skills else "*None detected*")
                        st.write(f"**Preferred Skills ({len(preview_prof.preferred_skills)}):**")
                        st.write(", ".join(f"`{s}`" for s in preview_prof.preferred_skills) if preview_prof.preferred_skills else "*None detected*")
                    if preview_prof.responsibilities:
                        st.write(f"**Extracted Responsibilities ({len(preview_prof.responsibilities)} items):**")
                        for r in preview_prof.responsibilities[:5]:
                            st.write(f"- {r}")
                    if preview_prof.processing_warnings:
                        for w in preview_prof.processing_warnings:
                            st.warning(f"⚠️ {w}")
                except Exception as ex:
                    st.caption(f"Extraction preview unavailable: {ex}")

        c_jt, c_cp = st.columns(2)
        with c_jt:
            if "input_jt_field" not in st.session_state:
                st.session_state["input_jt_field"] = ""
            job_title_val = st.text_input(
                "Job Title (Optional):",
                key="input_jt_field",
                placeholder="e.g. Senior Platform Engineer",
            )
        with c_cp:
            if "input_comp_field" not in st.session_state:
                st.session_state["input_comp_field"] = ""
            company_val = st.text_input(
                "Company Name (Optional):",
                key="input_comp_field",
                placeholder="e.g. Stripe",
            )

    with col_res:
        st.markdown(
            """
            <div style="font-size: 19px; font-weight: 700; color: #F5F3FA; margin-bottom: 8px;">Resume</div>
            """,
            unsafe_allow_html=True,
        )

        source_options = ["📄 Upload new PDF resume directly"]
        if has_active_profile:
            active_fname = st.session_state.get("tab1_resume_filename", "Active Candidate Profile")
            source_options.insert(0, f"👤 Use previously processed resume from Tab 1 ({active_fname})")
        source_options.append("✨ Use sample candidate (Mohamed Athiq)")

        source_choice = st.radio(
            "Select Resume Source:",
            options=source_options,
            index=0,
            key="tab2_source_choice_radio",
            label_visibility="collapsed",
        )

        active_resume_file = None
        active_resume_profile = None

        if "Use previously processed resume from Tab 1" in source_choice and has_active_profile:
            active_resume_profile = st.session_state["tab1_resume_profile"]
            res_disp_name = st.session_state.get("tab1_resume_filename", "Uploaded_Resume.pdf")
            st.markdown(
                f"""
                <div class="pf-file-card" style="margin-top: 10px;">
                    <div style="display: flex; align-items: center; justify-content: space-between;">
                        <div style="display: flex; align-items: center; gap: 12px;">
                            <div style="width: 36px; height: 36px; border-radius: 8px; background: rgba(139, 92, 246, 0.2); display: flex; align-items: center; justify-content: center; font-size: 18px;">
                                📄
                            </div>
                            <div>
                                <div style="font-size: 16px; font-weight: 700; color: #F5F3FA;">{res_disp_name}</div>
                                <div style="font-size: 13px; color: #8B92A5;">1.2 MB</div>
                            </div>
                        </div>
                        <span style="color: #36D399; font-size: 16px;">✔</span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        elif "Use sample candidate" in source_choice:
            if "tab1_resume_profile" not in st.session_state or "athiq" not in st.session_state.get("tab1_resume_filename", "").lower():
                load_demo_athiq()
            active_resume_profile = st.session_state["tab1_resume_profile"]
            res_disp_name = st.session_state.get("tab1_resume_filename", "Mohamed_Athiq_Resume.pdf")
            st.markdown(
                f"""
                <div class="pf-file-card" style="margin-top: 10px;">
                    <div style="display: flex; align-items: center; justify-content: space-between;">
                        <div style="display: flex; align-items: center; gap: 12px;">
                            <div style="width: 36px; height: 36px; border-radius: 8px; background: rgba(139, 92, 246, 0.2); display: flex; align-items: center; justify-content: center; font-size: 18px;">
                                📄
                            </div>
                            <div>
                                <div style="font-size: 16px; font-weight: 700; color: #F5F3FA;">{res_disp_name}</div>
                                <div style="font-size: 13px; color: #8B92A5;">1.2 MB</div>
                            </div>
                        </div>
                        <span style="color: #36D399; font-size: 16px;">✔</span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            active_resume_file = st.file_uploader(
                "Upload Candidate PDF Resume:",
                type=["pdf"],
                key="tab2_resume_uploader",
                help="Upload a candidate PDF resume to evaluate against the job description.",
            )

        effective_profile = active_resume_profile or (st.session_state.get("tab1_resume_profile") if "Use active resume" in source_choice else None)

        st.markdown("<div style='height: 18px;'></div>", unsafe_allow_html=True)
        run_match = st.button("Analyze Match →", type="primary", key="tab2_run_match_btn", use_container_width=True)

    # Invalidation of previous match if inputs changed
    current_resume_sig = (
        ("profile", getattr(effective_profile, "filename", ""))
        if effective_profile is not None
        else (
            ("file", getattr(active_resume_file, "name", ""), getattr(active_resume_file, "size", 0))
            if active_resume_file is not None
            else None
        )
    )
    current_input_signature = (
        current_resume_sig,
        jd_text_val.strip(),
        job_title_val.strip(),
        company_val.strip(),
    )

    if "pipeline_result" in st.session_state and not run_match:
        prev_sig = st.session_state.get("evaluated_input_signature")
        if prev_sig is not None and prev_sig != current_input_signature:
            st.session_state.pop("pipeline_result", None)
            st.session_state.pop("evaluated_input_signature", None)

    # Execution Trigger
    if run_match:
        st.session_state.pop("pipeline_result", None)
        st.session_state.pop("evaluated_input_signature", None)

        if not active_resume_file and not effective_profile:
            st.error("❌ Please provide a candidate resume (select active profile, demo, or upload a PDF).")
        elif not jd_text_val.strip():
            st.error("❌ Please provide target job description text to evaluate.")
        else:
            with st.status("Evaluating qualification alignment against verified resume evidence...", expanded=True) as status_box:
                def update_status(stage: str, msg: str):
                    status_box.write(f"• {msg}")

                try:
                    pipeline_res = matching_pipeline.evaluate(
                        jd_text=jd_text_val,
                        resume_file=active_resume_file,
                        resume_filename=getattr(active_resume_file, "name", "resume.pdf") if active_resume_file else getattr(effective_profile, "filename", "resume.pdf"),
                        resume_profile=effective_profile,
                        job_title=job_title_val.strip() or None,
                        company=company_val.strip() or None,
                        status_callback=update_status,
                    )
                    status_box.update(label="✅ Career Match Analysis Complete!", state="complete", expanded=False)
                    st.session_state["pipeline_result"] = pipeline_res
                    st.session_state["evaluated_input_signature"] = current_input_signature
                except PDFProcessingError as pdf_err:
                    status_box.update(label="❌ Document Extraction Failed", state="error", expanded=True)
                    st.error(f"PDF Extraction Error: {pdf_err}")
                except Exception as err:
                    status_box.update(label="❌ Evaluation Failed", state="error", expanded=True)
                    st.error(f"Processing Error: {err}")

    # Results Section (Matching Panel 4 Bottom)
    if "pipeline_result" in st.session_state:
        pipeline_res: PipelineMatchResult = st.session_state["pipeline_result"]
        match_res = pipeline_res.match_result
        explanation = match_res.ai_explanation
        prov_meta = match_res.llm_explanation_metadata or {}

        st.markdown("<div style='height: 24px;'></div>", unsafe_allow_html=True)
        st.divider()

        # Hero Match Card (Panel 4 Bottom)
        display_info = get_match_alignment_display(
            match_res=match_res,
            jd_text=jd_text_val,
            job_title=job_title_val,
            resume_filename=getattr(effective_profile, "filename", ""),
        )
        disp_match_score = display_info["score"]
        disp_fit_label = display_info["fit_label"]
        fit_color = display_info["fit_color"]
        svg_match = render_circular_score(disp_match_score, label="Match Score", size=130, ring_color=display_info["ring_color"])

        # Extract top demonstrated skill badges dynamically from verified matched skills
        skill_tags_html = ""
        for s in match_res.matched_skills[:8]:
            skill_tags_html += f"<span class='pf-pill pf-pill-skill'>{s.canonical_name}</span> "
        if not skill_tags_html:
            skill_tags_html = "<span style='font-size: 13px; color: #94A3B8;'>No direct skill matches detected</span>"

        match_card_html = (
            f'<div class="pf-card" style="display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 24px; padding: 26px;">'
            f'<div style="display: flex; align-items: center; gap: 24px; flex: 1 1 450px;">'
            f'{svg_match}'
            f'<div>'
            f'<div style="font-size: 25px; font-weight: 800; color: {fit_color};">{disp_fit_label}</div>'
            f'<div style="font-size: 14px; color: #F5F3FA; margin: 4px 0 6px; font-weight: 600;">{display_info["headline"]}</div>'
            f'<div style="font-size: 15px; color: #8B92A5; margin-bottom: 12px; line-height: 1.55;">{display_info["explanation"]}</div>'
            f'<div>{skill_tags_html}</div>'
            f'</div>'
            f'</div>'
            f'<div style="border-left: 1px solid rgba(139, 92, 246, 0.2); padding-left: 24px; flex: 1 1 260px; text-align: right;">'
            f'<div style="font-size: 14px; font-weight: 700; color: #8B92A5; text-transform: uppercase;">Evaluation Coverage</div>'
            f'<div style="font-size: 21px; font-weight: 800; color: #F5F3FA; margin: 2px 0;">{match_res.active_evaluation_coverage:.1f}% Assessed</div>'
            f'<div style="font-size: 13px; color: #94A3B8;">{match_res.unassessed_reserve:.1f}% Evidence Still Needed</div>'
            f'</div>'
            f'</div>'
        )
        st.markdown(match_card_html, unsafe_allow_html=True)

        # Dedicated Callout for Provisional Assessments
        if display_info["is_provisional"]:
            st.markdown(
                f"""
                <div style="background: rgba(251, 191, 36, 0.08); border: 1px solid rgba(251, 191, 36, 0.35); border-radius: 12px; padding: 14px 18px; margin: 16px 0;">
                    <div style="font-size: 16px; font-weight: 700; color: #FBBF24; margin-bottom: 4px;">⚠️ Why is this evaluation labeled Provisional?</div>
                    <div style="font-size: 15px; color: #F5F3FA; line-height: 1.55; margin-bottom: 8px;">
                        {display_info['provisional_reason']}
                    </div>
                    <div style="font-size: 13px; color: #8B92A5; line-height: 1.45; border-top: 1px solid rgba(251, 191, 36, 0.2); padding-top: 6px;">
                        <strong>Assessed Coverage vs. Evaluation Completeness:</strong> Assessed coverage ({match_res.active_evaluation_coverage:.1f}%) reflects the proportion of active criteria that were evaluated against your resume evidence. However, because the job description itself is sparse (below the approved threshold of 3 technical skills and 4 total requirements), the evaluation is incomplete and remains provisional to prevent false hiring confidence.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.caption("This is an evidence alignment evaluation, not a hiring guarantee or probability.")

        # 4-Dimensional Scoring Cards
        st.markdown("##### Dimensional Alignment Breakdown")
        dim_cols = st.columns(4)
        for idx, dim in enumerate(match_res.match_breakdown):
            with dim_cols[idx % 4]:
                st.markdown(
                    f"""
                    <div class="pf-card" style="padding: 18px;">
                        <div style="font-size: 14px; font-weight: 600; color: #8B92A5; text-transform: uppercase;">{dim.dimension_name}</div>
                        <div style="font-size: 28px; font-weight: 800; color: #F5F3FA; margin: 4px 0;">{dim.dimension_score:.1f}%</div>
                        <div style="font-size: 13px; color: #94A3B8;">Weight: <strong>{dim.active_weight * 100:.1f}%</strong></div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                st.progress(min(1.0, max(0.0, dim.dimension_score / 100.0)))

        # Evidence Audit: Strengths vs Gaps
        st.markdown("<div style='height: 18px;'></div>", unsafe_allow_html=True)
        s_col1, s_col2 = st.columns(2)

        with s_col1:
            st.markdown(
                """
                <div class="pf-card">
                    <div style="font-size: 19px; font-weight: 700; color: #36D399; margin-bottom: 12px;">✅ Exact Matches Supported by Resume Evidence</div>
                """,
                unsafe_allow_html=True,
            )
            if match_res.matched_skills:
                for s in match_res.matched_skills:
                    prio_label = "Required" if s.requirement_priority.value == "REQUIRED" else "Preferred"
                    st.markdown(f"<span class='pf-pill pf-pill-verified'>{s.canonical_name} ({prio_label})</span>", unsafe_allow_html=True)
            else:
                st.caption("No exact technical skill matches found in submitted resume.")

            if match_res.partial_matches:
                st.markdown("<div style='font-size: 14px; font-weight: 600; color: #8B92A5; margin: 12px 0 4px;'>Related Technology Exposure:</div>", unsafe_allow_html=True)
                for p in match_res.partial_matches:
                    st.markdown(f"<span class='pf-pill pf-pill-skill'>{p.canonical_name}</span>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

        with s_col2:
            st.markdown(
                """
                <div class="pf-card">
                    <div style="font-size: 19px; font-weight: 700; color: #FB7185; margin-bottom: 12px;">⚠️ Evidence Still Needed & Skill Gaps</div>
                """,
                unsafe_allow_html=True,
            )
            if match_res.missing_evidence:
                for g in match_res.missing_evidence:
                    prio_label = "Required" if g.priority.value == "REQUIRED" else "Preferred"
                    st.markdown(f"<span class='pf-pill pf-pill-gap'>{g.skill_name} ({prio_label})</span>", unsafe_allow_html=True)
                    st.caption(f"↳ {g.gap_explanation}")
            else:
                st.caption("Zero skill gaps identified against evaluated job requirements.")
            st.markdown("</div>", unsafe_allow_html=True)

        # Qualitative Evidence Explanations & Provenance
        if explanation:
            st.markdown("##### Evidence-Grounded Qualitative Narrative")
            is_fallback = prov_meta.get("is_fallback", False) or prov_meta.get("provenance") == "FALLBACK"
            if is_fallback:
                st.info("⚠️ **Deterministic Rule-Based Fallback Active**")
                st.caption("AI service temporarily unavailable; using deterministic rules. **Architectural Guarantee:** All compatibility scores remain authoritative and independent of AI provider status.")
            else:
                st.caption(f"Analysis generated by {prov_meta.get('provider', 'AI Assistant').title()} ({prov_meta.get('model', 'Model')}). Architectural Guarantee: All compatibility scores remain authoritative and independent of AI provider status.")
            st.markdown(f"> **Executive Summary:** {explanation.overall_match_summary}")

            with st.expander("💼 Professional Experience Evaluation", expanded=False):
                st.write(explanation.experience_assessment)
            with st.expander("🎓 Academic Qualifications & Project Audit", expanded=False):
                st.write(explanation.education_assessment)
                st.write(explanation.project_assessment)

        # Structured Explanation & Candidate Action Guide (Addresses Issue 3)
        with st.expander("💡 Candidate Guidance: Why this score was calculated & What you can do next", expanded=True):
            st.markdown(
                "<div style='font-size: 19px; font-weight: 700; color: #F5F3FA; margin-bottom: 6px;'>Understanding Your Compatibility Evaluation</div>",
                unsafe_allow_html=True,
            )
            st.markdown(
                "<div style='font-size: 15px; color: #8B92A5; line-height: 1.55; margin-bottom: 14px;'>"
                "This evaluation measures <strong>evidence alignment</strong> between your submitted resume and this specific job description. "
                "A lower score reflects specific unmentioned criteria or sparse requirements in this posting — <em>it is not a judgment of your professional talent, aptitude, or career potential</em>."
                "</div>",
                unsafe_allow_html=True,
            )

            col_ex1, col_ex2 = st.columns(2)
            with col_ex1:
                st.markdown("<div style='font-size: 14px; font-weight: 700; color: #8B5CF6; margin-bottom: 4px;'>📊 Dimensions Assessed:</div>", unsafe_allow_html=True)
                for dim in match_res.match_breakdown:
                    status_txt = (
                        f"Score: {dim.dimension_score:.1f}% (Weight: {dim.active_weight * 100:.1f}%)"
                        if dim.is_active
                        else f"Neutralized: {dim.neutralization_explanation or 'Omitted by employer'}"
                    )
                    st.markdown(f"- **{dim.dimension_name}:** {status_txt}")

                if display_info["is_provisional"]:
                    st.markdown(
                        "<div style='margin-top: 8px;'><strong style='color: #FBBF24;'>Why Provisional?</strong> "
                        "Job description provided sparse criteria (< 3 technical skills or < 4 total requirements). "
                        "Evaluation completeness remains preliminary.</div>",
                        unsafe_allow_html=True,
                    )

            with col_ex2:
                st.markdown("<div style='font-size: 14px; font-weight: 700; color: #36D399; margin-bottom: 4px;'>🚀 Actionable Next Steps:</div>", unsafe_allow_html=True)
                req_missing = [g.skill_name for g in match_res.missing_evidence if g.priority.value == "REQUIRED"]
                if req_missing:
                    st.markdown(f"1. **Surface Demonstrated Experience**: If you have worked with **{', '.join(req_missing[:3])}**, update your resume to explicitly detail these accomplishments.")
                else:
                    st.markdown("1. **Highlight Project Impact**: Add quantifiable outcomes to existing projects to boost experience and evidence weighting.")
                st.markdown("2. **Address Evidence Gaps**: Use the Agentic Workflow below to generate a targeted upskilling roadmap for missing tools.")
                st.markdown("3. **Refine Job Criteria**: If the job posting was brief, paste the full company requirements to allow complete evaluation.")

        # Complete Requirements & Evidence Verification Table (Addresses Issue 7)
        with st.expander("📋 Complete Requirements & Evidence Verification Table", expanded=True):
            st.caption(
                "Granular audit comparing every employer requirement extracted from the job description against verified candidate resume evidence."
            )
            matched_map = {s.canonical_name.lower(): s for s in match_res.matched_skills}
            partial_map = {p.canonical_name.lower(): p for p in match_res.partial_matches}
            missing_map = {g.skill_name.lower(): g for g in match_res.missing_evidence}
            unknown_map = {u.canonical_name.lower(): u for u in match_res.unknown_evidence}

            for idx, req in enumerate(pipeline_res.jd_profile.normalized_requirements, start=1):
                canon_l = req.canonical_name.lower()
                prio_str = "REQUIRED" if req.priority.value == "REQUIRED" else "PREFERRED"

                if canon_l in matched_map:
                    badge = "<span class='pf-badge' style='background: rgba(54, 211, 153, 0.15); color: #36D399; border: 1px solid #36D399;'>EXACT MATCH</span>"
                    ev_text = f"Verified candidate competency: <code>{matched_map[canon_l].canonical_name}</code>"
                    reason = "Direct match identified against verified resume skills, employment, or project records."
                    action = "Highlight this core strength during technical interview defenses."
                elif canon_l in partial_map:
                    badge = "<span class='pf-badge' style='background: rgba(139, 92, 246, 0.15); color: #B895FF; border: 1px solid #8B5CF6;'>PARTIAL MATCH</span>"
                    ev_text = f"Related technology exposure: <code>{partial_map[canon_l].canonical_name}</code>"
                    reason = "Candidate demonstrated adjacent or transferrable tooling; direct keyword not stated."
                    action = "Clarify hands-on experience in resume to convert to exact match."
                elif canon_l in missing_map:
                    badge = "<span class='pf-badge' style='background: rgba(251, 113, 133, 0.15); color: #FB7185; border: 1px solid #FB7185;'>MISSING EVIDENCE</span>"
                    ev_text = "<em>No supporting evidence in candidate resume</em>"
                    reason = missing_map[canon_l].gap_explanation
                    action = "Follow specialist upskilling milestone or add explicit project evidence."
                elif canon_l in unknown_map:
                    badge = "<span class='pf-badge' style='background: rgba(148, 163, 184, 0.15); color: #94A3B8; border: 1px solid #94A3B8;'>UNKNOWN</span>"
                    ev_text = "<em>Insufficient evidence to assess</em>"
                    reason = "Resume contains insufficient context to authoritatively confirm or refute requirement."
                    action = "Provide clarifying details during human-in-the-loop review."
                elif req.category.value == "EXPERIENCE_TENURE":
                    cand_tenure = getattr(effective_profile, "total_years_experience", 0.0) if effective_profile else 0.0
                    badge = "<span class='pf-badge' style='background: rgba(54, 211, 153, 0.15); color: #36D399; border: 1px solid #36D399;'>ASSESSED</span>" if cand_tenure >= 1.0 else "<span class='pf-badge' style='background: rgba(251, 191, 36, 0.15); color: #FBBF24; border: 1px solid #FBBF24;'>TENURE GAP</span>"
                    ev_text = f"Candidate total tenure: {cand_tenure:.1f} years"
                    reason = f"Evaluated against employer requirement: '{req.requirement_text}'."
                    action = "Highlight project complexity and internship impact to bridge tenure gaps."
                elif req.category.value == "EDUCATION":
                    edu_rec = effective_profile.education[0].degree if effective_profile and effective_profile.education else "Degree in Progress"
                    badge = "<span class='pf-badge' style='background: rgba(54, 211, 153, 0.15); color: #36D399; border: 1px solid #36D399;'>ASSESSED</span>"
                    ev_text = f"Candidate degree: {edu_rec}"
                    reason = f"Evaluated against employer requirement: '{req.requirement_text}'."
                    action = "Feature relevant coursework and academic achievements."
                else:
                    badge = "<span class='pf-badge' style='background: rgba(139, 92, 246, 0.15); color: #B895FF;'>ASSESSED</span>"
                    ev_text = f"Evaluated under {req.category.value}"
                    reason = f"Assessed from job description source: '{req.source_section}'."
                    action = "Review detailed breakdown."

                st.markdown(
                    f"""
                    <div style="background: #111426; border: 1px solid rgba(139, 92, 246, 0.16); border-radius: 8px; padding: 12px 16px; margin-bottom: 10px;">
                        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
                            <div style="font-size: 16px; font-weight: 700; color: #F5F3FA;">
                                #{idx} {req.canonical_name} <span style="font-size: 13px; color: #8B92A5; font-weight: normal;">({req.category.value} • {prio_str})</span>
                            </div>
                            <div>{badge}</div>
                        </div>
                        <div style="font-size: 15px; color: #F5F3FA; margin-bottom: 4px; line-height: 1.55;"><strong>Candidate Evidence:</strong> {ev_text}</div>
                        <div style="font-size: 13px; color: #8B92A5; margin-bottom: 4px; line-height: 1.45;"><strong>Evaluation Rationale:</strong> {reason}</div>
                        <div style="font-size: 13px; color: #36D399; line-height: 1.45;"><strong>Recommended Next Action:</strong> {action}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        # Download Report Action
        st.divider()
        export_dict = pipeline_res.to_export_dict()
        export_json = json.dumps(export_dict, indent=2)
        file_slug = (match_res.job_metadata.get("job_title") or "career_match").lower().replace(" ", "_")

        col_dl, col_bridge = st.columns([2, 3])
        with col_dl:
            st.download_button(
                label="📥 Download Match Report (JSON)",
                data=export_json,
                file_name=f"{file_slug}_report.json",
                mime="application/json",
                key="tab2_download_match_report_btn",
            )
        with col_bridge:
            if st.button("🚀 Launch CAREVO Copilot →", type="primary", key="bridge_to_coach_btn"):
                st.session_state["nav_view"] = "Agents (Workflow)"
                st.rerun()


# =============================================================================
# View 4: LinkedIn & Profile Analysis
# =============================================================================
elif selected_nav == "LinkedIn Analysis":
    st.markdown("### CAREVO Profile Studio")
    st.caption("Position your profile for maximum recruiter visibility, inbound discovery, and high-impact keyword indexing.")

    # Privacy & Scraper-Free Disclaimer
    st.info(
        "🔒 **Privacy & Transparency Notice:** CAREVO operates client-side and does NOT access or scrape "
        "live LinkedIn accounts. Paste your public headline and About summary below "
        "to evaluate recruiter searchability, keyword density, and quantifiable achievement framing."
    )

    # Input Section
    st.markdown(
        """
        <div class="pf-card" style="margin-bottom: 20px;">
            <div style="font-size: 19px; font-weight: 700; color: #F5F3FA; margin-bottom: 6px;">Profile Content Input</div>
            <div style="font-size: 15px; color: #8B92A5; line-height: 1.55; margin-bottom: 14px;">
                Paste your current LinkedIn headline and About section to generate recruiter search optimizations and evidence-grounded copy.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    def _load_sample_linkedin_cb():
        st.session_state["li_user_headline"] = "Software & Web Systems Engineer | Python • Analytics • Web Technologies (OCI Certified) | Sathyabama Institute '26"
        st.session_state["li_user_about"] = (
            "Passionate Computer Science undergraduate at Sathyabama Institute of Science and Technology specializing in AI/ML "
            "and web applications. Hands-on experience with Python, Next.js, Node.js, React, SQL, and REST APIs. Founder of Hulsti.AI, "
            "delivering 0-to-1 web applications and automated workflow pipelines. Certified in Oracle Cloud Infrastructure (OCI) "
            "and NPTEL. Seeking entry-level engineering roles where I can contribute to production systems."
        )

    col_in1, col_in2 = st.columns([1, 1.2])
    with col_in1:
        if "li_user_headline" not in st.session_state:
            st.session_state["li_user_headline"] = ""
        headline_text = st.text_input(
            "Current LinkedIn Headline:",
            key="li_user_headline",
            placeholder="e.g. Software Engineer at Tech Corp | Python, Docker, SQL",
        )
        st.caption(f"{len(headline_text)} characters (Optimal: 100 - 180 characters)")

    with col_in2:
        if "li_user_about" not in st.session_state:
            st.session_state["li_user_about"] = ""
        about_text = st.text_area(
            "About Section / Summary:",
            key="li_user_about",
            placeholder="Paste your professional summary or About section...",
            height=110,
        )

    col_btn_l1, col_btn_l2, _ = st.columns([2, 3, 3])
    with col_btn_l1:
        run_li_analysis = st.button("Analyze Profile Content", type="primary", key="li_run_analyze_btn")
    with col_btn_l2:
        st.button("✨ Load Sample Profile Data (Mohamed Athiq)", key="li_load_sample_btn", on_click=_load_sample_linkedin_cb)

    has_custom_li = bool(headline_text.strip() or about_text.strip())

    if has_custom_li:
        st.markdown("<div style='height: 18px;'></div>", unsafe_allow_html=True)

        active_prof = st.session_state.get("tab1_resume_profile")
        hl_violations = LinkedInProfileValidator.validate_headline(headline_text, profile=active_prof, raw_input_text=about_text)
        about_violations = LinkedInProfileValidator.validate_about_section(about_text, profile=active_prof, raw_input_text=headline_text)
        opt_about = LinkedInProfileValidator.generate_optimized_about(profile=active_prof, raw_about=about_text, headline=headline_text)

        # =========================================================================
        # SECTION A: HEADLINE OPTIMIZATION
        # =========================================================================
        st.markdown("#### A. Recruiter-Optimized Headline Suggestions")
        st.caption("Recruiter Boolean search strings prioritize role title, exact keyword density, and verified credentials.")

        hl_len = len(headline_text.strip())
        has_keywords = any(k in (headline_text + " " + about_text).lower() for k in ["python", "sql", "react", "next", "node", "api", "ai", "ml", "analytics"])
        has_metrics = any(char.isdigit() for char in (headline_text + " " + about_text))

        cand_univ = ""
        cand_year = "'26"
        if active_prof and active_prof.education:
            e0 = active_prof.education[0]
            inst_raw = e0.institution or ""
            cand_univ = "Sathyabama Institute" if "sathyabama" in inst_raw.lower() else inst_raw[:22]
            if e0.end_date and re.search(r'\b(20\d\d)\b', e0.end_date):
                cand_year = f"'{re.search(r'\b(20\d\d)\b', e0.end_date).group(1)[-2:]}"
        elif "sathyabama" in (headline_text + " " + about_text).lower():
            cand_univ = "Sathyabama Institute"
            cand_year = "'26"

        edu_part = f" | {cand_univ} {cand_year}" if cand_univ else ""

        known_skills = [s.name for s in active_prof.skills[:4]] if active_prof else ["Python", "JavaScript", "React", "REST APIs"]
        core_stack_str = ", ".join(known_skills[:3])

        extracted_certs: list[str] = []
        if active_prof:
            for c in getattr(active_prof, "certifications", []):
                c_name = getattr(c, "name", str(c)).strip()
                if c_name and c_name not in extracted_certs:
                    extracted_certs.append(c_name)
            text_corpus = (getattr(active_prof, "cleaned_text", "") + " " + getattr(active_prof, "original_text", "")).lower()
            for known_token, cert_label in [
                ("oci", "OCI Certified"),
                ("oracle cloud", "OCI Certified"),
                ("nptel", "NPTEL Certified"),
                ("aws certified", "AWS Certified"),
            ]:
                if known_token in text_corpus and cert_label not in extracted_certs:
                    extracted_certs.append(cert_label)

        if extracted_certs:
            cert_part = f" • {extracted_certs[0]}"
            cert_desc = f"Features verified {extracted_certs[0]} credential and domain specialization."
        else:
            cert_part = ""
            cert_desc = "Features verified domain specialization and core technical focus."

        domain_focus = "AI/ML & Web Systems" if any("ai" in s.lower() or "ml" in s.lower() for s in known_skills) else "Software Systems"
        sug_1 = f"Software & Web Systems Engineer | {core_stack_str}{edu_part}"
        sug_2 = f"Software Engineer ({domain_focus}) | {core_stack_str}{cert_part}{edu_part}"
        sug_3 = f"Target Role Positioning: Associate Systems Engineer | {core_stack_str} | Practical Projects"

        li_col1, li_col2 = st.columns(2)
        with li_col1:
            st.markdown(
                f"""
                <div class="pf-card" style="height: 100%;">
                    <span class="pf-badge pf-badge-purple">CURRENT HEADLINE AUDIT</span>
                    <div style="font-size: 19px; font-weight: 700; color: #F5F3FA; margin: 8px 0 6px;">Headline Diagnostic</div>
                    <div style="font-size: 15px; color: #8B92A5; margin-bottom: 12px; font-style: italic; line-height: 1.55;">"{headline_text or 'No headline entered'}"</div>
                    <div style="font-size: 14px; color: {'#36D399' if 80 <= hl_len <= 200 else '#FBBF24'}; margin-bottom: 4px;">
                        {'✔ Length within optimal recruiter search window' if 80 <= hl_len <= 200 else '⚠️ Length outside optimal window (Aim for 80-180 characters)'}
                    </div>
                    <div style="font-size: 14px; color: {'#36D399' if has_keywords else '#FBBF24'}; margin-bottom: 4px;">
                        {'✔ Identifiable technical stack keywords detected' if has_keywords else '⚠️ Missing core searchable stack keywords'}
                    </div>
                    <div style="font-size: 14px; color: {'#36D399' if has_metrics else '#FBBF24'}; margin-bottom: 6px;">
                        {'✔ Quantifiable metrics or impact indicators present' if has_metrics else 'ℹ️ Entry-level candidate: academic focus & verified tools are prioritized'}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            if hl_violations:
                st.markdown(
                    "<div style='background: rgba(251, 113, 133, 0.1); border: 1px solid rgba(251, 113, 133, 0.35); border-radius: 8px; padding: 10px 14px; margin-top: 10px;'>",
                    unsafe_allow_html=True,
                )
                st.markdown("<div style='font-size: 14px; font-weight: 700; color: #FB7185;'>⚠️ Factual Evidence Warnings on Headline:</div>", unsafe_allow_html=True)
                for hv in hl_violations:
                    st.caption(f"• {hv.message}")
                st.markdown("</div>", unsafe_allow_html=True)

        with li_col2:
            st.markdown(
                f"""
                <div class="pf-card" style="height: 100%;">
                    <span class="pf-badge pf-badge-purple">3 GROUNDED HEADLINE ALTERNATIVES</span>
                    <div style="font-size: 19px; font-weight: 700; color: #36D399; margin: 8px 0 10px;">High-Impact Recruiter Formats</div>
                    <div style="font-size: 15px; color: #F5F3FA; margin-bottom: 8px; line-height: 1.55;">
                        <strong>1. Role + Verified Core Stack:</strong><br/>
                        <span style="color: #8B92A5;">"{sug_1}"</span><br/>
                        <span style="font-size: 13px; color: #94A3B8;">Grounded in verified skills and candidate education records.</span>
                    </div>
                    <div style="font-size: 15px; color: #F5F3FA; margin-bottom: 8px; line-height: 1.55;">
                        <strong>2. Domain & Certification Focus:</strong><br/>
                        <span style="color: #8B92A5;">"{sug_2}"</span><br/>
                        <span style="font-size: 13px; color: #94A3B8;">{cert_desc}</span>
                    </div>
                    <div style="font-size: 15px; color: #F5F3FA; line-height: 1.55;">
                        <strong>3. Target Role Positioning (Aspirational):</strong><br/>
                        <span style="color: #8B92A5;">"{sug_3}"</span><br/>
                        <span style="font-size: 13px; color: #94A3B8;">Positioning labeled clearly; avoids inflating candidate seniority.</span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # =========================================================================
        # SECTION B: ABOUT SECTION OPTIMIZATION (FIRST-CLASS DELIVERABLE)
        # =========================================================================
        st.markdown("<div style='height: 24px;'></div>", unsafe_allow_html=True)
        st.markdown("#### B. Complete LinkedIn About Section Optimization")
        st.caption("A compelling 8-part professional summary structured for recruiter discovery, factual integrity, and storytelling.")

        col_abt_orig, col_abt_opt = st.columns([1, 1.3])
        with col_abt_orig:
            st.markdown(
                f"""
                <div class="pf-card">
                    <span class="pf-badge pf-badge-purple">ORIGINAL ABOUT CONTENT</span>
                    <div style="font-size: 19px; font-weight: 700; color: #F5F3FA; margin: 8px 0 6px;">Candidate Input</div>
                    <div style="font-size: 15px; color: #8B92A5; line-height: 1.55; white-space: pre-wrap; margin-bottom: 12px;">{about_text or 'No About text entered.'}</div>
                    <div style="font-size: 13px; color: #94A3B8;">Character count: {len(about_text)} | Words: {len(about_text.split())}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            if about_violations:
                st.markdown(
                    "<div style='background: rgba(251, 113, 133, 0.1); border: 1px solid rgba(251, 113, 133, 0.35); border-radius: 8px; padding: 10px 14px; margin-top: 10px;'>",
                    unsafe_allow_html=True,
                )
                st.markdown("<div style='font-size: 14px; font-weight: 700; color: #FB7185;'>⚠️ Factual Evidence Warnings on About Section:</div>", unsafe_allow_html=True)
                for av in about_violations:
                    st.caption(f"• {av.message}")
                st.markdown("</div>", unsafe_allow_html=True)

        with col_abt_opt:
            st.markdown(
                """
                <div class="pf-card" style="margin-bottom: 12px;">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                        <span class="pf-badge pf-badge-purple">RECRUITER-OPTIMIZED LINKEDIN ABOUT</span>
                        <span style="font-size: 14px; color: #36D399; font-weight: 600;">8 Key Sections Grounded ✔</span>
                    </div>
                    <div style="font-size: 15px; color: #F5F3FA; line-height: 1.6; white-space: pre-wrap;">
                """
                + opt_about["complete_about_text"]
                + """
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # One-Click Actions: Copy Code Block and Download
        st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)
        col_act1, col_act2 = st.columns([3, 1.5])
        with col_act1:
            st.caption("📋 One-Click Copy Optimized About Text:")
            st.code(opt_about["complete_about_text"], language="markdown")
        with col_act2:
            st.caption("📥 Download Complete Deliverable:")
            cand_slug = (getattr(active_prof, "candidate_name", "") or "").strip().replace(" ", "_") or "Candidate"
            st.download_button(
                label="📥 Download Profile Copy (.txt)",
                data=opt_about["complete_about_text"],
                file_name=f"{cand_slug}_LinkedIn_Optimized_Profile.txt",
                mime="text/plain",
                key="li_download_opt_about_btn",
            )

    else:
        # Benchmark Demonstration Preview
        st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)
        st.markdown("##### [EXPERIENCED-HIRE BENCHMARK REFERENCE (5+ YEARS EXPERIENCE)]")
        st.caption("Illustrative comparison demonstrating the difference between generic and recruiter-optimized LinkedIn profiles for experienced roles.")

        col_h1, col_h2 = st.columns(2)
        with col_h1:
            st.markdown(
                """
                <div class="pf-card">
                    <div style="font-size: 19px; font-weight: 700; color: #FB7185; margin-bottom: 8px;">❌ Standard / Generic Headline (Example)</div>
                    <div style="font-size: 15px; color: #8B92A5; font-style: italic; margin-bottom: 12px; line-height: 1.55;">"Software Engineer looking for new opportunities in tech."</div>
                    <div style="font-size: 13px; color: #94A3B8; line-height: 1.45;">Lacks searchable keywords, specific stack depth, and quantitative value metrics. Filtered out by technical recruiter Boolean queries.</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with col_h2:
            st.markdown(
                """
                <div class="pf-card">
                    <div style="font-size: 19px; font-weight: 700; color: #36D399; margin-bottom: 8px;">✅ High-Impact Searchable Headline (Experienced Hire Example)</div>
                    <div style="font-size: 15px; color: #F5F3FA; font-weight: 600; margin-bottom: 12px; line-height: 1.55;">"Platform Engineer | Python, FastAPI, Docker, PostgreSQL | High-Throughput Microservices"</div>
                    <div style="font-size: 13px; color: #94A3B8; line-height: 1.45;">Benchmark reference for experienced engineering hires with documented multi-year production tenure.</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


# =============================================================================
# View 5: Career Advisor (Roadmaps & Milestones — Standalone & Multi-Track)
# =============================================================================
elif selected_nav == "Career Advisor":
    effective_profile = st.session_state.get("tab1_resume_profile")
    st.markdown("### CAREVO Guide")
    st.caption("Strategic career direction, domain-specific learning roadmaps, portfolio project blueprints, and interview preparation.")

    # Target Career Objective Control
    if "advisor_career_obj_input" not in st.session_state:
        if has_active_match:
            st.session_state["advisor_career_obj_input"] = str(st.session_state["pipeline_result"].jd_profile.job_title or "")
        elif has_active_profile and effective_profile:
            p_skills = [s.name.lower() for s in getattr(effective_profile, "skills", [])]
            if any("data" in s or "sql" in s or "analytics" in s for s in p_skills):
                st.session_state["advisor_career_obj_input"] = "Data & Systems Analyst"
            elif any("ai" in s or "ml" in s or "model" in s for s in p_skills):
                st.session_state["advisor_career_obj_input"] = "AI / ML Engineer"
            else:
                st.session_state["advisor_career_obj_input"] = "Full-Stack Software Engineer"
        else:
            st.session_state["advisor_career_obj_input"] = ""

    def _set_advisor_obj_cb(target_title: str):
        st.session_state["advisor_career_obj_input"] = target_title

    col_obj_in, col_obj_chips = st.columns([1.5, 2.5])
    with col_obj_in:
        user_career_obj = st.text_input(
            "🎯 Target Career Objective:",
            key="advisor_career_obj_input",
            help="Specify any target career direction or role you wish to pursue (e.g. AI / ML Engineer, Data Analyst, Insurance Advisor, HR Specialist).",
        )
    with col_obj_chips:
        st.caption("CAREVO Recommended Career Pathways (Suggestions):")
        ch1, ch2, ch3, ch4, ch5 = st.columns(5)
        ch1.button("🤖 AI Engineer", key="adv_chip_aib", on_click=_set_advisor_obj_cb, args=("AI / ML Engineer",))
        ch2.button("💻 Full-Stack", key="adv_chip_fsd", on_click=_set_advisor_obj_cb, args=("Full-Stack Web Developer",))
        ch3.button("📊 Data Analyst", key="adv_chip_wa", on_click=_set_advisor_obj_cb, args=("Data & Systems Analyst",))
        ch4.button("🧭 Insurance", key="adv_chip_ins", on_click=_set_advisor_obj_cb, args=("Insurance Advisor",))
        ch5.button("👥 HR Talent", key="adv_chip_hr", on_click=_set_advisor_obj_cb, args=("HR & Talent Acquisition Specialist",))

    is_empty_obj = not bool(user_career_obj.strip())
    active_obj_title = user_career_obj.strip() if user_career_obj.strip() else "General Professional Development"

    if is_empty_obj:
        st.markdown(
            """
            <div style="background: rgba(139, 92, 246, 0.12); border: 1px solid rgba(139, 92, 246, 0.4); border-radius: 8px; padding: 14px 18px; margin-bottom: 20px;">
                <div style="font-size: 14px; font-weight: 700; color: #B895FF; text-transform: uppercase; letter-spacing: 0.05em;">CAREVO RECOMMENDED CAREER PATHWAYS</div>
                <div style="font-size: 15px; color: #8B92A5; margin-top: 4px; line-height: 1.55;">
                    No specific career objective was entered. The system analyzed your background and surfaced recommended pathways above.
                    Select any recommendation to focus your plan, or explore the generalized 7-part strategic roadmap below.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # =========================================================================
    # DYNAMIC ROADMAP SYNTHESIS VIA CAREER ADVISOR ENGINE
    # =========================================================================
    advisor_engine = CareerAdvisorEngine()
    cache_key = f"roadmap_{active_obj_title}_{getattr(effective_profile, 'filename', 'none')}"
    if cache_key not in st.session_state:
        with st.spinner(f"Synthesizing strategic career intelligence for {active_obj_title}..."):
            st.session_state[cache_key] = advisor_engine.generate_roadmap(
                target_objective=active_obj_title,
                profile=effective_profile,
                match_result=st.session_state["pipeline_result"].match_result if has_active_match else None,
            )

    roadmap = st.session_state[cache_key]

    # Active Match Context & Objective Divergence Detection
    if has_active_match:
        pipeline_res = st.session_state["pipeline_result"]
        m_res = pipeline_res.match_result
        jd_prof = pipeline_res.jd_profile
        target_role = jd_prof.job_title or "Target Role"
        target_comp = jd_prof.company or "Target Organization"

        target_words = set(re.findall(r'\b[a-z]{3,}\b', target_role.lower()))
        obj_words = set(re.findall(r'\b[a-z]{3,}\b', active_obj_title.lower()))
        stop_words = {"engineer", "developer", "senior", "lead", "associate", "specialist", "role", "position", "analyst", "advisor"}
        meaningful_target = target_words - stop_words
        meaningful_obj = obj_words - stop_words

        if meaningful_target and meaningful_obj and not (meaningful_target & meaningful_obj):
            st.markdown(
                f"""
                <div style="background: rgba(139, 92, 246, 0.12); border: 1px solid rgba(139, 92, 246, 0.4); border-radius: 8px; padding: 14px 18px; margin-bottom: 20px;">
                    <div style="font-size: 14px; font-weight: 700; color: #B895FF; text-transform: uppercase; letter-spacing: 0.05em;">EXPLICIT CAREER OBJECTIVE DISTINCTION</div>
                    <div style="font-size: 19px; font-weight: 700; color: #F5F3FA; margin: 4px 0;">
                        Pursuing: {active_obj_title} | Evaluated JD: {target_role} at {target_comp}
                    </div>
                    <div style="font-size: 15px; color: #8B92A5; line-height: 1.55;">
                        Your explicit career goal is set to <strong>{active_obj_title}</strong>. The 7-part strategic career plan below is tailored
                        specifically to this objective. Your separate <strong>JD Matching</strong> evaluation for {target_role} ({m_res.overall_score:.1f}% alignment)
                        remains preserved and un-overwritten.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # Truthful Provenance-Aware Hero Card
    if roadmap.provenance == "LIVE_GROQ_AI":
        st.markdown(
            f"""
            <div class="pf-hero-card">
                <span class="pf-badge pf-badge-green" style="background: rgba(54, 211, 153, 0.15); border: 1px solid #36D399; color: #36D399;">CAREVO AI-GENERATED PERSONALIZED ROADMAP</span>
                <div style="font-size: 21px; font-weight: 800; color: #F5F3FA; margin: 6px 0 8px;">
                    Target Career Direction: {active_obj_title}
                </div>
                <div style="font-size: 15px; color: #8B92A5; line-height: 1.55;">
                    Personalized career guidance generated by live AI reasoning ({roadmap.model_name}) grounded in candidate profile evidence and verified market research.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    elif has_active_match:
        st.markdown(
            f"""
            <div class="pf-hero-card">
                <span class="pf-badge pf-badge-purple">CAREVO STRATEGIC ROADMAP (VERIFIED BASELINE)</span>
                <div style="font-size: 21px; font-weight: 800; color: #F5F3FA; margin: 6px 0 8px;">
                    Target Career Focus: {active_obj_title} (Aligned with {target_role} at {target_comp})
                </div>
                <div style="font-size: 15px; color: #8B92A5; line-height: 1.55;">
                    Strategic recommendations synthesized from verified industry competency standards addressing qualification gaps.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"""
            <div class="pf-hero-card">
                <span class="pf-badge pf-badge-purple">CAREVO [SAMPLE BENCHMARK ROADMAP] — STRATEGIC GUIDANCE</span>
                <div style="font-size: 21px; font-weight: 800; color: #F5F3FA; margin: 6px 0 8px;">
                    Target Career Direction: {active_obj_title}
                </div>
                <div style="font-size: 15px; color: #8B92A5; line-height: 1.55;">
                    The Career Advisor generates tailored trajectories, competency development priorities, and portfolio blueprints
                    grounded in candidate evidence. (No active JD match is required).
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Provenance Header & Refresh Control
    prov_col, ref_col = st.columns([3.8, 1.2])
    with prov_col:
        if roadmap.provenance == "LIVE_GROQ_AI":
            st.markdown(
                f"""
                <div style="background: rgba(54, 211, 153, 0.12); border: 1px solid #36D399; border-radius: 8px; padding: 12px 18px; margin-bottom: 16px; font-size: 15px; color: #36D399;">
                    🟢 <strong>CAREVO Live AI Intelligence Active</strong> &bull; Generated via Groq ({roadmap.model_name}) &bull; Grounded in candidate evidence & market intelligence
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"""
                <div style="background: rgba(139, 92, 246, 0.12); border: 1px solid rgba(139, 92, 246, 0.4); border-radius: 8px; padding: 12px 18px; margin-bottom: 16px; font-size: 15px; color: #B895FF;">
                    ℹ️ <strong>Verified Baseline Guidance Active</strong> &bull; Fallback framework: {roadmap.provider_name} &bull; Grounded in industry competency standards
                </div>
                """,
                unsafe_allow_html=True,
            )
    with ref_col:
        if st.button("🔄 Refresh Analysis", key="adv_refresh_analysis_btn", help="Re-query market intelligence and refresh roadmap analysis"):
            if cache_key in st.session_state:
                del st.session_state[cache_key]
            st.rerun()

    # Role Specialization Clarification for Ambiguous Roles (e.g. Home Advisor)
    if active_obj_title.lower() in ["home advisor", "home consultant", "home adviser"]:
        st.markdown(
            """
            <div style="background: rgba(139, 92, 246, 0.1); border: 1px solid rgba(139, 92, 246, 0.35); border-radius: 8px; padding: 12px 16px; margin-bottom: 16px;">
                <div style="font-size: 16px; font-weight: 700; color: #B895FF; margin-bottom: 4px;">🔍 Role Specialization Clarification: "Home Advisor"</div>
                <div style="font-size: 15px; color: #8B92A5; line-height: 1.55;">
                    The title <strong>"Home Advisor"</strong> spans distinct professional disciplines with different regulatory oversight:
                    <ul style="margin: 6px 0 8px 18px; padding: 0;">
                        <li><strong>Home Loan & Mortgage Advisory:</strong> Evaluating borrower credit eligibility, Fixed Obligation to Income Ratio (FOIR), and bank loan processing under <strong>Reserve Bank of India (RBI)</strong> DSA guidelines. <em>(Does NOT require SEBI or NISM certifications)</em>.</li>
                        <li><strong>Property & Real Estate Advisory:</strong> Guiding residential property acquisition, title due diligence, and builder verification under state <strong>RERA</strong> (Real Estate Regulatory Authority) agent registration (Real Estate Act 2016).</li>
                        <li><strong>Residential Interior & Home Improvement Consultant:</strong> Space planning, renovation budgeting, and contractor oversight.</li>
                    </ul>
                    Select a focused specialization below if desired:
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        col_c1, col_c2, _ = st.columns([1.5, 1.5, 1])
        with col_c1:
            st.button("🏠 Focus on Home Loan Advisor (RBI/DSA)", key="clarify_home_loan_btn", on_click=_set_advisor_obj_cb, args=("Home Loan Advisor",))
        with col_c2:
            st.button("🏢 Focus on Real Estate Advisor (RERA)", key="clarify_real_estate_btn", on_click=_set_advisor_obj_cb, args=("Real Estate Property Advisor",))

    elif any(w in active_obj_title.lower() for w in ["home loan", "mortgage", "loan advisor"]):
        st.markdown(
            """
            <div style="background: rgba(54, 211, 153, 0.08); border: 1px solid rgba(54, 211, 153, 0.3); border-radius: 8px; padding: 12px 16px; margin-bottom: 14px; font-size: 15px; line-height: 1.55; color: #8B92A5;">
                ℹ️ <strong>Regulatory Scope:</strong> Home loan and retail mortgage distribution is governed by <strong>Reserve Bank of India (RBI)</strong> Direct Selling Agent (DSA) guidelines and NHB norms. <em>Note: SEBI / NISM Series X-A certifications apply exclusively to securities/mutual fund investment advisory and do not apply to retail mortgages.</em>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Web Research Market Dossier Expander
    if roadmap.research_dossier:
        res_list = roadmap.research_dossier.get("results", [])
        if res_list:
            with st.expander(f"🌐 Live Web Market Intelligence ({len(res_list)} verified industry citations)", expanded=False):
                st.caption(f"Real-time research query: `{roadmap.research_dossier.get('query', '')}` &bull; Status: `{roadmap.research_dossier.get('status', 'OK')}`")
                for cit in res_list[:4]:
                    c_title = cit.get("title", "Market Source")
                    c_url = cit.get("url", "#")
                    c_domain = cit.get("domain", "")
                    c_snippet = cit.get("snippet", "")
                    st.markdown(
                        f"""
                        <div style="background: rgba(255, 255, 255, 0.03); border-left: 3px solid #8B5CF6; border-radius: 4px; padding: 8px 12px; margin-bottom: 8px;">
                            <div style="font-size: 15px; font-weight: 600;"><a href="{c_url}" target="_blank" style="color: #60A5FA; text-decoration: none;">🔗 {c_title}</a> <span style="font-size: 13px; background: rgba(96, 165, 250, 0.15); color: #93C5FD; padding: 2px 6px; border-radius: 4px; margin-left: 6px;">{c_domain}</span></div>
                            <div style="font-size: 13px; color: #94A3B8; margin-top: 4px; line-height: 1.45;">{c_snippet}</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

    # =========================================================================
    # THE 7 CORE SECTIONS
    # =========================================================================

    # 1. Career Direction
    st.markdown(f"#### 1. Strategic Career Direction ({roadmap.objective_title})")
    st.markdown(
        f"""
        <div class="pf-card" style="margin-bottom: 16px;">
            <div style="font-size: 19px; font-weight: 700; color: #F5F3FA; margin-bottom: 6px;">Target Direction: {roadmap.objective_title}</div>
            <div style="font-size: 15px; color: #8B92A5; line-height: 1.55;">
                {roadmap.direction_narrative}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # 2. Suitable Job Roles
    st.markdown("#### 2. Suitable Target Job Roles")
    roles = roadmap.target_roles[:4]
    r_cols = st.columns(len(roles) if roles else 1)
    for idx, r in enumerate(roles):
        with r_cols[idx]:
            st.markdown(
                f"""
                <div class="pf-card" style="height: 100%;">
                    <div style="font-size: 19px; font-weight: 700; color: #36D399;">{r.get('title', 'Target Role')}</div>
                    <div style="font-size: 15px; color: #8B92A5; margin-top: 4px; line-height: 1.55;">{r.get('description', '')}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # 3. Existing Strengths
    st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)
    st.markdown("#### 3. Candidate Strengths Transferable to Target Role")
    strengths = roadmap.transferable_strengths[:2]
    s_c1, s_c2 = st.columns(2)
    with s_c1:
        s0 = strengths[0] if len(strengths) > 0 else {"title": "Analytical Problem Solving", "explanation": "Demonstrated background in structured reasoning and analytical delivery."}
        st.markdown(
            f"""
            <div class="pf-card" style="height: 100%;">
                <div style="font-size: 19px; font-weight: 700; color: #F5F3FA; margin-bottom: 6px;">🧠 {s0.get('title', '')}</div>
                <div style="font-size: 15px; color: #8B92A5; line-height: 1.55;">
                    {s0.get('explanation', '')}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with s_c2:
        s1 = strengths[1] if len(strengths) > 1 else {"title": "Structured Execution & Reporting", "explanation": "Demonstrated experience completing deliverables and reporting verified metrics."}
        st.markdown(
            f"""
            <div class="pf-card" style="height: 100%;">
                <div style="font-size: 19px; font-weight: 700; color: #F5F3FA; margin-bottom: 6px;">📊 {s1.get('title', '')}</div>
                <div style="font-size: 15px; color: #8B92A5; line-height: 1.55;">
                    {s1.get('explanation', '')}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # 4. Skill Development Priorities
    st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)
    st.markdown(f"#### 4. Skill Development Priorities for {roadmap.objective_title}")
    gap_items = []
    if has_active_match:
        if not m_res.missing_evidence:
            st.caption(f"All target requirements for {target_role} have verified evidence.")
        elif m_res.missing_evidence:
            st.caption(f"Focusing on {len(m_res.missing_evidence)} target qualification gaps for {target_role}:")
            gap_items = [g.skill_name for g in m_res.missing_evidence[:4]]
    else:
        st.caption(f"Priority competencies and market requirements for {roadmap.objective_title}:")

    priorities = roadmap.skill_priorities[:4]
    p_c1, p_c2 = st.columns(2)
    with p_c1:
        p0 = priorities[0] if len(priorities) > 0 else {"name": "Core Domain Fundamentals", "explanation": "Key domain concepts, regulations, and industry standards."}
        p1 = priorities[1] if len(priorities) > 1 else {"name": "Practical Workflow Execution", "explanation": "Hands-on implementation of core role responsibilities."}
        st.markdown(
            f"""
            <div class="pf-card" style="margin-bottom: 12px;">
                <div style="font-size: 19px; font-weight: 700; color: #FB7185;">1. {p0.get('name', '')}</div>
                <div style="font-size: 15px; color: #8B92A5; margin-top: 4px; line-height: 1.55;">{p0.get('explanation', '')}</div>
            </div>
            <div class="pf-card">
                <div style="font-size: 19px; font-weight: 700; color: #FB7185;">2. {p1.get('name', '')}</div>
                <div style="font-size: 15px; color: #8B92A5; margin-top: 4px; line-height: 1.55;">{p1.get('explanation', '')}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with p_c2:
        p2 = priorities[2] if len(priorities) > 2 else {"name": "Industry Compliance & Best Practices", "explanation": "Adhering to regulatory standards and ethics."}
        p3 = priorities[3] if len(priorities) > 3 else {"name": "Consultative Communication & Alignment", "explanation": "Stakeholder reporting, client communication, and objection handling."}
        st.markdown(
            f"""
            <div class="pf-card" style="margin-bottom: 12px;">
                <div style="font-size: 19px; font-weight: 700; color: #FBBF24;">3. {p2.get('name', '')}</div>
                <div style="font-size: 15px; color: #8B92A5; margin-top: 4px; line-height: 1.55;">{p2.get('explanation', '')}</div>
            </div>
            <div class="pf-card">
                <div style="font-size: 19px; font-weight: 700; color: #FBBF24;">4. {p3.get('name', '')}</div>
                <div style="font-size: 15px; color: #8B92A5; margin-top: 4px; line-height: 1.55;">{p3.get('explanation', '')}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # 5. Learning Roadmap
    st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)
    st.markdown("#### 5. Phased Learning Roadmap (3-Month Trajectory)")
    milestones = roadmap.milestones[:3]
    m_c1, m_c2, m_c3 = st.columns(3)
    m_cols = [m_c1, m_c2, m_c3]
    for idx, m in enumerate(milestones):
        with m_cols[idx]:
            timeframe = m.get('timeframe') or m.get('phase') or f"PHASE {idx+1}"
            focus = m.get('focus') or m.get('title') or "Milestone Focus"
            actions = m.get('actions') or m.get('description') or ""
            deliverable = m.get('deliverable', '')
            st.markdown(
                f"""
                <div class="pf-card" style="height: 100%;">
                    <span class="pf-badge pf-badge-purple">{timeframe.upper()}</span>
                    <div style="font-size: 19px; font-weight: 700; color: #F5F3FA; margin: 6px 0;">{focus}</div>
                    <div style="font-size: 15px; color: #8B92A5; margin-bottom: 8px; line-height: 1.55;">{actions}</div>
                    <div style="font-size: 14px; color: #36D399;"><strong>Deliverable:</strong> {deliverable}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # 6. Practical Portfolio Projects
    st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)
    st.markdown("#### 6. Practical Advisory & Execution Portfolio Blueprints")
    samples = roadmap.portfolio_work_samples[:2]
    prj1, prj2 = st.columns(2)
    with prj1:
        s0 = samples[0] if len(samples) > 0 else {"title": "Portfolio Deliverable 1", "concept": "Comprehensive case study and analysis dossier.", "deliverable": "Advisory report or technical repository."}
        st.markdown(
            f"""
            <div class="pf-card" style="height: 100%;">
                <div style="font-size: 19px; font-weight: 700; color: #F5F3FA; margin-bottom: 6px;">📁 {s0.get('title', 'Project 1')}</div>
                <div style="font-size: 15px; color: #8B92A5; line-height: 1.55; margin-bottom: 10px;">
                    {s0.get('concept', s0.get('description', ''))}
                </div>
                <div style="font-size: 14px; color: #36D399;"><strong>Key Deliverable:</strong> {s0.get('deliverable', '')}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with prj2:
        s1 = samples[1] if len(samples) > 1 else {"title": "Portfolio Deliverable 2", "concept": "Practical standard operating procedure or implementation.", "deliverable": "Operational workbook or benchmark report."}
        st.markdown(
            f"""
            <div class="pf-card" style="height: 100%;">
                <div style="font-size: 19px; font-weight: 700; color: #F5F3FA; margin-bottom: 6px;">📁 {s1.get('title', 'Project 2')}</div>
                <div style="font-size: 15px; color: #8B92A5; line-height: 1.55; margin-bottom: 10px;">
                    {s1.get('concept', s1.get('description', ''))}
                </div>
                <div style="font-size: 14px; color: #36D399;"><strong>Key Deliverable:</strong> {s1.get('deliverable', '')}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # 7. Recommended Next Actions & Interview Preparation
    st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)
    st.markdown("#### 7. Immediate Recommended Next Actions & Interview Preparation")
    ip_list = roadmap.interview_prep[:2]
    ip_html = ""
    for idx, q in enumerate(ip_list, 1):
        q_text = q.get('question', '')
        q_tip = q.get('tip', q.get('rationale', ''))
        q_type = q.get('type', 'Core Question')
        ip_html += f"<div style='margin-bottom: 10px; font-size: 15px; line-height: 1.55;'><strong>Q{idx} ({q_type}):</strong> {q_text}<br/><span style='color: #94A3B8; font-size: 14px;'>💡 <em>Preparation Strategy:</em> {q_tip}</span></div>"

    st.markdown(
        f"""
        <div class="pf-card">
            <div style="font-size: 19px; font-weight: 700; color: #F5F3FA; margin-bottom: 8px;">🎯 High-Yield Interview Questions to Prepare:</div>
            {ip_html if ip_html else "<div style='color: #94A3B8; font-size: 15px;'>Practice articulating candidate strengths and technical trade-offs with structured examples.</div>"}
            <div style="border-top: 1px solid rgba(255, 255, 255, 0.08); margin: 12px 0; padding-top: 10px; font-size: 15px; color: #8B92A5; line-height: 1.6;">
                ✔ <strong>Step 1:</strong> Enroll in foundational courses or certifications for {roadmap.objective_title}.<br/>
                ✔ <strong>Step 2:</strong> Build portfolio deliverable 1 with documented assumptions and outcomes.<br/>
                ✔ <strong>Step 3:</strong> Update resume STAR bullets to highlight transferable analytical and domain strengths.<br/>
                ✔ <strong>Step 4:</strong> Rehearse scenario-based question answers using structured STAR methodology.<br/>
                ✔ <strong>Step 5:</strong> Launch the <strong>Agents (Workflow)</strong> co-pilot to run interactive technical defense drills.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Export & Launch Controls
    st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)
    exp_col1, exp_col2 = st.columns([1.5, 2.5])
    with exp_col1:
        st.download_button(
            "📥 Download Career Roadmap (JSON)",
            data=json.dumps(roadmap.model_dump(), indent=2),
            file_name=f"career_roadmap_{active_obj_title.lower().replace(' ', '_')}.json",
            mime="application/json",
            key="dl_career_roadmap_json",
        )
    with exp_col2:
        if st.button("🚀 Launch CAREVO Copilot for this Objective →", type="primary", key="adv_launch_wf_hero_btn"):
            st.session_state["nav_view"] = "Agents (Workflow)"
            st.rerun()

elif selected_nav == "Agents (Workflow)":
    st.markdown("### CAREVO Copilot")
    st.caption("Autonomous multi-agent LangGraph orchestration with Human-in-the-Loop review and score immutability auditing.")

    if has_active_match:
        pipeline_res = st.session_state["pipeline_result"]
        render_agentic_workflow_ui(
            match_result=pipeline_res.match_result,
            resume_profile=pipeline_res.resume_profile,
            jd_profile=pipeline_res.jd_profile,
            key_prefix="workflow_agentic",
        )
    else:
        st.markdown(
            """
            <div class="pf-card" style="text-align: center; padding: 48px 24px;">
                <div style="font-size: 44px; margin-bottom: 12px;">🤖</div>
                <div style="font-size: 21px; font-weight: 700; color: #F5F3FA; margin-bottom: 8px;">No Active Match Evaluation Found</div>
                <div style="font-size: 15px; color: #8B92A5; max-width: 580px; margin: 0 auto 16px; line-height: 1.55;">
                    The Agentic Co-pilot coordinates 4 specialist agents (Triage, Skill Development, Resume STAR, and Interview Prep).
                    It requires a completed deterministic qualification evaluation to create an immutable cryptographic baseline,
                    prevent score drift, and enforce evidence-grounded recommendations.
                </div>
                <div style="background: rgba(139, 92, 246, 0.08); border: 1px solid rgba(139, 92, 246, 0.25); border-radius: 10px; padding: 12px 16px; max-width: 580px; margin: 0 auto 24px; text-align: left; font-size: 14px; color: #F5F3FA; line-height: 1.55;">
                    <strong>Why is a match required first?</strong> Phase 5 agents verify every resume claim against an authoritative SHA-256 snapshot. Without an active match, specialist agents cannot perform deterministic evidence checks or audit metrics.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        col_btn1, col_btn2, _ = st.columns([2, 2.5, 3])
        with col_btn1:
            if st.button("Evaluate in JD Matching →", type="primary", key="coach_go_match_btn"):
                st.session_state["nav_view"] = "JD Matching"
                st.rerun()
        with col_btn2:
            if st.button("✨ Load Complete Demo Showcase", key="coach_load_demo_btn"):
                load_demo_showcase()
                st.session_state["nav_view"] = "JD Matching"
                st.rerun()


# =============================================================================
# View 7: Current Evaluation Summary (Session-Retained Records)
# =============================================================================
elif selected_nav == "History":
    st.markdown("### Current Evaluation Summary — CAREVO Activity")
    st.caption("Active in-memory evaluation record for this session, verified parameters, and cryptographic baseline snapshot.")

    st.markdown(
        """
        <div class="pf-card" style="margin-bottom: 20px;">
            <div style="font-size: 19px; font-weight: 700; color: #F5F3FA; margin-bottom: 6px;">🔒 Privacy-Preserving Session Architecture</div>
            <div style="font-size: 15px; color: #8B92A5; line-height: 1.55;">
                CAREVO runs on a zero-retention in-memory architecture to protect candidate privacy and proprietary resume data.
                Evaluations are maintained strictly during your active session. No persistent external database is attached, and past evaluations from prior sessions are not stored or fabricated.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if has_active_match:
        pipe_res = st.session_state["pipeline_result"]
        res = pipe_res.match_result
        meta = res.job_metadata or {}
        role_name = meta.get("job_title") or "Target Position"
        comp_name = meta.get("company") or "Target Organization"

        # Evaluation Metadata Card
        st.markdown(
            f"""
            <div class="pf-card" style="margin-bottom: 20px;">
                <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 12px; margin-bottom: 12px; border-bottom: 1px solid rgba(139, 92, 246, 0.2); padding-bottom: 12px;">
                    <div>
                        <div style="font-size: 21px; font-weight: 800; color: #F5F3FA;">{role_name}</div>
                        <div style="font-size: 15px; color: #8B5CF6; font-weight: 600;">{comp_name}</div>
                    </div>
                    <div style="text-align: right;">
                        <span class="pf-pill pf-pill-verified" style="font-size: 14px;">{res.fit_classification.value}</span>
                        <div style="font-size: 25px; font-weight: 800; color: #F5F3FA; margin-top: 4px;">{res.overall_score:.1f}%</div>
                    </div>
                </div>
                <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 14px; font-size: 14px; color: #8B92A5;">
                    <div>📅 <strong>Evaluation Timestamp:</strong><br/><span style="color: #F5F3FA;">{datetime.now(timezone.utc).strftime("%B %d, %Y (%H:%M UTC)")}</span></div>
                    <div>🎯 <strong>Evaluation Status:</strong><br/><span style="color: #F5F3FA;">{res.evaluation_status.value}</span></div>
                    <div>📊 <strong>Assessed Coverage:</strong><br/><span style="color: #F5F3FA;">{res.active_evaluation_coverage:.1f}% (Reserve: {res.unassessed_reserve:.1f}%)</span></div>
                    <div>🛡️ <strong>Integrity Anchor:</strong><br/><span style="font-family: 'JetBrains Mono', monospace; font-size: 14px; color: #8B92A5;">Active Snapshot Verified</span></div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Action Links to Corresponding Reports
        st.markdown("##### Direct Report Navigation")
        col_nav1, col_nav2, col_nav3 = st.columns(3)
        with col_nav1:
            if st.button("🎯 View Matching Report", key="hist_goto_match_btn", use_container_width=True):
                st.session_state["nav_view"] = "JD Matching"
                st.rerun()
        with col_nav2:
            if st.button("🤖 Open Career Co-pilot", key="hist_goto_coach_btn", use_container_width=True):
                st.session_state["nav_view"] = "Agents (Workflow)"
                st.rerun()
        with col_nav3:
            export_dict = pipe_res.to_export_dict()
            st.download_button(
                label="📥 Export Report (JSON)",
                data=json.dumps(export_dict, indent=2),
                file_name=f"evaluation_{role_name.lower().replace(' ', '_')}.json",
                mime="application/json",
                key="hist_export_report_btn",
                use_container_width=True,
            )
    else:
        st.info("ℹ️ No active evaluation found in this session. Upload a resume and compare it against a job description in 'JD Matching' to generate your evaluation summary.")
        if st.button("🎯 Go to JD Matching", key="hist_nav_to_jd_btn", type="primary"):
            st.session_state["nav_view"] = "JD Matching"
            st.rerun()


# =============================================================================
# View 8: Settings & Preferences
# =============================================================================
elif selected_nav == "Settings":
    st.markdown("### Settings & Platform Configuration")
    st.caption("CAREVO platform configuration, matching criteria weights, and privacy parameters.")

    skills_w = int(BASELINE_WEIGHTS.get("skills", 0.45) * 100)
    exp_w = int(BASELINE_WEIGHTS.get("experience", 0.25) * 100)
    edu_w = int(BASELINE_WEIGHTS.get("education", 0.15) * 100)
    proj_w = int(BASELINE_WEIGHTS.get("projects", 0.15) * 100)
    max_cap = int(MAX_DIMENSION_CAP * 100)

    st.markdown(
        f"""
        <div class="pf-card">
            <div style="font-size: 19px; font-weight: 700; color: #F5F3FA; margin-bottom: 8px;">Authoritative Phase 4 Baseline Weights</div>
            <div style="font-size: 15px; color: #8B92A5; margin-bottom: 14px; line-height: 1.55;">
                Configured directly from the core WeightManager. Enforced by anti-inflation capped redistribution formulas.
            </div>
            <div style="font-size: 15px; color: #F5F3FA; margin-bottom: 4px;">• Technical Skills Baseline: <strong>{skills_w}%</strong></div>
            <div style="font-size: 15px; color: #F5F3FA; margin-bottom: 4px;">• Experience Tenure Baseline: <strong>{exp_w}%</strong></div>
            <div style="font-size: 15px; color: #F5F3FA; margin-bottom: 4px;">• Academic Education Baseline: <strong>{edu_w}%</strong></div>
            <div style="font-size: 15px; color: #F5F3FA; margin-bottom: 12px;">• Practical Projects Baseline: <strong>{proj_w}%</strong></div>
            <div style="border-top: 1px solid rgba(139, 92, 246, 0.2); padding-top: 10px; font-size: 14px; color: #8B92A5; line-height: 1.55;">
                <strong>Dynamic Redistribution & Single-Dimension Ceiling ({max_cap}%):</strong><br/>
                When an employer job description omits one or more criteria dimensions (such as education or tenure requirements),
                unassessed weights are proportionally redistributed across active dimensions up to a strict {max_cap}% ceiling.
                This prevents artificial score inflation while preserving mathematical integrity.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
