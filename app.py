# app.py — Subtext AI: Beyond the Words — Same words. Different meanings.
# =============================================================================
# An NLP student project exploring emotion classification and contextual
# interpretation in short text messages using the MELD dataset,
# TF-IDF vectorization, and Logistic Regression.
# Styled with a clean, professional red-and-white visual theme.
# =============================================================================

import os
import re
import ssl
import pickle
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

# -----------------------------------------------------------------------------
# 0. Streamlit Page Configuration (MUST be the first Streamlit command)
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Subtext AI — Beyond the Words",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------------------------------------------------------
# 1. NLTK Setup & Text Preprocessing
# -----------------------------------------------------------------------------
# macOS SSL fix for NLTK resource downloads
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

nltk.download("stopwords", quiet=True)
nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)

STOP_WORDS = set(stopwords.words("english"))
stemmer = PorterStemmer()


def preprocess(text: str) -> str:
    """
    Clean and normalise a raw message.
    Identical to the preprocessing function in train_model.py:
      1. Lowercase text
      2. Remove URLs
      3. Keep only letters and whitespace
      4. Tokenise on whitespace
      5. Remove stopwords and single characters, then stem
    """
    if not isinstance(text, str):
        return ""

    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    tokens = text.split()
    tokens = [
        stemmer.stem(tok)
        for tok in tokens
        if tok not in STOP_WORDS and len(tok) > 1
    ]
    return " ".join(tokens)


# -----------------------------------------------------------------------------
# 2. Model Loading (Cached)
# -----------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading trained model artifacts...")
def load_artifacts():
    """
    Loads saved model artifacts from the models/ directory.
    Returns (vectorizer, classifier, label_encoder, evaluation_results, missing_files).
    """
    base_path = Path("models")
    vec_path = base_path / "tfidf_vectorizer.pkl"
    model_path = base_path / "emotion_model.pkl"
    le_path = base_path / "label_encoder.pkl"
    eval_path = base_path / "evaluation_results.pkl"

    required_files = [vec_path, model_path, le_path, eval_path]
    missing = [str(p) for p in required_files if not p.exists()]

    if missing:
        return None, None, None, None, missing

    with open(vec_path, "rb") as f:
        vectorizer = pickle.load(f)
    with open(model_path, "rb") as f:
        clf = pickle.load(f)
    with open(le_path, "rb") as f:
        le = pickle.load(f)
    with open(eval_path, "rb") as f:
        eval_results = pickle.load(f)

    return vectorizer, clf, le, eval_results, []


vectorizer, clf, le, eval_results, missing_files = load_artifacts()

# -----------------------------------------------------------------------------
# 3. Custom CSS Theme: Premium Red-and-White Design
# -----------------------------------------------------------------------------
st.markdown(
    """
    <style>
    /* Main Background & Base Typography */
    .stApp {
        background-color: #FFFFFF;
        color: #252525;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }

    /* Streamlit top header & main containers */
    header[data-testid="stHeader"] {
        background-color: #FFFFFF;
    }

    [data-testid="stAppViewContainer"] {
        background-color: #FFFFFF;
    }

    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background-color: #FCFCFD;
        border-right: 1px solid #F0C9CC;
    }
    [data-testid="stSidebar"] * {
        color: #252525;
    }

    /* Content Cards */
    .subtext-card {
        background-color: #FFF8F8;
        border: 1px solid #F0C9CC;
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 18px;
        box-shadow: 0 1px 3px rgba(214, 40, 57, 0.04);
    }

    .subtext-card-subtle {
        background-color: #FFFFFF;
        border: 1px solid #F0C9CC;
        border-radius: 10px;
        padding: 14px 18px;
        margin-bottom: 14px;
    }

    .context-col-card {
        background-color: #FFF8F8;
        border: 1px solid #F0C9CC;
        border-radius: 10px;
        padding: 16px;
        height: 100%;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        box-shadow: 0 1px 2px rgba(214, 40, 57, 0.03);
    }

    /* Metric Cards */
    .metric-card {
        background-color: #FFF8F8;
        border: 1px solid #F0C9CC;
        border-radius: 10px;
        padding: 16px;
        text-align: center;
        box-shadow: 0 1px 2px rgba(214, 40, 57, 0.03);
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #D62839;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #666666;
        margin-top: 4px;
        font-weight: 500;
    }

    /* Accent Badges */
    .emotion-badge {
        display: inline-block;
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 0.95rem;
        letter-spacing: 0.5px;
        text-transform: uppercase;
        background-color: #FDEBEC;
        color: #A4161A;
        border: 1px solid #F0C9CC;
    }

    .uncertainty-badge {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 12px;
        font-size: 0.82rem;
        font-weight: 600;
    }
    .uncertainty-high {
        background-color: #FDEBEC;
        color: #A4161A;
        border: 1px solid #F0C9CC;
    }
    .uncertainty-mid {
        background-color: #FFFBEB;
        color: #B45309;
        border: 1px solid #FDE68A;
    }
    .uncertainty-low {
        background-color: #ECFDF5;
        color: #065F46;
        border: 1px solid #A7F3D0;
    }

    .subtext-tag {
        display: inline-block;
        padding: 3px 10px;
        border-radius: 12px;
        font-size: 0.78rem;
        background-color: #FDEBEC;
        color: #A4161A;
        border: 1px solid #F0C9CC;
        font-weight: 600;
    }

    /* Buttons: Primary Red (#D62839) with Hover Dark Red (#A4161A) */
    .stButton>button {
        background-color: #D62839 !important;
        color: #FFFFFF !important;
        border: 1px solid #D62839 !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        padding: 0.5rem 1.1rem !important;
        white-space: normal !important;
        height: auto !important;
        min-height: 2.4rem !important;
        text-align: center;
        line-height: 1.25;
        box-shadow: 0 1px 2px rgba(214, 40, 57, 0.15);
        transition: all 0.15s ease-in-out;
    }
    .stButton>button:hover {
        background-color: #A4161A !important;
        border-color: #A4161A !important;
        color: #FFFFFF !important;
        box-shadow: 0 2px 4px rgba(164, 22, 26, 0.25);
    }
    .stButton>button:active {
        background-color: #851216 !important;
        border-color: #851216 !important;
    }

    /* Text Area */
    .stTextArea textarea {
        background-color: #FFFFFF !important;
        color: #252525 !important;
        border: 1px solid #F0C9CC !important;
        border-radius: 8px !important;
    }
    .stTextArea textarea:focus {
        border-color: #D62839 !important;
        box-shadow: 0 0 0 1px #D62839 !important;
    }

    /* Headings */
    h1, h2, h3, h4, h5, h6 {
        color: #252525 !important;
        font-weight: 700 !important;
    }

    /* Radio buttons & Selectbox */
    .stSelectbox>div>div {
        background-color: #FFFFFF !important;
        border: 1px solid #F0C9CC !important;
        color: #252525 !important;
        border-radius: 8px !important;
    }
    .stRadio [role="radiogroup"] {
        gap: 8px;
    }
    .stRadio label span {
        color: #252525 !important;
    }

    /* Progress Bar: Red indicator on pale red track */
    .stProgress > div > div > div > div {
        background-color: #D62839 !important;
    }
    .stProgress > div > div {
        background-color: #FDEBEC !important;
        border-radius: 4px !important;
    }

    /* Code & Copy Containers (for Reply Lab) */
    div[data-testid="stCodeBlock"] {
        background-color: #FFFFFF !important;
        border: 1px solid #F0C9CC !important;
        border-radius: 8px !important;
    }
    div[data-testid="stCodeBlock"] code {
        color: #252525 !important;
        background-color: #FFFFFF !important;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, monospace !important;
        font-size: 0.92rem !important;
    }
    div[data-testid="stCodeBlock"] button {
        color: #A4161A !important;
        border-color: #F0C9CC !important;
    }

    /* Dataframe Table styling */
    [data-testid="stDataFrame"] {
        border: 1px solid #F0C9CC;
        border-radius: 8px;
        overflow: hidden;
    }

    /* Horizontal divider */
    hr {
        border-color: #F0C9CC !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# 4. Context Explanations & Reply Lab Templates
# -----------------------------------------------------------------------------

# General baseline emotion interpretations
baseline_interpretations = {
    "joy": "The message conveys positive, uplifting, or appreciative energy.",
    "sadness": "The message conveys low-energy disappointment, sorrow, or feeling down.",
    "anger": "The message conveys irritation, frustration, grievance, or hostility.",
    "fear": "The message conveys anxiety, worry, caution, or apprehension.",
    "surprise": "The message conveys astonishment, shock, or unexpectedness.",
    "disgust": "The message conveys distaste, disapproval, revulsion, or strong aversion.",
    "neutral": "The message conveys matter-of-fact information without strong emotional cues.",
}

# Ambiguity explanations based on emotion & short text dynamics
ambiguity_notes = {
    "joy": (
        "Positive wording ('Oh, great', 'Wonderful') is frequently used sarcastically in real dialogue. "
        "Without vocal pitch or facial cues, lexical models cannot reliably distinguish enthusiasm from exasperation."
    ),
    "anger": (
        "Expressions like 'Fine' or 'Whatever' may indicate mild boundary-setting, deep resentment, or simply "
        "resigned agreement. Text alone omits the interpersonal tension."
    ),
    "sadness": (
        "Subdued phrasing can reflect genuine distress, emotional withdrawal, or merely physical tiredness "
        "and brevity at the end of a long day."
    ),
    "fear": (
        "Anxious words might signify urgent concern, risk awareness, or casual colloquial exaggeration "
        "('I'm terrified of missing this train')."
    ),
    "surprise": (
        "Exclamations of surprise can be delightfully positive, deeply alarming, or neutral news, "
        "depending entirely on shared situational context."
    ),
    "disgust": (
        "Expressions of distaste may target physical sensory experiences (e.g., bad food or weather) or "
        "moral/personal disapproval, which cannot be discerned in isolation."
    ),
    "neutral": (
        "Short, direct statements ('Okay', 'Sure', 'Got it') are often misperceived as curt or passive-aggressive, "
        "even when the sender simply intended efficient communication."
    ),
}

# Context Explanations: Transparent rule-based templates for side-by-side comparison
context_explanations = {
    "Personal or friendly conversation": {
        "joy": "Likely genuine enthusiasm, celebration, or warm affection between peers.",
        "sadness": "An invitation for emotional support, empathy, and personal connection.",
        "anger": "Interpersonal friction or venting about an external hassle to a trusted friend.",
        "fear": "Personal vulnerability or anxiety shared in an informal, supportive space.",
        "surprise": "Excitement or curiosity regarding shared news or social developments.",
        "disgust": "Casual venting about an unpleasant experience, bad taste, or awkward incident.",
        "neutral": "Informal check-in or simple logistical confirmation among friends.",
    },
    "Professional conversation": {
        "joy": "Professional satisfaction, project success, or constructive positive reinforcement.",
        "sadness": "Concern regarding project setbacks, unmet deliverables, or workload fatigue.",
        "anger": "Workplace friction regarding blocked workflows, missed deadlines, or unclear ownership.",
        "fear": "Risk awareness, compliance apprehension, or concern about project milestones.",
        "surprise": "An unanticipated project development, market change, or sudden priority shift.",
        "disgust": "Firm professional rejection of a proposal, poor work quality, or ethical breach.",
        "neutral": "Standard, efficient business communication focused purely on logistics and deliverables.",
    },
    "Disagreement or argument": {
        "joy": "High likelihood of sarcastic, biting, or dismissive intent (e.g., 'Oh, wonderful').",
        "sadness": "Feeling overwhelmed, hurt, or emotionally withdrawing from the confrontation.",
        "anger": "Direct confrontation, boundary enforcement, or escalating interpersonal resentment.",
        "fear": "Anxiety regarding relationship damage or escalation of the dispute.",
        "surprise": "Disbelief at the other person's statements, accusations, or sudden stance.",
        "disgust": "Strong disapproval, moral indignation, or contempt toward the opposing position.",
        "neutral": "Emotional detachment, stonewalling, or an attempt to de-escalate the tension.",
    },
}

# Reply Lab Templates: Constructive, non-confrontational response templates for all 3 styles
reply_lab_templates = {
    "Gentle": {
        "joy": "I'm so glad to hear that! It's wonderful to celebrate good moments.",
        "sadness": "I hear you, and I'm sorry things feel heavy right now. Take all the time you need.",
        "anger": "I can understand why this situation feels frustrating. Let's take a moment and work through it together.",
        "fear": "It makes complete sense to feel cautious here. We can take this one step at a time.",
        "surprise": "That's quite an unexpected turn! How are you feeling about everything?",
        "disgust": "I completely understand why that felt unpleasant. Let's see how we can handle it.",
        "neutral": "Thanks for keeping me posted! Let me know if there's anything else you need.",
    },
    "Direct": {
        "joy": "Good to hear. Let's maintain this momentum moving forward.",
        "sadness": "I recognize this is difficult. What concrete steps can we take to improve the situation?",
        "anger": "Let's be candid about what isn't working so we can address the root issue directly.",
        "fear": "Let's identify the specific risks involved and put a clear mitigation plan in place.",
        "surprise": "That changes the situation. What are the immediate implications we need to prepare for?",
        "disgust": "That's problematic. Let's clearly outline what needs to change going forward.",
        "neutral": "Received and understood. I will proceed accordingly.",
    },
    "Neutral": {
        "joy": "Thank you for the update. It is good to see positive progress.",
        "sadness": "Noted. Please let me know how you would like to proceed from here.",
        "anger": "I acknowledge your perspective on this matter. Let's review the details objectively.",
        "fear": "Understood. We will keep these factors in mind as we evaluate the next steps.",
        "surprise": "Noted. We will factor this new information into our planning.",
        "disgust": "Acknowledged. We will review the matter objectively.",
        "neutral": "Understood. Thank you for the confirmation.",
    },
}

AMBIGUOUS_KEYWORDS = {"fine", "whatever", "great", "sure", "okay", "ok", "k", "cool", "right", "alright"}


def compute_model_uncertainty(probabilities: np.ndarray):
    """
    Computes a mathematically sound uncertainty indicator using Normalized Shannon Entropy:
      H = - sum(p_i * log2(p_i)) / log2(K)
    Scales strictly between 0.0 (0% uncertainty) and 1.0 (100% uncertainty / uniform distribution).
    Also computes the confidence margin between top-1 and runner-up prediction.
    """
    probs = np.array(probabilities, dtype=float)
    probs = probs[probs > 0]
    num_classes = len(probabilities)
    entropy = -np.sum(probs * np.log2(probs))
    max_entropy = np.log2(num_classes)
    normalized_entropy = float(entropy / max_entropy) if max_entropy > 0 else 0.0

    sorted_p = np.sort(probabilities)[::-1]
    top_margin = float(sorted_p[0] - sorted_p[1]) if len(sorted_p) > 1 else float(sorted_p[0])

    return normalized_entropy, top_margin


def is_short_or_ambiguous(raw_text: str, confidence: float) -> bool:
    """Detect if an utterance has high natural ambiguity risk."""
    words = raw_text.strip().lower().split()
    if len(words) <= 3:
        return True
    if any(re.sub(r"[^a-z]", "", w) in AMBIGUOUS_KEYWORDS for w in words):
        return True
    if confidence < 0.35:
        return True
    return False


# -----------------------------------------------------------------------------
# 5. Session State Management
# -----------------------------------------------------------------------------
if "history" not in st.session_state:
    st.session_state.history = []

if "message_input" not in st.session_state:
    st.session_state.message_input = ""

if "selected_style" not in st.session_state:
    st.session_state.selected_style = "Gentle"


# -----------------------------------------------------------------------------
# 6. Sidebar Navigation & Branding (Clean White & Red Theme)
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 💬 Subtext AI")
    st.markdown(
        "<span style='color: #D62839; font-weight: 700; font-size: 0.98rem;'>Beyond the Words</span><br>"
        "<span style='color: #666666; font-size: 0.85rem; font-style: italic;'>Same words. Different meanings.</span>",
        unsafe_allow_html=True
    )
    st.markdown("<div style='height: 6px;'></div>", unsafe_allow_html=True)
    st.caption(
        "An NLP student project exploring how short digital messages carry "
        "divergent subtexts depending on wording, context, and tone."
    )
    st.markdown("---")

    nav_page = st.radio(
        "Explore Project",
        ["💬 Message Decoder", "📊 Emotion Lab"],
        index=0,
        help="Switch between the interactive message analysis workspace and evaluation metrics."
    )

    st.markdown("---")
    st.markdown(
        """
        <div class="subtext-card-subtle">
            <div style="font-size: 0.78rem; color: #A4161A; font-weight: 700; text-transform: uppercase; letter-spacing: 0.5px;">Model Summary</div>
            <div style="margin-top: 8px; font-size: 0.86rem; color: #252525;"><b>Architecture:</b> TF-IDF + Logistic Regression</div>
            <div style="font-size: 0.86rem; color: #252525;"><b>Dataset:</b> MELD (Dialogue)</div>
            <div style="font-size: 0.86rem; color: #252525;"><b>Test Accuracy:</b> <span style="color: #D62839; font-weight: 700;">35.98%</span></div>
            <div style="font-size: 0.86rem; color: #252525;"><b>Classes:</b> 7 Emotions</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.caption(
        "⚠️ **Disclaimer:** This application demonstrates classical machine learning on surface text patterns. "
        "It does not detect genuine human intent, emotional sincerity, or hidden psychological states."
    )

# -----------------------------------------------------------------------------
# 7. Graceful Handling of Missing Model Files
# -----------------------------------------------------------------------------
if missing_files:
    st.error(
        f"⚠️ **Missing Model Artifacts Detected!**\n\n"
        f"The following required artifact files were not found:\n"
        + "".join(f"- `{m}`\n" for m in missing_files)
        + "\n**How to fix:** Please run the training script in your terminal:\n"
        "```bash\npython3 train_model.py\n```"
    )
    st.stop()


# =============================================================================
# VIEW 1: MESSAGE DECODER
# =============================================================================
if nav_page == "💬 Message Decoder":
    st.title("Message Decoder")
    st.markdown(
        "<p style='color: #666666; font-size: 1.05rem; margin-top: -10px;'>"
        "Analyze the predicted emotion of a message, examine mathematical model uncertainty, "
        "and compare how interpretation shifts across three conversational contexts side by side."
        "</p>",
        unsafe_allow_html=True,
    )

    # -------------------------------------------------------------------------
    # Feature 5: Example Messages (Clean, spacious, non-truncated layout)
    # -------------------------------------------------------------------------
    st.markdown("##### 📌 Try an example message:")
    example_messages = [
        "Fine, do whatever you want.",
        "Oh, great.",
        "Thank you so much for helping me.",
        "I can't believe this happened.",
        "Okay, I'll see you tomorrow."
    ]

    # 2-row grid to guarantee full readability and zero label truncation
    ex_row1 = st.columns(3)
    ex_row2 = st.columns(2)

    for i, ex_text in enumerate(example_messages[:3]):
        with ex_row1[i]:
            if st.button(f"💬 \"{ex_text}\"", key=f"ex_{i}", use_container_width=True):
                st.session_state.message_input = ex_text
                st.rerun()

    for i, ex_text in enumerate(example_messages[3:]):
        with ex_row2[i]:
            if st.button(f"💬 \"{ex_text}\"", key=f"ex_{i+3}", use_container_width=True):
                st.session_state.message_input = ex_text
                st.rerun()

    st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)

    # -------------------------------------------------------------------------
    # Input Workspace
    # -------------------------------------------------------------------------
    with st.container():
        user_message = st.text_area(
            "Enter a message to decode:",
            value=st.session_state.message_input,
            placeholder="Type or edit a short message here (e.g., 'Fine, do whatever you want.')...",
            height=100,
            key="user_text_input",
            help="You can type any message or click one of the example cards above."
        )
        st.session_state.message_input = user_message

        btn_col1, btn_col2, btn_spacer = st.columns([1.6, 1.6, 4.8])
        with btn_col1:
            analyze_clicked = st.button("🔍 Analyze Message", use_container_width=True)
        with btn_col2:
            reset_clicked = st.button("🔄 Reset Analysis", use_container_width=True)

    # Handle Reset Button
    if reset_clicked:
        st.session_state.message_input = ""
        st.session_state.history = []
        st.rerun()

    # -------------------------------------------------------------------------
    # Analysis Logic & Output
    # -------------------------------------------------------------------------
    if analyze_clicked:
        raw_input = user_message.strip()

        # Validation 1: Empty text
        if not raw_input:
            st.warning("⚠️ Please enter a message or select an example message above before analyzing.")
        else:
            cleaned_input = preprocess(raw_input)

            # Validation 2: Text becomes empty after preprocessing
            if not cleaned_input:
                st.warning(
                    "⚠️ The message contains no recognizable vocabulary words after preprocessing "
                    "(e.g., it contains only numbers, punctuation, or excluded stopwords). "
                    "Please enter a message containing standard words."
                )
            else:
                # TF-IDF Feature Extraction & Prediction
                X_feat = vectorizer.transform([cleaned_input])
                probs = clf.predict_proba(X_feat)[0]
                pred_idx = int(np.argmax(probs))
                predicted_emotion = le.inverse_transform([pred_idx])[0]
                model_confidence = float(probs[pred_idx])

                # Mathematical Uncertainty Indicator (Normalized Shannon Entropy)
                norm_entropy, top_margin = compute_model_uncertainty(probs)

                # Class labels & sorted probabilities
                class_labels = le.classes_
                sorted_prob_indices = np.argsort(probs)[::-1]

                # Natural ambiguity detection
                natural_ambiguity = is_short_or_ambiguous(raw_input, model_confidence)

                # Append to history
                record = {
                    "raw_text": raw_input,
                    "cleaned_text": cleaned_input,
                    "emotion": predicted_emotion,
                    "confidence": model_confidence,
                    "entropy": norm_entropy,
                    "top_margin": top_margin,
                    "natural_ambiguity": natural_ambiguity,
                    "probs": {class_labels[i]: float(probs[i]) for i in sorted_prob_indices},
                }
                st.session_state.history.insert(0, record)

    # -------------------------------------------------------------------------
    # Display Current Result (if history exists)
    # -------------------------------------------------------------------------
    if st.session_state.history:
        current = st.session_state.history[0]
        pred_emo = current["emotion"]
        conf = current["confidence"]
        entropy = current["entropy"]
        margin = current["top_margin"]

        st.markdown("<hr style='margin: 24px 0 20px 0;'>", unsafe_allow_html=True)
        st.subheader("Analysis Results")

        # Two-column layout: Prediction Summary & Predictive Uncertainty
        pred_col1, pred_col2 = st.columns([1.1, 1.2])

        with pred_col1:
            st.markdown(
                f"""
                <div class="subtext-card">
                    <div style="font-size: 0.8rem; color: #A4161A; font-weight: 700; text-transform: uppercase; letter-spacing: 0.5px;">Predicted Emotion</div>
                    <div style="margin-top: 8px;">
                        <span class="emotion-badge">{pred_emo}</span>
                    </div>
                    <div style="margin-top: 14px; font-size: 0.95rem; color: #252525;">
                        <b>Model Probability:</b> <span style="color: #D62839; font-weight: 700; font-size: 1.15rem;">{conf:.1%}</span>
                    </div>
                    <div style="margin-top: 6px; font-size: 0.82rem; color: #666666; line-height: 1.4;">
                        <em>Note: Probability reflects the statistical weight assigned by Logistic Regression based on MELD training data. It is not an assessment of genuine human emotional intent.</em>
                    </div>
                    <div style="margin-top: 16px;">
                        <div style="font-size: 0.82rem; font-weight: 600; color: #252525; margin-bottom: 6px;">Baseline Model Interpretation:</div>
                        <div style="font-size: 0.88rem; color: #252525; background: #FFFFFF; padding: 10px 12px; border-radius: 6px; border: 1px solid #F0C9CC;">
                            {baseline_interpretations.get(pred_emo, '')}
                        </div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with pred_col2:
            if entropy > 0.80:
                unc_class = "uncertainty-high"
                unc_label = "HIGH SPREAD"
                unc_desc = "Probabilities are distributed across multiple competing emotion classes."
            elif entropy > 0.60:
                unc_class = "uncertainty-mid"
                unc_label = "MODERATE SPREAD"
                unc_desc = "The top emotion leads, but noticeable probability is assigned to other categories."
            else:
                unc_class = "uncertainty-low"
                unc_label = "CONCENTRATED"
                unc_desc = "The classifier concentrated probability primarily in one category."

            st.markdown(
                f"""
                <div class="subtext-card">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <div style="font-size: 0.8rem; color: #A4161A; font-weight: 700; text-transform: uppercase; letter-spacing: 0.5px;">Model Predictive Uncertainty</div>
                        <span class="uncertainty-badge {unc_class}">{unc_label}</span>
                    </div>
                    <div style="margin-top: 12px; display: flex; gap: 24px;">
                        <div>
                            <div style="font-size: 0.8rem; color: #666666;">Normalized Shannon Entropy:</div>
                            <div style="font-size: 1.25rem; font-weight: 700; color: #D62839;">{entropy:.2f} <span style="font-size: 0.8rem; color: #666666; font-weight: 400;">/ 1.00</span></div>
                        </div>
                        <div>
                            <div style="font-size: 0.8rem; color: #666666;">Runner-Up Margin:</div>
                            <div style="font-size: 1.25rem; font-weight: 700; color: #252525;">+{margin:.1%}</div>
                        </div>
                    </div>
                    <div style="margin-top: 10px; font-size: 0.86rem; color: #252525; line-height: 1.4;">
                        {unc_desc}
                    </div>
                    <div style="margin-top: 12px; font-size: 0.78rem; color: #666666; line-height: 1.4; border-top: 1px solid #F0C9CC; padding-top: 8px;">
                        ℹ️ <b>What this measures:</b> The mathematical dispersion of the model's 7-class probability output (information entropy).<br>
                        ⚠️ <b>What it cannot establish:</b> It does <i>not</i> measure psychological ambiguity, deceit, or sarcasm. A phrase like "Fine." can have low model entropy if its words skewed heavily toward one training class, despite being socially ambiguous.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # Class Probability Distribution across all 7 emotions
        st.markdown(
            """
            <div class="subtext-card" style="padding-bottom: 12px; margin-bottom: 12px;">
                <div style="font-size: 0.8rem; color: #A4161A; font-weight: 700; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 10px;">Class Probability Distribution (All 7 Emotions)</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        prob_cols = st.columns(7)
        for idx, (emo_name, prob_val) in enumerate(current["probs"].items()):
            with prob_cols[idx]:
                is_top = (emo_name == pred_emo)
                border_style = "border: 2px solid #D62839;" if is_top else "border: 1px solid #F0C9CC;"
                bg_style = "background-color: #FDEBEC;" if is_top else "background-color: #FFFFFF;"
                title_color = "#A4161A" if is_top else "#252525"
                num_color = "#D62839" if is_top else "#666666"
                st.markdown(
                    f"""
                    <div style="{bg_style} {border_style} border-radius: 8px; padding: 10px 8px; text-align: center; margin-bottom: 8px;">
                        <div style="font-size: 0.82rem; font-weight: 700; color: {title_color}; text-transform: capitalize;">{emo_name}</div>
                        <div style="font-size: 1.1rem; font-weight: 700; color: {num_color}; margin-top: 4px;">{prob_val:.1%}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                st.progress(prob_val)

        # ---------------------------------------------------------------------
        # Feature 2: Side-by-Side Context Switcher (Compare interpretations)
        # ---------------------------------------------------------------------
        st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)
        st.markdown(
            """
            <div style="display: flex; justify-content: space-between; align-items: baseline;">
                <h4 style="margin-bottom: 4px;">Context Switcher: Side-by-Side Interpretation</h4>
                <span class="subtext-tag">Rule-Based Context Exploration</span>
            </div>
            <p style="color: #666666; font-size: 0.9rem; margin-top: -2px;">
                How might the exact same message be perceived in different conversational settings?
                <i>Note: These transparent rule-based templates illustrate perception differences; they do <b>not</b> alter the trained classifier's prediction.</i>
            </p>
            """,
            unsafe_allow_html=True,
        )

        ctx_col1, ctx_col2, ctx_col3 = st.columns(3)

        contexts_list = [
            ("Personal or friendly conversation", "Friendly & Informal", "#D62839", ctx_col1),
            ("Professional conversation", "Workplace & Professional", "#A4161A", ctx_col2),
            ("Disagreement or argument", "Conflict & Argument", "#D62839", ctx_col3),
        ]

        for ctx_key, ctx_title, tag_color, col_elem in contexts_list:
            ctx_text = context_explanations.get(ctx_key, {}).get(pred_emo, "Interpretation not available.")
            with col_elem:
                st.markdown(
                    f"""
                    <div class="context-col-card">
                        <div>
                            <div style="font-size: 0.8rem; font-weight: 700; color: {tag_color}; text-transform: uppercase; letter-spacing: 0.5px;">{ctx_title}</div>
                            <div style="font-size: 0.92rem; color: #252525; margin-top: 10px; line-height: 1.5;">
                                {ctx_text}
                            </div>
                        </div>
                        <div style="margin-top: 14px; font-size: 0.75rem; color: #666666; border-top: 1px solid #F0C9CC; padding-top: 8px;">
                            Context: <em>{ctx_key}</em>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        if current["natural_ambiguity"]:
            st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
            st.markdown(
                """
                <div style="font-size: 0.86rem; color: #92400E; background: #FFFBEB; padding: 10px 14px; border-radius: 8px; border: 1px solid #FDE68A;">
                    ⚠️ <b>Short Text Ambiguity Warning:</b> The available text is very brief and lacks contextual prosody (vocal pitch, timing, facial expression). In real human interactions, tone and relationship history matter significantly more than literal surface vocabulary.
                </div>
                """,
                unsafe_allow_html=True,
            )

        # ---------------------------------------------------------------------
        # Feature 3: Enhanced Reply Lab (Gentle, Direct, Neutral with One-Click Copy)
        # ---------------------------------------------------------------------
        st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)
        st.markdown(
            """
            <div style="display: flex; justify-content: space-between; align-items: baseline;">
                <h4 style="margin-bottom: 4px;">Reply Lab: Suggested Responses</h4>
                <span class="subtext-tag">Predefined Tone Templates</span>
            </div>
            <p style="color: #666666; font-size: 0.9rem; margin-top: -2px;">
                Constructive, non-confrontational replies to keep communication productive. Use the copy button on any suggestion to copy it to your clipboard.
                <i>These responses are communicative templates, not machine learning outputs.</i>
            </p>
            """,
            unsafe_allow_html=True,
        )

        reply_col1, reply_col2, reply_col3 = st.columns(3)

        styles_info = [
            ("Gentle", "Gentle & Empathetic", "Empathetic, validating, and calming.", reply_col1),
            ("Direct", "Direct & Candid", "Clear, candid, and solution-focused.", reply_col2),
            ("Neutral", "Neutral & Balanced", "Objective, professional, and balanced.", reply_col3),
        ]

        for style_key, style_label, style_sub, r_col in styles_info:
            reply_text = reply_lab_templates.get(style_key, {}).get(pred_emo, "Thank you for your message.")
            with r_col:
                st.markdown(
                    f"""
                    <div style="background-color: #FFF8F8; border: 1px solid #F0C9CC; border-radius: 10px; padding: 14px 14px 8px 14px;">
                        <div style="font-size: 0.85rem; font-weight: 700; color: #A4161A;">{style_label}</div>
                        <div style="font-size: 0.78rem; color: #666666; margin-bottom: 8px;">{style_sub}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                # Streamlit's st.code includes a built-in one-click "Copy to clipboard" icon
                st.code(reply_text, language=None)

        # ---------------------------------------------------------------------
        # Conversation / Analysis History
        # ---------------------------------------------------------------------
        if len(st.session_state.history) > 1:
            st.markdown("<hr style='margin: 28px 0 16px 0;'>", unsafe_allow_html=True)
            st.markdown("##### 📜 Recent Analyses in this Session")
            for idx, hist in enumerate(st.session_state.history[1:6], start=2):
                with st.expander(f"Message #{idx}: \"{hist['raw_text']}\" → {hist['emotion'].upper()} ({hist['confidence']:.1%}) | Entropy: {hist['entropy']:.2f}"):
                    st.write(f"**Cleaned Tokens:** `{hist['cleaned_text']}`")
                    st.write(f"**Top Emotion:** {hist['emotion']} ({hist['confidence']:.1%})")
                    st.write(f"**Predictive Entropy:** {hist['entropy']:.2f} / 1.00")


# =============================================================================
# VIEW 2: EMOTION LAB (MODEL PERFORMANCE, PIPELINE & METRICS)
# =============================================================================
elif nav_page == "📊 Emotion Lab":
    st.title("Emotion Lab")
    st.markdown(
        "<p style='color: #666666; font-size: 1.05rem; margin-top: -10px;'>"
        "Transparent evaluation results, model pipeline architecture, class distributions, and limitations "
        "of the TF-IDF + Logistic Regression emotion classifier trained on the MELD dataset."
        "</p>",
        unsafe_allow_html=True,
    )

    if eval_results:
        # High-level metrics loaded directly from saved evaluation results
        test_acc = eval_results.get("test_accuracy", 0.0)
        dev_acc = eval_results.get("dev_accuracy", 0.0)
        train_sz = eval_results.get("train_size", 9989)
        test_sz = eval_results.get("test_size", 2610)
        vocab_sz = eval_results.get("vocab_size", 5637)

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-value">{test_acc:.2%}</div>
                    <div class="metric-label">Test Accuracy (Held-out)</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with col2:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-value">{dev_acc:.2%}</div>
                    <div class="metric-label">Validation Accuracy</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with col3:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-value">{train_sz:,}</div>
                    <div class="metric-label">Training Utterances</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with col4:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-value">{vocab_sz:,}</div>
                    <div class="metric-label">TF-IDF Vocabulary Size</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)

        # ---------------------------------------------------------------------
        # Feature 6: Explanation of TF-IDF and Logistic Regression
        # ---------------------------------------------------------------------
        st.subheader("How the Pipeline Works: TF-IDF & Logistic Regression")
        st.markdown(
            """
            This project implements a classical Natural Language Processing (NLP) pipeline composed of two foundational components:
            """
        )

        pipe_col1, pipe_col2 = st.columns(2)
        with pipe_col1:
            st.markdown(
                """
                <div class="subtext-card">
                    <div style="font-size: 0.95rem; font-weight: 700; color: #A4161A; margin-bottom: 8px;">1. TF-IDF Feature Extraction</div>
                    <div style="font-size: 0.88rem; color: #252525; line-height: 1.5;">
                        <b>TF-IDF (Term Frequency – Inverse Document Frequency)</b> converts variable-length text into fixed numerical vectors:
                        <ul style="margin-top: 6px; padding-left: 18px; color: #252525;">
                            <li><b>Term Frequency (TF):</b> Measures how frequently a word appears in a specific utterance.</li>
                            <li><b>Inverse Document Frequency (IDF):</b> Downweights ubiquitous words across all dialogues, accentuating distinct emotional cues.</li>
                            <li><b>Preprocessing:</b> Words are lowercased, stripped of punctuation, filtered through English stopword lists, and reduced to grammatical stems via the Porter Stemmer.</li>
                            <li><b>Vocabulary:</b> Captures unigrams and bigrams up to <b>5,637 distinct features</b>.</li>
                        </ul>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with pipe_col2:
            st.markdown(
                """
                <div class="subtext-card">
                    <div style="font-size: 0.95rem; font-weight: 700; color: #A4161A; margin-bottom: 8px;">2. Multinomial Logistic Regression</div>
                    <div style="font-size: 0.88rem; color: #252525; line-height: 1.5;">
                        <b>Logistic Regression</b> is a linear classification model adapted for 7-class multi-class categorization via the softmax function:
                        <ul style="margin-top: 6px; padding-left: 18px; color: #252525;">
                            <li><b>Decision Boundaries:</b> Learns a linear weight for each TF-IDF feature per emotion class.</li>
                            <li><b>Probability Calibration:</b> Applies the softmax function to output normalized probability distributions summing to 100%.</li>
                            <li><b>Class Weighting:</b> Configured with <code>class_weight='balanced'</code> to prevent the dominant <i>neutral</i> class from eclipsing rare classes like <i>fear</i> and <i>disgust</i>.</li>
                            <li><b>Why Classical NLP?</b> Provides an interpretable, transparent, and reproducible baseline without the opaque black-box characteristics of massive neural networks.</li>
                        </ul>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)

        # ---------------------------------------------------------------------
        # Feature 6: Classification Metrics Table
        # ---------------------------------------------------------------------
        st.subheader("Classification Metrics by Emotion Class")
        st.caption("Per-class performance on the held-out test split (2,610 utterances):")

        classes = eval_results.get("classes", [])
        precision = eval_results.get("precision", [])
        recall = eval_results.get("recall", [])
        f1 = eval_results.get("f1", [])
        support = eval_results.get("support", [])

        df_metrics = pd.DataFrame({
            "Emotion Category": [c.capitalize() for c in classes],
            "Precision": [f"{p:.2%}" for p in precision],
            "Recall": [f"{r:.2%}" for r in recall],
            "F1-Score": [f"{f:.2%}" for f in f1],
            "Test Support (Samples)": support,
        })

        st.dataframe(df_metrics, use_container_width=True, hide_index=True)

        st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)

        # ---------------------------------------------------------------------
        # Feature 6: Confusion Matrix & Distribution Plots (Clean on White)
        # ---------------------------------------------------------------------
        plot_col1, plot_col2 = st.columns(2)

        with plot_col1:
            st.subheader("Confusion Matrix")
            cm_image_path = Path("models/confusion_matrix.png")
            if cm_image_path.exists():
                st.image(str(cm_image_path), use_container_width=True, caption="Confusion Matrix on Held-Out Test Set")
            else:
                cm = eval_results.get("confusion_matrix")
                if cm is not None:
                    fig, ax = plt.subplots(figsize=(6, 5))
                    fig.patch.set_facecolor("#FFFFFF")
                    ax.set_facecolor("#FFFFFF")
                    sns.heatmap(
                        cm,
                        annot=True,
                        fmt="d",
                        cmap="Reds",
                        xticklabels=classes,
                        yticklabels=classes,
                        ax=ax,
                        cbar=False
                    )
                    ax.tick_params(colors="#252525")
                    ax.set_xlabel("Predicted Emotion", color="#252525")
                    ax.set_ylabel("True Emotion", color="#252525")
                    st.pyplot(fig)

        with plot_col2:
            st.subheader("Emotion Class Distribution")
            dist_image_path = Path("models/emotion_distribution.png")
            if dist_image_path.exists():
                st.image(str(dist_image_path), use_container_width=True, caption="MELD Emotion Split Distributions")
            else:
                test_dist = eval_results.get("test_distribution", {})
                if test_dist:
                    fig2, ax2 = plt.subplots(figsize=(6, 5))
                    fig2.patch.set_facecolor("#FFFFFF")
                    ax2.set_facecolor("#FFF8F8")
                    emotions_list = list(test_dist.keys())
                    counts_list = list(test_dist.values())
                    ax2.bar(emotions_list, counts_list, color="#D62839", edgecolor="#A4161A")
                    ax2.tick_params(colors="#252525", rotation=30)
                    ax2.set_ylabel("Number of Samples (Test)", color="#252525")
                    st.pyplot(fig2)

        st.markdown("<hr style='margin: 24px 0;'>", unsafe_allow_html=True)

        # ---------------------------------------------------------------------
        # Feature 6: Explanation of the Seven Emotion Categories
        # ---------------------------------------------------------------------
        st.subheader("The Seven MELD Emotion Categories")
        st.markdown(
            """
            The MELD dataset annotates dialogue utterances into seven discrete categories originally based on Ekman's universal emotion model:
            - **Joy**: Happiness, amusement, triumph, or positive social bonding.
            - **Neutral**: Purely factual, logistical, or communicative statements without distinct emotional charge.
            - **Surprise**: Sudden astonishment, startling news, or unanticipated occurrences.
            - **Anger**: Irritation, frustration, confrontation, grievance, or resentment.
            - **Sadness**: Disappointment, sorrow, fatigue, or low-energy emotional states.
            - **Disgust**: Revulsion, aversion, moral disapproval, or sensory distaste.
            - **Fear**: Anxiety, concern, nervousness, or apprehension about negative outcomes.
            """
        )

        st.markdown("<hr style='margin: 24px 0;'>", unsafe_allow_html=True)

        # ---------------------------------------------------------------------
        # Feature 6: Honest Discussion of Project Limitations
        # ---------------------------------------------------------------------
        st.subheader("Project Limitations & Scientific Reality")
        st.markdown(
            """
            As an academic NLP project, understanding where and why text classification models struggle is crucial:

            1. **Modest Test Accuracy (35.98%):**
               While 35.98% is substantially higher than the 14.3% random baseline for a 7-class problem, roughly 64% of test utterances are categorized differently from the human annotators' labels. This modest performance reflects the immense challenge of classifying isolated text snippets without acoustic or visual prosody.

            2. **Severe Class Imbalance:**
               `Neutral` constitutes nearly half of the entire dataset (48.1% of test samples). Conversely, `fear` (1.9%) and `disgust` (2.6%) represent tiny fractions. Even with balanced inverse-frequency weighting, rare classes suffer from lower precision and recall due to minimal training variety.

            3. **Surface Text vs. Hidden Intent:**
               TF-IDF and Logistic Regression map n-gram word co-occurrences. They cannot understand human subtext, psychological defense mechanisms, or conversational history. Sarcasm (e.g., saying *"Oh, great"* when a mistake occurs) is commonly classified as `joy` because the literal words are positive.

            4. **Domain & Scripting Bias:**
               The MELD dataset was extracted from the television series *Friends*. Utterances reflect comedic, scripted television dialogue written for entertainment, which differs significantly in tone, pacing, and vocabulary from real-world digital texting, professional email exchanges, or private disputes.
            """
        )
    else:
        st.info("Evaluation results file `models/evaluation_results.pkl` was not found.")

# -----------------------------------------------------------------------------
# End of app.py
# -----------------------------------------------------------------------------
