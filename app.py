import os
import io
import time
import pandas as pd
import streamlit as st
import torch
import plotly.express as px
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.nn.functional import softmax

# ----------------------------- Page setup -----------------------------
st.set_page_config(
    page_title="EmotionAI | Emotion Detection",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

MODEL_DIR = os.path.join(os.path.dirname(__file__), "fine_tuned_emotion_model")

EMOTION_META = {
    "sadness":  {"emoji": "😢", "color": "#6485d8", "description": "Low mood, loss, or disappointment"},
    "joy":      {"emoji": "😊", "color": "#e9a23b", "description": "Happiness, excitement, or pleasure"},
    "love":     {"emoji": "❤️", "color": "#e36a91", "description": "Affection, care, or connection"},
    "anger":    {"emoji": "😠", "color": "#e06b5f", "description": "Frustration, irritation, or anger"},
    "fear":     {"emoji": "😨", "color": "#9b7bd4", "description": "Worry, anxiety, or uncertainty"},
    "surprise": {"emoji": "😲", "color": "#49a9a1", "description": "Unexpected events or amazement"},
}

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');
:root { --ink:#182033; --muted:#718096; --line:#e8edf5; --panel:#ffffff; --accent:#6558d3; }
html, body, [class*="css"] { font-family:'DM Sans',sans-serif; }
.stApp { background:linear-gradient(135deg,#f6f7ff 0%,#f8fbff 48%,#f7f9fc 100%); color:var(--ink); }
.block-container { padding-top:2rem; padding-bottom:3rem; max-width:1400px; }
h1,h2,h3 { font-family:'Space Grotesk',sans-serif !important; color:#182033; }
.hero { padding:28px 32px; border-radius:24px; color:white; background:linear-gradient(115deg,#252b55 0%,#5145a8 58%,#796de0 100%); box-shadow:0 16px 38px rgba(64,62,137,.16); margin-bottom:22px; }
.hero h1 { color:white !important; font-size:2.35rem; margin:0 0 8px 0; }
.hero p { color:#e5e6ff; font-size:1rem; margin:0; }
.eyebrow { color:#c9caff; font-size:.75rem; font-weight:700; letter-spacing:.16em; text-transform:uppercase; margin-bottom:10px; }
.panel { background:white; border:1px solid #e9edf5; border-radius:18px; padding:20px 22px; box-shadow:0 7px 22px rgba(31,45,75,.045); }
.metric-label { color:#77839a; font-size:.82rem; font-weight:600; margin-bottom:6px; }
.metric-value { color:#1d2740; font-family:'Space Grotesk',sans-serif; font-size:1.75rem; font-weight:700; }
.small-muted { color:#7d879b; font-size:.84rem; }
.result-card { background:white; border:1px solid #e8edf5; border-radius:20px; padding:22px; box-shadow:0 8px 24px rgba(31,45,75,.05); }
.result-emoji { font-size:2.6rem; }
.result-name { font-family:'Space Grotesk',sans-serif; font-size:1.75rem; font-weight:700; text-transform:capitalize; }
.confidence { color:#5145a8; font-weight:700; font-size:1.15rem; }
.stButton>button { border-radius:11px; border:0; background:linear-gradient(110deg,#5548bd,#786be0); color:white; font-weight:700; padding:.62rem 1.1rem; }
.stButton>button:hover { color:white; border:0; background:linear-gradient(110deg,#4539a8,#6659cf); }
div[data-testid="stTextArea"] textarea { border-radius:12px; border:1px solid #dfe5ef; }
div[data-testid="stFileUploader"] { background:#fafbff; border:1px dashed #cbd3e2; border-radius:14px; padding:10px; }
hr { border-color:#e8edf5; }
footer {visibility:hidden;}
</style>
""", unsafe_allow_html=True)

# ----------------------------- Model loading -----------------------------
@st.cache_resource(show_spinner="Loading your fine-tuned DistilBERT model...")
def load_model():
    if not os.path.isdir(MODEL_DIR):
        raise FileNotFoundError(f"Model folder not found: {MODEL_DIR}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_DIR, local_files_only=True
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    return tokenizer, model, device

def get_labels(model):
    mapping = model.config.id2label or {}
    labels = []
    for i in range(model.config.num_labels):
        labels.append(str(mapping.get(i, mapping.get(str(i), f"label_{i}"))).lower())
    return labels

def predict_text(text, tokenizer, model, device):
    encoded = tokenizer(
        text, return_tensors="pt", truncation=True, max_length=128
    )
    encoded = {k: v.to(device) for k, v in encoded.items()}
    with torch.inference_mode():
        logits = model(**encoded).logits
        probs = softmax(logits, dim=-1)[0].detach().cpu().numpy()
    labels = get_labels(model)
    ranked = sorted(
        [{"emotion": labels[i], "probability": float(probs[i])} for i in range(len(labels))],
        key=lambda x: x["probability"], reverse=True
    )
    return ranked

# ----------------------------- Sidebar -----------------------------
with st.sidebar:
    st.markdown("## 🧠 EmotionAI")
    st.caption("Fine-tuned NLP classifier")
    st.markdown("---")
    page = st.radio(
        "WORKSPACE",
        ["Analyze text", "Batch analysis", "About the model"],
        label_visibility="collapsed"
    )
    st.markdown("---")
    st.markdown("**Model status**")
    try:
        tokenizer, model, device = load_model()
        st.success("Model loaded", icon="✅")
        st.caption(f"Running on: **{str(device).upper()}**")
        st.caption(f"Classes: **{model.config.num_labels} emotions**")
    except Exception as e:
        st.error("Could not load model")
        st.caption(str(e))
        st.stop()
    st.markdown("---")
    st.caption("Built with Streamlit · PyTorch · Hugging Face")

labels = get_labels(model)

# ----------------------------- Header -----------------------------
st.markdown("""
<div class="hero">
  <div class="eyebrow">NLP · TRANSFER LEARNING · TEXT CLASSIFICATION</div>
  <h1>Understand the emotion behind every sentence.</h1>
  <p>Analyze text with your fine-tuned DistilBERT model and explore its emotion probabilities.</p>
</div>
""", unsafe_allow_html=True)

# ----------------------------- Analyze text -----------------------------
if page == "Analyze text":
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown('<div class="panel"><div class="metric-label">MODEL</div><div class="metric-value">DistilBERT</div><div class="small-muted">Fine-tuned transformer</div></div>', unsafe_allow_html=True)
    with c2:
        st.markdown(f'<div class="panel"><div class="metric-label">EMOTION CLASSES</div><div class="metric-value">{len(labels)}</div><div class="small-muted">Six emotion categories</div></div>', unsafe_allow_html=True)
    with c3:
        st.markdown(f'<div class="panel"><div class="metric-label">COMPUTE DEVICE</div><div class="metric-value">{str(device).upper()}</div><div class="small-muted">Automatic CPU / GPU selection</div></div>', unsafe_allow_html=True)

    st.write("")
    left, right = st.columns([1.05, .95], gap="large")
    with left:
        st.subheader("✍️ Enter your text")
        sentence = st.text_area(
            "Text to analyze",
            placeholder="Example: I can't believe I finally achieved my dream!",
            height=175,
            label_visibility="collapsed",
            key="single_text"
        )
        sample_cols = st.columns(3)
        samples = [
            ("😊 Positive", "I am so happy that everything worked out today."),
            ("😠 Frustrated", "I am tired of dealing with the same problem again."),
            ("😨 Worried", "I am nervous about what will happen tomorrow.")
        ]
        for col, (label, sample) in zip(sample_cols, samples):
            with col:
                if st.button(label, use_container_width=True):
                    st.session_state["single_text"] = sample
                    st.rerun()
        analyze = st.button("Analyze emotion  →", type="primary", use_container_width=True)

    with right:
        st.subheader("📊 Prediction")
        if analyze:
            if not sentence.strip():
                st.warning("Enter a sentence before running the analysis.")
            else:
                with st.spinner("Analyzing emotional signals..."):
                    ranked = predict_text(sentence.strip(), tokenizer, model, device)
                best = ranked[0]
                meta = EMOTION_META.get(best["emotion"], {"emoji":"💬","color":"#6558d3","description":"Model prediction"})
                st.markdown(f"""
                <div class="result-card">
                  <div class="result-emoji">{meta['emoji']}</div>
                  <div class="small-muted">MOST LIKELY EMOTION</div>
                  <div class="result-name">{best['emotion']}</div>
                  <div class="confidence">{best['probability']*100:.2f}% confidence</div>
                  <p class="small-muted">{meta['description']}</p>
                </div>
                """, unsafe_allow_html=True)
                st.session_state["last_prediction"] = {
                    "text": sentence.strip(),
                    "emotion": best["emotion"],
                    "confidence": best["probability"]
                }
                st.markdown("#### Emotion probability breakdown")
                prob_df = pd.DataFrame(ranked)
                prob_df["Probability (%)"] = prob_df["probability"] * 100
                fig = px.bar(
                    prob_df.sort_values("Probability (%)"),
                    x="Probability (%)", y="emotion", orientation="h",
                    text=prob_df.sort_values("Probability (%)")["Probability (%)"].map(lambda x: f"{x:.1f}%"),
                    color="emotion",
                    color_discrete_map={k:v["color"] for k,v in EMOTION_META.items()},
                )
                fig.update_layout(
                    showlegend=False, height=300, margin=dict(l=0,r=10,t=10,b=0),
                    xaxis_title="Probability", yaxis_title="",
                    plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)"
                )
                fig.update_traces(textposition="outside", cliponaxis=False)
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.markdown('<div class="panel"><div style="font-size:2rem">🔎</div><h3>Ready when you are</h3><p class="small-muted">Enter text on the left and click Analyze emotion to see the model prediction and probability distribution.</p></div>', unsafe_allow_html=True)

    if "last_prediction" in st.session_state:
        st.markdown("---")
        st.subheader("Latest analysis")
        latest = st.session_state["last_prediction"]
        st.info(f"**Text:** {latest['text']}\n\n**Prediction:** {latest['emotion'].title()} · {latest['confidence']*100:.2f}%")

# ----------------------------- Batch analysis -----------------------------
elif page == "Batch analysis":
    st.subheader("📂 Analyze multiple sentences")
    st.write("Upload a CSV containing a text column. The app predicts an emotion for every row and lets you download the results.")
    uploaded = st.file_uploader("Upload CSV file", type=["csv"])
    if uploaded:
        try:
            df = pd.read_csv(uploaded)
            st.write("Data preview")
            st.dataframe(df.head(8), use_container_width=True)
            text_col = st.selectbox("Choose the column containing text", df.columns)
            if st.button("Run batch analysis", type="primary"):
                texts = df[text_col].fillna("").astype(str).tolist()
                results = []
                progress = st.progress(0)
                status = st.empty()
                for i, text in enumerate(texts):
                    if text.strip():
                        ranked = predict_text(text, tokenizer, model, device)
                        results.append({
                            "predicted_emotion": ranked[0]["emotion"],
                            "confidence": round(ranked[0]["probability"], 5),
                            "top_2_emotion": ranked[1]["emotion"] if len(ranked) > 1 else "",
                        })
                    else:
                        results.append({
                            "predicted_emotion": "empty",
                            "confidence": 0.0,
                            "top_2_emotion": "",
                        })
                    progress.progress((i + 1) / max(len(texts), 1))
                    status.caption(f"Processed {i+1} of {len(texts)} rows")
                output = df.copy()
                output = pd.concat([output, pd.DataFrame(results)], axis=1)
                st.session_state["batch_output"] = output
                st.success(f"Finished analyzing {len(texts)} rows.")
            if "batch_output" in st.session_state:
                output = st.session_state["batch_output"]
                st.dataframe(output, use_container_width=True)
                csv = output.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "⬇️ Download predictions as CSV",
                    data=csv,
                    file_name="emotion_predictions.csv",
                    mime="text/csv",
                    use_container_width=True
                )
                counts = output["predicted_emotion"].value_counts().reset_index()
                counts.columns = ["emotion", "count"]
                fig = px.pie(counts, names="emotion", values="count", hole=.55,
                             title="Predicted emotion distribution")
                st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Could not read or process the CSV: {e}")
    else:
        st.markdown('<div class="panel"><h3>Batch workflow</h3><p class="small-muted">1. Prepare a CSV file · 2. Upload it · 3. Select the text column · 4. Run predictions · 5. Download the enriched CSV.</p></div>', unsafe_allow_html=True)
        example = pd.DataFrame({"text": [
            "I am delighted with the results.",
            "This situation makes me furious.",
            "I am worried about my exam."
        ]})
        st.download_button(
            "Download sample CSV",
            example.to_csv(index=False).encode("utf-8"),
            "sample_emotion_texts.csv",
            "text/csv"
        )

# ----------------------------- About -----------------------------
else:
    st.subheader("About this project")
    st.markdown("""
    This application uses a **fine-tuned DistilBERT sequence-classification model**.
    The model was trained to classify text into six emotion categories.

    **Emotion labels**
    """)
    cols = st.columns(3)
    for i, label in enumerate(labels):
        meta = EMOTION_META.get(label, {"emoji":"💬","description":"Emotion category"})
        with cols[i % 3]:
            st.markdown(
                f'<div class="panel" style="margin-bottom:12px"><div style="font-size:1.8rem">{meta["emoji"]}</div><b style="text-transform:capitalize">{label}</b><div class="small-muted">{meta["description"]}</div></div>',
                unsafe_allow_html=True
            )
    st.markdown("---")
    st.subheader("How it works")
    st.markdown("""
    1. **Tokenization:** The sentence is converted into tokens using the DistilBERT tokenizer.
    2. **Inference:** The fine-tuned model produces a score for each emotion class.
    3. **Probabilities:** Softmax converts the scores into a probability distribution.
    4. **Prediction:** The class with the highest probability is displayed.

    **Important:** This is a text-classification model, not a psychological or clinical assessment.
    Predictions can be uncertain, especially for sarcasm, mixed emotions, short text, or language outside the training distribution.
    """)
    with st.expander("Model details"):
        st.write({
            "Architecture": model.config.model_type,
            "Number of labels": model.config.num_labels,
            "Labels": labels,
            "Maximum input length": 128,
            "Device": str(device),
            "Model directory": MODEL_DIR,
        })

st.markdown("---")
st.caption("EmotionAI · Academic mini project · Fine-tuned DistilBERT")
