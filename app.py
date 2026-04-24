import streamlit as st
import torch
import shap
import numpy as np
import plotly.graph_objects as go
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import warnings
warnings.filterwarnings("ignore")

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Legal Risk Scorer",
    page_icon="⚖️",
    layout="wide"
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Mono:wght@300;400;500&display=swap');

    :root {
        --bg: #0d0d0d;
        --surface: #161616;
        --border: #2a2a2a;
        --accent: #c9a84c;
        --accent-dim: #8a6f2e;
        --danger: #e05252;
        --safe: #52a875;
        --text: #e8e4dc;
        --muted: #6b6b6b;
    }

    html, body, .stApp {
        background-color: var(--bg) !important;
        color: var(--text) !important;
        font-family: 'DM Mono', monospace;
    }

    h1, h2, h3 {
        font-family: 'DM Serif Display', serif !important;
        color: var(--accent) !important;
        letter-spacing: -0.02em;
    }

    .stTextArea textarea {
        background: var(--surface) !important;
        border: 1px solid var(--border) !important;
        border-radius: 4px !important;
        color: var(--text) !important;
        font-family: 'DM Mono', monospace !important;
        font-size: 13px !important;
    }

    .stButton > button {
        background: var(--accent) !important;
        color: #0d0d0d !important;
        border: none !important;
        border-radius: 2px !important;
        font-family: 'DM Mono', monospace !important;
        font-weight: 500 !important;
        letter-spacing: 0.05em !important;
        text-transform: uppercase !important;
        padding: 0.6rem 2rem !important;
    }

    .stButton > button:hover {
        background: #e8c46a !important;
    }

    .stTabs [data-baseweb="tab"] {
        font-family: 'DM Mono', monospace !important;
        color: var(--muted) !important;
        border-bottom: 2px solid transparent !important;
    }

    .stTabs [aria-selected="true"] {
        color: var(--accent) !important;
        border-bottom: 2px solid var(--accent) !important;
    }

    .risk-card {
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: 4px;
        padding: 1.5rem;
        margin: 1rem 0;
    }

    .high-risk { border-left: 4px solid var(--danger); }
    .low-risk  { border-left: 4px solid var(--safe); }

    .token-high { background: rgba(224,82,82,0.25); border-radius: 3px; padding: 1px 3px; }
    .token-low  { background: rgba(82,168,117,0.25); border-radius: 3px; padding: 1px 3px; }

    .stDataFrame { background: var(--surface) !important; }
    .stSpinner > div { border-top-color: var(--accent) !important; }

    hr { border-color: var(--border) !important; }

    .disclaimer {
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: 4px;
        padding: 0.75rem 1rem;
        font-size: 11px;
        color: var(--muted);
        margin-top: 2rem;
    }
</style>
""", unsafe_allow_html=True)

# ── Load Model ────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    model_path = "ankitpadhi04/legal-risk-scorer"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.eval()
    return tokenizer, model

@st.cache_resource
def load_explainer(_tokenizer, _model):
    def predict(texts):
        if isinstance(texts, np.ndarray):
            texts = texts.tolist()
        texts = [str(t) for t in texts]
        inputs = _tokenizer(texts, return_tensors="pt", truncation=True, padding=True, max_length=512)
        with torch.no_grad():
            outputs = _model(**inputs)
        return torch.softmax(outputs.logits, dim=1).numpy()
    return shap.Explainer(predict, _tokenizer, output_names=["Low Risk", "High Risk"])

# ── Helpers ───────────────────────────────────────────────────────────────────
CLAUSE_QUESTIONS = {
    "Non-Compete":               'Highlight the parts (if any) of this contract related to "Non-Compete" that should be reviewed by a lawyer.',
    "Uncapped Liability":        'Highlight the parts (if any) of this contract related to "Uncapped Liability" that should be reviewed by a lawyer.',
    "Termination For Convenience":'Highlight the parts (if any) of this contract related to "Termination For Convenience" that should be reviewed by a lawyer.',
    "IP Ownership Assignment":   'Highlight the parts (if any) of this contract related to "Ip Ownership Assignment" that should be reviewed by a lawyer.',
    "Anti-Assignment":           'Highlight the parts (if any) of this contract related to "Anti-Assignment" that should be reviewed by a lawyer.',
    "Liquidated Damages":        'Highlight the parts (if any) of this contract related to "Liquidated Damages" that should be reviewed by a lawyer.',
    "Change Of Control":         'Highlight the parts (if any) of this contract related to "Change Of Control" that should be reviewed by a lawyer.',
    "Auto-Renewal":              'Highlight the parts (if any) of this contract related to "Renewal Term" that should be reviewed by a lawyer.',
}

def predict_risk(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    probs = torch.softmax(outputs.logits, dim=1).numpy()[0]
    return probs[0], probs[1]  # low_risk, high_risk

def split_into_clauses(text):
    import re
    pattern = re.compile(
        r'(?=\n\s*\d+[\.\)]\s+[A-Z]'
        r'|\n\s*[A-Z]{3,}[\s:\.]'
        r'|\n\s*(?:Article|Section|WHEREAS|NOW THEREFORE)\s)',
        re.MULTILINE
    )
    chunks = pattern.split(text)
    return [c.strip() for c in chunks if len(c.strip().split()) >= 15]

def risk_gauge(score):
    color = "#e05252" if score > 0.5 else "#52a875"
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=round(score * 100, 1),
        number={"suffix": "%", "font": {"color": color, "size": 36, "family": "DM Mono"}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": "#2a2a2a"},
            "bar": {"color": color},
            "bgcolor": "#161616",
            "bordercolor": "#2a2a2a",
            "steps": [
                {"range": [0, 40],  "color": "#1a2e1e"},
                {"range": [40, 60], "color": "#2e2a14"},
                {"range": [60, 100],"color": "#2e1414"},
            ],
            "threshold": {"line": {"color": color, "width": 3}, "value": score * 100}
        },
        title={"text": "HIGH RISK PROBABILITY", "font": {"color": "#6b6b6b", "size": 12, "family": "DM Mono"}}
    ))
    fig.update_layout(
        height=250,
        paper_bgcolor="#0d0d0d",
        plot_bgcolor="#0d0d0d",
        font_color="#e8e4dc",
        margin=dict(t=40, b=10, l=30, r=30)
    )
    return fig

def shap_bar_chart(tokens, values):
    top = sorted(zip(tokens, values), key=lambda x: abs(x[1]), reverse=True)[:12]
    top_tokens, top_values = zip(*top)
    colors = ["#e05252" if v > 0 else "#52a875" for v in top_values]
    fig = go.Figure(go.Bar(
        x=list(top_values),
        y=list(top_tokens),
        orientation="h",
        marker_color=colors,
        marker_line_width=0,
    ))
    fig.update_layout(
        title={"text": "TOKEN INFLUENCE ON RISK SCORE", "font": {"size": 11, "color": "#6b6b6b", "family": "DM Mono"}},
        height=350,
        paper_bgcolor="#0d0d0d",
        plot_bgcolor="#161616",
        font={"color": "#e8e4dc", "family": "DM Mono", "size": 11},
        xaxis={"gridcolor": "#2a2a2a", "zerolinecolor": "#2a2a2a"},
        yaxis={"gridcolor": "#2a2a2a"},
        margin=dict(t=40, b=20, l=10, r=20)
    )
    return fig

# ── Main App ──────────────────────────────────────────────────────────────────
st.markdown("# ⚖️ Legal Risk Scorer")
st.markdown("*Fine-tuned Legal-BERT · CUAD Dataset · SHAP Explainability*")
st.markdown("---")

tokenizer, model = load_model()

tab1, tab2 = st.tabs(["SINGLE CLAUSE ANALYSIS", "FULL CONTRACT SCAN"])

# ── Tab 1 — Single Clause ─────────────────────────────────────────────────────
with tab1:
    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("### Clause Input")
        clause_type = st.selectbox("Clause Type", list(CLAUSE_QUESTIONS.keys()))
        clause_text = st.text_area(
            "Paste clause text",
            height=200,
            placeholder="Paste a single contract clause here..."
        )
        analyze_btn = st.button("ANALYZE CLAUSE", key="single")

    with col2:
        if analyze_btn and clause_text.strip():
            question = CLAUSE_QUESTIONS[clause_type]
            input_text = question + " [SEP] " + clause_text.strip()

            with st.spinner("Analyzing..."):
                low_risk, high_risk = predict_risk(input_text, tokenizer, model)

            verdict = "HIGH RISK" if high_risk > 0.5 else "LOW RISK"
            card_class = "high-risk" if high_risk > 0.5 else "low-risk"
            verdict_color = "#e05252" if high_risk > 0.5 else "#52a875"

            st.markdown(f"""
            <div class="risk-card {card_class}">
                <div style="font-size:11px;color:#6b6b6b;letter-spacing:0.1em;">VERDICT</div>
                <div style="font-size:28px;font-weight:500;color:{verdict_color};font-family:'DM Serif Display',serif;">{verdict}</div>
                <div style="font-size:12px;color:#6b6b6b;margin-top:4px;">Clause Type: {clause_type}</div>
            </div>
            """, unsafe_allow_html=True)

            st.plotly_chart(risk_gauge(high_risk), use_container_width=True)

            # SHAP
            with st.spinner("Generating SHAP explanation..."):
                explainer = load_explainer(tokenizer, model)
                shap_values = explainer([input_text], fixed_context=1)
                tokens = shap_values.data[0]
                values = shap_values.values[0, :, 1]

            st.plotly_chart(shap_bar_chart(tokens, values), use_container_width=True)

        elif analyze_btn:
            st.warning("Please paste a clause first.")

# ── Tab 2 — Full Contract ─────────────────────────────────────────────────────
with tab2:
    st.markdown("### Full Contract Scan")
    contract_text = st.text_area(
        "Paste full contract text",
        height=300,
        placeholder="Paste your full contract here. The app will split it into clauses and scan each one..."
    )
    scan_btn = st.button("SCAN CONTRACT", key="full")

    if scan_btn and contract_text.strip():
        clauses = split_into_clauses(contract_text)

        if not clauses:
            st.warning("Could not detect clause boundaries. Try pasting a longer contract with numbered sections.")
        else:
            st.markdown(f"*Detected {len(clauses)} clauses — scanning...*")
            results = []

            progress = st.progress(0)
            for i, clause in enumerate(clauses):
                # Check against all clause types, take max risk score
                max_risk = 0
                riskiest_type = "General"
                for ctype, question in CLAUSE_QUESTIONS.items():
                    input_text = question + " [SEP] " + clause[:800]
                    _, high_risk = predict_risk(input_text, tokenizer, model)
                    if high_risk > max_risk:
                        max_risk = high_risk
                        riskiest_type = ctype

                results.append({
                    "Clause Preview": clause[:120] + "...",
                    "Riskiest Type": riskiest_type,
                    "Risk Score": round(max_risk * 100, 1),
                    "Verdict": "🔴 High Risk" if max_risk > 0.5 else "🟢 Low Risk"
                })
                progress.progress((i + 1) / len(clauses))

            import pandas as pd
            results_df = pd.DataFrame(results).sort_values("Risk Score", ascending=False)

            high_count = sum(1 for r in results if r["Risk Score"] > 50)
            overall = "HIGH RISK" if high_count > len(results) * 0.3 else "LOW RISK"
            overall_color = "#e05252" if overall == "HIGH RISK" else "#52a875"

            st.markdown(f"""
            <div class="risk-card {'high-risk' if overall == 'HIGH RISK' else 'low-risk'}">
                <div style="font-size:11px;color:#6b6b6b;letter-spacing:0.1em;">OVERALL CONTRACT RISK</div>
                <div style="font-size:24px;font-weight:500;color:{overall_color};font-family:'DM Serif Display',serif;">{overall}</div>
                <div style="font-size:12px;color:#6b6b6b;margin-top:4px;">{high_count} of {len(results)} clauses flagged as high risk</div>
            </div>
            """, unsafe_allow_html=True)

            st.dataframe(results_df, use_container_width=True, hide_index=True)

    elif scan_btn:
        st.warning("Please paste contract text first.")

# ── Disclaimer ────────────────────────────────────────────────────────────────
st.markdown("""
<div class="disclaimer">
    ⚠️ This tool is for educational and informational purposes only. It does not constitute legal advice.
    Always consult a qualified attorney before making decisions based on contract analysis.
</div>
""", unsafe_allow_html=True)