# app.py — AI-Based Student Burnout Prediction & Wellness Dashboard
# ─────────────────────────────────────────────────────────────────
import os
import pickle
import base64
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.ensemble import RandomForestClassifier

# ══════════════════════════════════════════════════════════════════
# PAGE CONFIG  ← must be the very first Streamlit call
# ══════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Student Burnout Predictor",
    
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════════════
# CONSTANTS
# ══════════════════════════════════════════════════════════════════
MODEL_PATH   = "model/burnout_model.pkl"
OUTPUT_DIR   = "output"
LOG_FILE     = os.path.join(OUTPUT_DIR, "prediction_history.csv")
LOG_SEP      = ","

RISK_CONFIG = {
    "High":   {"emoji": "🚨", "color": "#FF5252", "label": "HIGH RISK"},
    "Medium": {"emoji": "⚠️",  "color": "#FFD740", "label": "MODERATE RISK"},
    "Low":    {"emoji": "✅",  "color": "#69F0AE", "label": "LOW RISK"},
}

# ══════════════════════════════════════════════════════════════════
# CUSTOM CSS
# ══════════════════════════════════════════════════════════════════
st.markdown("""
<style>
/* ── Base ── */
html, body, [data-testid="stAppViewContainer"] {
    background-color: #0D1117;
    color: #E6EDF3;
    font-family: 'Segoe UI', sans-serif;
}

/* ── Hide Streamlit branding ── */
#MainMenu, footer { visibility: hidden; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background-color: #161B22;
    border-right: 1px solid #30363D;
}

/* ── Metric cards ── */
[data-testid="metric-container"] {
    background: #161B22;
    border: 1px solid #30363D;
    border-radius: 12px;
    padding: 16px;
}

/* ── Section headers ── */
.section-header {
    font-size: 1.1rem;
    font-weight: 600;
    color: #8B949E;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin: 1.5rem 0 0.75rem;
    border-bottom: 1px solid #21262D;
    padding-bottom: 6px;
}

/* ── Result card ── */
.result-card {
    background: #161B22;
    border: 1px solid #30363D;
    border-radius: 16px;
    padding: 24px 28px;
    margin: 12px 0;
}
.result-card h2 { margin: 0 0 4px; font-size: 1.6rem; }
.result-card p  { margin: 0; color: #8B949E; font-size: 0.9rem; }

/* ── Tip bullet ── */
.tip-item {
    display: flex;
    align-items: flex-start;
    gap: 10px;
    padding: 8px 0;
    border-bottom: 1px solid #21262D;
    font-size: 0.95rem;
    color: #C9D1D9;
}
.tip-item:last-child { border-bottom: none; }

/* ── Sentiment card ── */
.sentiment-card {
    background: #161B22;
    border-radius: 12px;
    padding: 18px 22px;
    margin-top: 12px;
}

/* ── Primary button ── */
div.stButton > button {
    background: linear-gradient(135deg, #6E40C9, #8957E5);
    color: #fff;
    border: none;
    border-radius: 10px;
    padding: 12px 28px;
    font-size: 1rem;
    font-weight: 600;
    width: 100%;
    transition: opacity 0.2s;
}
div.stButton > button:hover { opacity: 0.88; }

/* ── Expander ── */
details { background: #161B22; border-radius: 10px; padding: 4px 12px; }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════
# BACKGROUND IMAGE  (optional — skipped gracefully if missing)
# ══════════════════════════════════════════════════════════════════
def set_background(path: str) -> None:
    if not os.path.exists(path):
        return
    with open(path, "rb") as f:
        data = base64.b64encode(f.read()).decode()
    st.markdown(f"""
    <style>
    [data-testid="stAppViewContainer"] {{
        background-image: url("data:image/jpeg;base64,{data}");
        background-size: cover;
        background-attachment: fixed;
    }}
    </style>
    """, unsafe_allow_html=True)

set_background("assets/bg.jpeg")

# ══════════════════════════════════════════════════════════════════
# CACHED RESOURCES
# ══════════════════════════════════════════════════════════════════
@st.cache_resource
def load_model() -> RandomForestClassifier:
    """Load model from disk, or create a tiny demo model if missing."""
    if os.path.exists(MODEL_PATH):
        with open(MODEL_PATH, "rb") as f:
            return pickle.load(f)

    # Demo model — remove once you have real data
    X = np.array([
        [10, 2, 5, 50, 1],
        [1,  9, 0, 95, 5],
        [6,  6, 2, 85, 3],
        [8,  4, 4, 60, 2],
        [3,  8, 1, 90, 4],
    ])
    y = ["High", "Low", "Medium", "High", "Low"]
    clf = RandomForestClassifier(n_estimators=50, random_state=42)
    clf.fit(X, y)
    os.makedirs("model", exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(clf, f)
    return clf


@st.cache_resource
def load_vader() -> SentimentIntensityAnalyzer:
    return SentimentIntensityAnalyzer()


model = load_model()
vader = load_vader()

# ══════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════
def compute_sentiment(text: str) -> tuple[float, str, str]:
    """Return (score, label, hex_color)."""
    blob  = TextBlob(text).sentiment.polarity
    vader_score = vader.polarity_scores(text)["compound"]
    score = (blob + vader_score) / 2

    # Opposing signals → Neutral
    if (blob > 0 and vader_score < 0) or (blob < 0 and vader_score > 0):
        return score, "Neutral 😐", "#FFD740"

    if score >= 0.25:
        return score, "Positive 😊", "#69F0AE"
    if score >= -0.25:
        return score, "Neutral 😐", "#FFD740"
    return score, "Negative 😟", "#FF5252"


def compute_stress_index(study: int, sleep: int, delays: int, mood: int) -> int:
    return study * 2 + delays * 3 - sleep * 2 - mood * 3


def save_log(row: dict) -> None:
    import csv
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    write_header = not os.path.exists(LOG_FILE)
    df = pd.DataFrame([row])
    # QUOTE_MINIMAL: only quote fields that contain the separator or quotes
    # This keeps column headers clean (no extra wrapping quotes)
    df.to_csv(
        LOG_FILE,
        mode="a",
        header=write_header,
        index=False,
        sep=LOG_SEP,
        quoting=csv.QUOTE_MINIMAL,
        escapechar="\\",
    )


def load_history() -> pd.DataFrame | None:
    import csv
    if not os.path.exists(LOG_FILE):
        return None
    try:
        df = pd.read_csv(
            LOG_FILE,
            sep=LOG_SEP,
            quoting=csv.QUOTE_MINIMAL,
            escapechar="\\",
            on_bad_lines="skip",
        )
        # Strip accidental whitespace from column names
        df.columns = df.columns.str.strip()
        return df if not df.empty else None
    except Exception as e:
        st.error(f"Could not read history: {e}")
        return None


def tip(icon: str, text: str) -> str:
    return f'<div class="tip-item"><span>{icon}</span><span>{text}</span></div>'


# ══════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("##  Student Wellness")
    st.markdown("---")

    student_name = st.text_input("👤 Your name", placeholder="e.g. Ananya")
    year = st.selectbox(
        "📘 Academic Year",
        ["1st Year", "2nd Year", "3rd Year", "4th Year"],
    )

    st.markdown("---")
    st.markdown("""
    **What this system does**
    - 🤖 ML burnout risk prediction
    - 💬 Dual-engine sentiment analysis
    - 📊 Real-time stress index
    - 🌿 Personalised wellness tips
    - 📈 Session history tracking
    """)

    st.markdown("---")
    st.caption("AI-Based Student Wellness System · v1.0")


# 🚨 BLOCK APP IF NAME NOT ENTERED
if not student_name:
    st.info("👋 Welcome! Please enter your name in the sidebar to start")
    st.stop()

# ══════════════════════════════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════════════════════════════
st.markdown("""
<h1 style='text-align:center; color:#8957E5; margin-bottom:4px;'>
   Student Burnout & Wellness Predictor
</h1>
<p style='text-align:center; color:#8B949E; margin-bottom:1.5rem;'>
  AI-powered burnout detection · Sentiment analysis · Personalised insights
</p>
""", unsafe_allow_html=True)
st.divider()

# ══════════════════════════════════════════════════════════════════
# INPUT SECTION
# ══════════════════════════════════════════════════════════════════
st.markdown('<p class="section-header">📝 Academic & Lifestyle Inputs</p>', unsafe_allow_html=True)

col1, col2 = st.columns(2, gap="large")

with col1:
    study_hours      = st.slider("📚 Study Hours per Day",    0, 12, 6)
    sleep_hours      = st.slider("😴 Sleep Hours per Day",    1, 10, 7)
    attendance       = st.slider("🎓 Attendance (%)",        30, 100, 85)

with col2:
    assignment_delays = st.number_input("📝 Assignment Delays (count)", 0, 10, 1, step=1)
    mood              = st.slider("😊 Mood Level  (1 = Low · 5 = High)", 1, 5, 3)

    # Quick overview metrics
    stress_index = compute_stress_index(study_hours, sleep_hours, assignment_delays, mood)
    st.metric("⚡ Stress Index (live)", stress_index,
              help="study×2 + delays×3 − sleep×2 − mood×3")

st.divider()

# ══════════════════════════════════════════════════════════════════
# SENTIMENT SECTION
# ══════════════════════════════════════════════════════════════════
st.markdown('<p class="section-header">💬 Emotional Feedback</p>', unsafe_allow_html=True)

feedback = st.text_area(
    "How do you feel about your studies recently?",
    placeholder="e.g. I feel overwhelmed with exams and haven't slept well…",
    height=100,
)

sentiment_score  = 0.0
sentiment_label  = "Neutral 😐"
sentiment_color  = "#FFD740"

if feedback.strip():
    sentiment_score, sentiment_label, sentiment_color = compute_sentiment(feedback)

    c1, c2 = st.columns([2, 1])
    with c1:
        st.markdown(f"""
        <div class="sentiment-card" style="border-left:5px solid {sentiment_color};">
            <div style="font-size:0.8rem;color:#8B949E;text-transform:uppercase;
                        letter-spacing:0.06em;">Emotional state</div>
            <div style="font-size:1.5rem;font-weight:700;color:{sentiment_color};
                        margin:6px 0 2px;">{sentiment_label}</div>
            <div style="color:#8B949E;font-size:0.85rem;">
                TextBlob + VADER combined analysis
            </div>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.metric("Combined Score", f"{sentiment_score:+.2f}",
                  help="Ranges from −1.0 (very negative) to +1.0 (very positive)")

st.divider()

# ══════════════════════════════════════════════════════════════════
# PREDICT BUTTON
# ══════════════════════════════════════════════════════════════════
predict_clicked = st.button("🔍 Predict My Burnout Risk")

if predict_clicked:
    input_data   = np.array([[study_hours, sleep_hours, assignment_delays, attendance, mood]])
    prediction   = model.predict(input_data)[0]
    proba        = model.predict_proba(input_data)[0]
    burnout_pct  = int(max(proba) * 100)
    cfg          = RISK_CONFIG.get(prediction, RISK_CONFIG["Medium"])

    # ── Result banner ──────────────────────────────────────────
    st.markdown('<p class="section-header">📊 Prediction Results</p>', unsafe_allow_html=True)

    r1, r2, r3 = st.columns(3)
    r1.metric("🔥 Burnout Risk",   f"{cfg['emoji']} {prediction}")
    r2.metric("📊 Confidence",     f"{burnout_pct}%")
    r3.metric("⚡ Stress Index",   stress_index)

    st.markdown(f"""
    <div class="result-card" style="border-left:5px solid {cfg['color']};">
        <h2 style="color:{cfg['color']};">{cfg['emoji']} {cfg['label']}</h2>
        <p>Based on your study load, sleep, attendance, mood and emotional feedback.</p>
    </div>
    """, unsafe_allow_html=True)

    # ── AI Insights ────────────────────────────────────────────
    st.markdown('<p class="section-header">🧠 AI Insights</p>', unsafe_allow_html=True)

    insights_html = ""
    if study_hours > 9 and sleep_hours < 6:
        insights_html += tip("🔴", "High study load + low sleep is significantly raising your burnout risk.")
    if assignment_delays >= 3:
        insights_html += tip("🔴", "Frequent assignment delays suggest mounting academic pressure.")
    if sentiment_label.startswith("Negative"):
        insights_html += tip("🔴", "Negative emotional feedback strongly correlates with burnout.")
    if attendance < 75:
        insights_html += tip("🟡", "Low attendance may indicate disengagement or chronic stress.")
    if study_hours <= 8 and sleep_hours >= 7:
        insights_html += tip("🟢", "Balanced study hours and good sleep — healthy habits detected.")
    if assignment_delays == 0:
        insights_html += tip("🟢", "Zero assignment delays show great academic discipline.")
    if sentiment_label.startswith("Positive"):
        insights_html += tip("🟢", "Positive emotional state reflects strong mental well-being.")
    if attendance >= 85:
        insights_html += tip("🟢", "High attendance shows consistent engagement and reliability.")
    if not insights_html:
        insights_html += tip("📌", "Your inputs are within an average range — keep monitoring regularly.")

    st.markdown(insights_html, unsafe_allow_html=True)

    # ── Personalised Recommendations ───────────────────────────
    st.markdown('<p class="section-header">🔔 Recommendations</p>', unsafe_allow_html=True)

    if prediction == "High":
        recs = [
            ("🛑", "Reduce academic overload — drop non-essential tasks immediately."),
            ("😴", "Prioritise 7–8 hours of sleep; set a fixed sleep schedule."),
            ("🧘", "Practice mindfulness, yoga, or deep breathing for 10 min daily."),
            ("📅", "Break large tasks into small daily goals to avoid overwhelm."),
            ("🤝", "Reach out to a mentor, counsellor, or trusted peer today."),
        ]
    elif prediction == "Medium":
        recs = [
            ("📋", "Use time-blocking to spread your workload more evenly."),
            ("⏱️", "Try the Pomodoro technique — 25 min focus, 5 min break."),
            ("📵", "Limit social media to set windows; avoid doom-scrolling late at night."),
            ("🚶", "Add a 20-minute walk or light exercise to your daily routine."),
        ]
    else:
        recs = [
            ("🌟", "Keep up your great habits — consistency is your superpower."),
            ("📓", "Keep a weekly wellness journal to catch early warning signs."),
            ("🎯", "Set one small stretch goal each week to stay motivated."),
            ("💬", "Check in with friends regularly — social support is protective."),
        ]

    recs_html = "".join(tip(icon, text) for icon, text in recs)
    st.markdown(recs_html, unsafe_allow_html=True)

    # ── Wellness Tips (sentiment-driven) ───────────────────────
    st.markdown('<p class="section-header">🌿 Wellness Tips</p>', unsafe_allow_html=True)

    if sentiment_score < -0.3:
        wellness = [
            ("🫁", "Pause and do 4-7-8 breathing: inhale 4s, hold 7s, exhale 8s."),
            ("✍️", "Express feelings through journaling — even 5 minutes helps."),
            ("📵", "Reduce screen time, especially social media, for the next 48 hours."),
            ("🤝", "Talk to someone you trust — you don't have to carry this alone."),
        ]
    elif sentiment_score < 0:
        wellness = [
            ("📝", "Plan tomorrow the night before — clarity reduces morning anxiety."),
            ("🛌", "Aim to sleep and wake at the same time every day."),
            ("🚶", "A 15-minute walk outside can reset your mood noticeably."),
            ("📵", "Limit overthinking time — schedule a 'worry window' of 10 minutes."),
        ]
    else:
        wellness = [
            ("✅", "Keep doing what's working — your routine is effective."),
            ("📊", "Do a weekly self-check to catch any creeping stress early."),
            ("🎨", "Protect time for hobbies — creativity replenishes focus."),
            ("💪", "Celebrate small wins; positive reinforcement sustains motivation."),
        ]

    wellness_html = "".join(tip(icon, text) for icon, text in wellness)
    st.markdown(wellness_html, unsafe_allow_html=True)

    # ── Save to log ────────────────────────────────────────────
    save_log({
        "Time":            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Name":            student_name or "Anonymous",
        "Year":            year,
        "Study Hours":     study_hours,
        "Sleep Hours":     sleep_hours,
        "Delays":          assignment_delays,
        "Attendance":      attendance,
        "Mood":            mood,
        "Sentiment Score": round(sentiment_score, 3),
        "Sentiment":       sentiment_label,
        "Stress Index":    stress_index,
        "Burnout Risk":    prediction,
        "Confidence (%)":  burnout_pct,
    })
    st.success("✅ Prediction saved to history.", icon="💾")

# ══════════════════════════════════════════════════════════════════
# HISTORY SECTION
# ══════════════════════════════════════════════════════════════════
st.divider()
st.markdown('<p class="section-header">📂 Prediction History</p>', unsafe_allow_html=True)

if st.checkbox("Show prediction history"):
    history = load_history()

    if history is None:
        st.info("No history yet — run a prediction first.")
    else:
        # Normalize all column names: strip spaces and quotes
        history.columns = (
            history.columns
            .str.strip()
            .str.strip('"')
            .str.strip("'")
        )

        # Show actual columns in an expander for debugging
        with st.expander("🔍 Debug — column names in CSV", expanded=False):
            st.code(history.columns.tolist())
            st.dataframe(history.head(2))

        # Use whichever name column actually exists
        name_col = next(
            (c for c in history.columns if c.lower() == "name"),
            None
        )

        if name_col is None:
            st.error("❌ 'Name' column not found. Delete `output/prediction_history.csv` and run a fresh prediction.")
            st.stop()

        names    = sorted(history[name_col].dropna().astype(str).unique())
        selected = st.selectbox("Filter by student", ["All"] + list(names))

        view = history if selected == "All" else history[history[name_col] == selected]

        st.dataframe(
            view.sort_values("Time", ascending=False).reset_index(drop=True),
            use_container_width=True,
        )

        if selected != "All" and len(view) > 1:
            st.markdown(f"**📈 Trend for {selected}**")
            trend_map = {
                "Study Hours":     "Study Hours",
                "Sleep Hours":     "Sleep Hours",
                "Mood":            "Mood",
                "Stress Index":    "Stress Index",
                "Sentiment Score": "Sentiment Score",
            }
            choice = st.selectbox("Select metric", list(trend_map.keys()))
            col    = trend_map[choice]
            if col in view.columns:
                st.line_chart(view[col].reset_index(drop=True))
                st.metric(
                    f"Latest {choice}",
                    round(float(view[col].iloc[-1]), 2),
                    delta=round(float(view[col].iloc[-1]) - float(view[col].iloc[-2]), 2)
                    if len(view) >= 2 else None,
                )
            else:
                st.warning(f"Column '{col}' not found. Available: {view.columns.tolist()}")
# ══════════════════════════════════════════════════════════════════
# FOOTER
# ══════════════════════════════════════════════════════════════════
st.divider()
with st.expander("🔮 Planned future enhancements"):
    st.markdown("""
    - 📱 Mobile app (Flutter)
    - ⌚ Wearable device data integration
    - 🤖 AI mental-health chatbot
    - 📧 Automated counsellor email alerts
    - 🏫 Institution-wide analytics dashboard
    """)

st.caption("🧠 AI-Based Student Wellness System · Streamlit + scikit-learn + NLP")