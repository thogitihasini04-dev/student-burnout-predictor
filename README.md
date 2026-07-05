# AI-Powered Student Burnout & Wellness Prediction System
### Domain: Data Science | Machine Learning | Natural Language Processing (NLP)

---

## 📌 What is this project?

This is a **mini project** that uses **Artificial Intelligence** to predict whether a student
is at risk of academic burnout. It analyzes:
- 📚 Academic behavior (study hours, sleep, attendance, etc.)
- 💬 Emotional state (using NLP sentiment analysis on journal text)
- 📊 A custom Stress Index score (0–100)

The system predicts burnout risk as **Low / Medium / High** and gives
**personalized recommendations** to help the student.

---

## 🎯 Project Title
> **"AI-Powered Student Burnout and Wellness Prediction System
>   Using Machine Learning and Sentiment Analysis"**

---

## 🏷️ Domain
| Level | Domain |
|-------|--------|
| Primary | Data Science & Artificial Intelligence |
| Secondary | Machine Learning |
| Secondary | Natural Language Processing (NLP) |
| Application | Student Mental Health & Academic Wellness |

---

## 📁 Project Structure
```
burnout_system/
├── app.py                   # Main Gradio UI — run this file
├── model.py                 # Random Forest ML model logic
├── sentiment.py             # TextBlob + VADER sentiment analysis
├── eda.py                   # EDA charts & statistical analysis
├── utils.py                 # Stress index formula & recommendations
├── data/
│   └── dataset.csv          # Sample student dataset (auto-generated)
├── prediction_history.csv   # Auto-created when you make predictions
├── requirements.txt         # All Python packages needed
└── README.md                # This file
```

---

## ⚙️ How It Works (Simple Explanation)

```
Student fills form
       ↓
[ Academic Data ]          [ Journal Text ]
study hrs, sleep,          "I feel stressed
attendance, mood           and overwhelmed..."
       ↓                          ↓
[ Random Forest ]          [ TextBlob + VADER ]
  ML Prediction              NLP Sentiment
       ↓                          ↓
  Burnout Risk              Emotional State
Low / Medium / High      Positive/Neutral/Negative
       ↓                          ↓
         [ Stress Index 0-100 ]
                  ↓
     [ Personalized Recommendations ]
                  ↓
        [ Save to History CSV ]
```

---

## 🚀 Features

| # | Feature | What it does |
|---|---------|--------------|
| 1 | 🔴 Burnout Risk Prediction | Random Forest predicts Low/Medium/High |
| 2 | 📊 Stress Index | Custom formula gives a 0–100 stress score |
| 3 | 💬 Sentiment Analysis | Analyzes your journal text for emotions |
| 4 | 📈 EDA Dashboard | Explores dataset with charts & statistics |
| 5 | 💡 Recommendations | Personalized tips based on your risk level |
| 6 | 📋 Prediction History | Saves all predictions to a CSV file |
| 7 | 📉 Trend Graphs | Visualize your stress over time |
| 8 | ⬇️ CSV Export | Download your history anytime |

---

## 🛠️ Tech Stack

| Purpose | Library |
|---------|---------|
| UI & Deployment | Gradio |
| ML Model | scikit-learn (Random Forest) |
| Sentiment Analysis | TextBlob + VADER |
| Data Analysis | Pandas + NumPy |
| Visualizations | Plotly + Matplotlib + Seaborn |
| Language | Python 3.9+ |

---

## 📥 Input Features

| Feature | Type | Range | Description |
|---------|------|-------|-------------|
| Study Hours | Float | 1–14 hrs | Daily study duration |
| Sleep Hours | Float | 3–10 hrs | Nightly sleep duration |
| Assignment Delays | Integer | 0–10 | Late submissions per month |
| Attendance | Float | 40–100% | Class attendance rate |
| Mood Score | Integer | 1–10 | Self-rated emotional mood |
| Journal Text | String | Optional | Free-form emotional feedback |

---

## 📤 Output

| Output | Description |
|--------|-------------|
| Burnout Risk | Low / Medium / High |
| Confidence % | How confident the model is |
| Stress Index | Score from 0 to 100 |
| Emotional State | Positive / Neutral / Negative |
| Recommendations | Personalized action tips |

---

## 📐 Stress Index Formula
```
Raw Score = (study_hours x 3) + (delays x 5) - (sleep x 4) - (mood x 2) - (attendance x 0.1)
Stress Index = clamp((Raw + 20) / 60 x 100,  min=0,  max=100)
```
- Higher study hours + more delays = higher stress
- More sleep + better mood + attendance = lower stress

---
## 💡 Recommendations Logic

| Risk Level | Key Recommendations |
|------------|-------------------|
| Low  | Maintain balance, keep up good habits |
| Medium | Reduce study load, improve sleep, take breaks |
| High | Seek counselor, reduce workload, self-care urgently |
| Negative Sentiment | Extra mental health support message added |

---





