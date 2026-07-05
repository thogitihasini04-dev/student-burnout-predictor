# AI-Based Student Burnout Prediction & Wellness System

An intelligent, interactive web application that predicts student burnout risk using **Machine Learning (Random Forest)** and analyzes emotional state using **NLP (TextBlob + VADER)**. Built with **Python and Streamlit**, this system provides early detection of academic burnout along with personalized wellness insights and recommendations.

---

## 📌 Project Overview

Student burnout is a growing concern caused by academic pressure, lack of sleep, and emotional stress. Traditional counselling systems are reactive and fail to detect early warning signs.

This project introduces an **AI-powered proactive wellness system** that:
- Predicts burnout risk (Low / Medium / High)
- Analyzes emotional sentiment from student feedback
- Generates personalized AI-driven insights
- Tracks historical trends for monitoring progress

---

## 🚀 Key Features

### 🎯 Burnout Prediction
- Uses **Random Forest Classifier**
- Inputs:
  - Study hours
  - Sleep hours
  - Assignment delays
  - Attendance percentage
  - Mood level
- Output: **Low / Medium / High Risk**

---

### 💬 Sentiment Analysis (NLP)
- Dual-engine system:
  - TextBlob (lexicon-based polarity)
  - VADER (rule-based sentiment)
- Combines both scores for accurate emotional understanding
- Detects:
  - Positive
  - Neutral
  - Negative sentiment

---
## ⚙️ How It Works 

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
         [ Stress Index]
                  ↓
     [ Personalized Recommendations ]
                  ↓
        [ Save to History CSV ]
```

---


### 🤖 AI Insights & Recommendations
- Personalized feedback based on:
  - Risk level
  - Sentiment score
- Includes:
  - Study-life balance tips
  - Sleep improvement guidance
  - Stress reduction techniques

---

### 📈 History Tracking
- Stores all predictions in a **CSV file**
- Enables:
  - Trend analysis
  - Progress tracking
  - Visual charts of student wellness

---

### 🌐 Interactive Dashboard
- Built with **Streamlit**
- Features:
  - Clean UI with dark theme
  - Sidebar inputs
  - Real-time prediction updates
  - Sentiment result cards
  - Trend visualization charts

---

## Machine Learning & NLP Models

### 🔷 Random Forest Classifier
- Ensemble learning method (bagging)
- Improves accuracy and reduces overfitting
- Uses majority voting across decision trees

---

### 🔷 Sentiment Analysis Pipeline
- **TextBlob** → polarity score (-1 to +1)
- **VADER** → compound sentiment score (-1 to +1)
- Final score = average of both methods
- Handles disagreement → classified as Neutral

---

---

## 🛠️ Technologies Used

| Technology      | Purpose |
|----------------|--------|
| Python         | Core programming language |
| Streamlit      | Web dashboard UI |
| scikit-learn   | Machine Learning (Random Forest) |
| Pandas         | Data handling & CSV management |
| NumPy          | Numerical computations |
| TextBlob       | Sentiment analysis |
| VADER          | Social-media text sentiment analysis |
| Pickle / Joblib| Model serialization |
| CSV            | Data storage |

---



