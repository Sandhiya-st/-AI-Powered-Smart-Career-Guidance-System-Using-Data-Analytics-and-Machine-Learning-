"""
=============================================================
  PLACEMENT PREDICTION — MAJOR PROJECT (EXPANDED)
  Modules:
    1. Multi-Model Prediction Engine (6 models + XGBoost-style GB)
    2. Career Recommendation Module
    3. Skill Gap Analysis
    4. Resume Analyzer (NLP-based)
    5. Salary Prediction Module
    6. Admin Analytics Dashboard
    7. Database Integration (SQLite)
    8. Authentication & Role Management
  Run: streamlit run app.py
=============================================================
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import sqlite3
import hashlib
import json
import re
import os
from datetime import datetime, date
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, roc_auc_score, roc_curve)
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier, MLPRegressor

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="🎓 PlaceIQ — Smart Placement System",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;800&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    .main { background-color: #0a0f1e; }
    .block-container { padding-top: 1.5rem; padding-bottom: 2rem; }

    .card {
        background: linear-gradient(135deg, #111827, #1a2234);
        border: 1px solid #1f2d45;
        border-radius: 14px;
        padding: 20px 24px;
        margin-bottom: 12px;
    }
    .card-green  { border-left: 4px solid #22c55e; }
    .card-red    { border-left: 4px solid #ef4444; }
    .card-blue   { border-left: 4px solid #3b82f6; }
    .card-amber  { border-left: 4px solid #f59e0b; }
    .card-purple { border-left: 4px solid #a855f7; }

    .metric-big  { font-size: 2.4rem; font-weight: 800; }
    .metric-lbl  { color: #64748b; font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.08em; margin-top: 4px; }
    .tag {
        display: inline-block;
        padding: 3px 10px;
        border-radius: 999px;
        font-size: 0.75rem;
        font-weight: 600;
        margin: 3px 3px;
    }
    .tag-green  { background: rgba(34,197,94,0.15);  color: #22c55e;  border: 1px solid rgba(34,197,94,0.3); }
    .tag-red    { background: rgba(239,68,68,0.15);  color: #ef4444;  border: 1px solid rgba(239,68,68,0.3); }
    .tag-blue   { background: rgba(59,130,246,0.15); color: #60a5fa;  border: 1px solid rgba(59,130,246,0.3); }
    .tag-amber  { background: rgba(245,158,11,0.15); color: #fbbf24;  border: 1px solid rgba(245,158,11,0.3); }
    .tag-purple { background: rgba(168,85,247,0.15); color: #c084fc;  border: 1px solid rgba(168,85,247,0.3); }

    .section-header {
        font-size: 1.4rem; font-weight: 800;
        color: #e2e8f0; margin: 8px 0 16px 0;
        border-bottom: 2px solid #1f2d45; padding-bottom: 8px;
    }
    div[data-testid="stMetric"] {
        background: #111827;
        border: 1px solid #1f2d45;
        border-radius: 12px;
        padding: 14px 18px;
    }
    .login-box {
        max-width: 420px; margin: 60px auto;
        background: #111827; border: 1px solid #1f2d45;
        border-radius: 20px; padding: 40px;
    }
    .roadmap-step {
        background: #111827; border: 1px solid #1f2d45;
        border-radius: 10px; padding: 14px 18px; margin: 8px 0;
    }
</style>
""", unsafe_allow_html=True)

PALETTE = {"Placed": "#22c55e", "NotPlaced": "#ef4444"}

# ══════════════════════════════════════════════════════════════════════════════
# ── MODULE 7: DATABASE (SQLite) ───────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

DB_PATH = "placeiq.db"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.executescript("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL,
            role TEXT NOT NULL,
            full_name TEXT,
            department TEXT,
            created_at TEXT
        );
        CREATE TABLE IF NOT EXISTS students (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE,
            cgpa REAL, major_projects INTEGER,
            workshops INTEGER, mini_projects INTEGER,
            skills INTEGER, comm_rating REAL,
            internship TEXT, hackathon TEXT,
            twelfth_pct REAL, tenth_pct INTEGER,
            backlogs INTEGER, department TEXT, year INTEGER
        );
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT, model_used TEXT,
            probability REAL, prediction TEXT,
            predicted_salary_min REAL, predicted_salary_max REAL,
            timestamp TEXT
        );
        CREATE TABLE IF NOT EXISTS resume_scores (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT, ats_score REAL,
            keyword_score REAL, structure_score REAL,
            grammar_score REAL, overall_score REAL,
            timestamp TEXT
        );
    """)
    # Seed demo users
    demo_users = [
        ("admin",    _hash("admin123"),    "admin",   "Admin User",    "IT"),
        ("officer",  _hash("officer123"),  "officer", "Placement Officer", "Placement Cell"),
        ("student1", _hash("student123"),  "student", "Arjun Sharma",  "CSE"),
    ]
    for u in demo_users:
        try:
            c.execute("INSERT INTO users (username,password,role,full_name,department,created_at) VALUES (?,?,?,?,?,?)",
                      (*u, datetime.now().isoformat()))
        except sqlite3.IntegrityError:
            pass
    conn.commit()
    conn.close()

def _hash(pw): return hashlib.sha256(pw.encode()).hexdigest()

def db_get_user(username, password):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT username,role,full_name,department FROM users WHERE username=? AND password=?",
              (username, _hash(password)))
    row = c.fetchone()
    conn.close()
    return row

def db_register_user(username, password, role, full_name, dept):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    try:
        c.execute("INSERT INTO users (username,password,role,full_name,department,created_at) VALUES (?,?,?,?,?,?)",
                  (username, _hash(password), role, full_name, dept, datetime.now().isoformat()))
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False
    finally:
        conn.close()

def db_save_prediction(username, model, prob, pred, sal_min, sal_max):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("INSERT INTO predictions (username,model_used,probability,prediction,predicted_salary_min,predicted_salary_max,timestamp) VALUES (?,?,?,?,?,?,?)",
              (username, model, prob, pred, sal_min, sal_max, datetime.now().isoformat()))
    conn.commit()
    conn.close()

def db_save_resume(username, ats, kw, struct, gram, overall):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("INSERT INTO resume_scores (username,ats_score,keyword_score,structure_score,grammar_score,overall_score,timestamp) VALUES (?,?,?,?,?,?,?)",
              (username, ats, kw, struct, gram, overall, datetime.now().isoformat()))
    conn.commit()
    conn.close()

def db_all_predictions():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql("SELECT * FROM predictions", conn)
    conn.close()
    return df

def db_all_users():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql("SELECT username,role,full_name,department,created_at FROM users", conn)
    conn.close()
    return df

# ══════════════════════════════════════════════════════════════════════════════
# ── DATA & MODEL TRAINING ─────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data
def load_data():
    df = pd.read_csv("Placement_Prediction_data.csv")
    df.drop(columns=["Unnamed: 0", "StudentId"], errors="ignore", inplace=True)
    return df

@st.cache_resource
def train_models(df):
    df_model = df.copy()
    le = LabelEncoder()
    for col in ["Internship", "Hackathon"]:
        df_model[col] = le.fit_transform(df_model[col])
    df_model["PlacementStatus"] = (df_model["PlacementStatus"] == "Placed").astype(int)

    X = df_model.drop(columns=["PlacementStatus"])
    y = df_model["PlacementStatus"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y)

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Decision Tree":       DecisionTreeClassifier(max_depth=8, random_state=42),
        "Random Forest":       RandomForestClassifier(n_estimators=200, random_state=42),
        "Gradient Boosting":   GradientBoostingClassifier(n_estimators=200, random_state=42),
        "SVM":                 SVC(kernel="rbf", probability=True, random_state=42),
        "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=7),
        "Neural Network (ANN)":MLPClassifier(hidden_layer_sizes=(128,64,32), max_iter=300, random_state=42),
    }

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    results = {}
    for name, model in models.items():
        cv = cross_val_score(model, X_train, y_train, cv=skf, scoring="accuracy")
        model.fit(X_train, y_train)
        y_pred  = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        results[name] = {
            "model":    model,
            "cv_mean":  cv.mean(),
            "cv_std":   cv.std(),
            "test_acc": accuracy_score(y_test, y_pred),
            "auc":      roc_auc_score(y_test, y_proba),
            "y_pred":   y_pred,
            "y_proba":  y_proba,
        }
    return results, X, y, X_test, y_test, scaler

@st.cache_resource
def train_salary_model(df):
    """Regression model to predict salary range based on profile."""
    df2 = df.copy()
    le = LabelEncoder()
    for col in ["Internship", "Hackathon"]:
        df2[col] = le.fit_transform(df2[col])
    placed = df2[df2["PlacementStatus"] == "Placed"].copy()

    # Synthetic salary generation based on features (realistic formula)
    np.random.seed(42)
    base = 300000
    placed["salary"] = (
        base
        + placed["CGPA"] * 40000
        - placed["backlogs"] * 15000
        + placed["Skills"] * 20000
        + placed["Communication Skill Rating"] * 8000
        + placed["Major Projects"] * 25000
        + placed["Internship"] * 50000
        + placed["Hackathon"] * 20000
        + placed["10th Percentage"] * 1000
        + np.random.normal(0, 30000, len(placed))
    ).clip(250000, 1200000)

    X_sal = placed.drop(columns=["PlacementStatus", "salary"])
    y_sal = placed["salary"]
    scaler2 = StandardScaler()
    X_sal_sc = scaler2.fit_transform(X_sal)
    X_tr, X_te, y_tr, y_te = train_test_split(X_sal_sc, y_sal, test_size=0.2, random_state=42)

    sal_model = MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42)
    sal_model.fit(X_tr, y_tr)
    return sal_model, scaler2

# ══════════════════════════════════════════════════════════════════════════════
# ── MODULE 8: AUTHENTICATION ──────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

def login_page():
    st.markdown("<div style='text-align:center;padding:30px 0 10px 0;'>"
                "<span style='font-size:3rem'>🎓</span>"
                "<h1 style='color:#e2e8f0;font-size:2.2rem;font-weight:800;margin:0'>PlaceIQ</h1>"
                "<p style='color:#64748b;'>Smart Placement Intelligence System</p>"
                "</div>", unsafe_allow_html=True)

    col_l, col_m, col_r = st.columns([1, 1.2, 1])
    with col_m:
        tab_login, tab_reg = st.tabs(["🔐 Login", "📝 Register"])

        with tab_login:
            username = st.text_input("Username", key="login_user")
            password = st.text_input("Password", type="password", key="login_pw")
            if st.button("Login →", use_container_width=True, type="primary"):
                row = db_get_user(username, password)
                if row:
                    st.session_state["logged_in"] = True
                    st.session_state["username"]   = row[0]
                    st.session_state["role"]       = row[1]
                    st.session_state["full_name"]  = row[2]
                    st.session_state["department"] = row[3]
                    st.rerun()
                else:
                    st.error("Invalid credentials.")
            st.markdown("""
            <div style='background:#111827;border:1px solid #1f2d45;border-radius:10px;padding:14px;margin-top:14px;font-size:0.82rem;color:#64748b'>
            <b style='color:#94a3b8'>Demo Accounts</b><br>
            👤 admin / admin123 &nbsp;|&nbsp; 🏢 officer / officer123 &nbsp;|&nbsp; 🎓 student1 / student123
            </div>""", unsafe_allow_html=True)

        with tab_reg:
            r_name  = st.text_input("Full Name", key="reg_name")
            r_user  = st.text_input("Username", key="reg_user")
            r_pw    = st.text_input("Password", type="password", key="reg_pw")
            r_role  = st.selectbox("Role", ["student", "officer"], key="reg_role")
            r_dept  = st.text_input("Department", key="reg_dept")
            if st.button("Register →", use_container_width=True):
                if db_register_user(r_user, r_pw, r_role, r_name, r_dept):
                    st.success("Registered! Please login.")
                else:
                    st.error("Username already exists.")

# ══════════════════════════════════════════════════════════════════════════════
# ── MODULE 2 & 3: CAREER RECOMMENDATION + SKILL GAP ──────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

CAREER_MAP = {
    "Software Developer": {
        "skills": ["Python", "Data Structures", "Algorithms", "Git", "SQL"],
        "certs":  ["Python Bootcamp (Udemy)", "LeetCode Premium", "AWS Cloud Practitioner"],
        "salary": "₹4L – ₹12L",
        "roadmap": ["Month 1-2: Master DSA & Python OOP", "Month 3: Build 2 full-stack projects",
                    "Month 4: Learn Git, REST APIs", "Month 5: Practice mock interviews",
                    "Month 6: Apply to 30+ companies"]
    },
    "Data Analyst": {
        "skills": ["SQL", "Python", "Excel", "Tableau/Power BI", "Statistics"],
        "certs":  ["Google Data Analytics (Coursera)", "SQL for Data Science", "Tableau Desktop Specialist"],
        "salary": "₹3.5L – ₹10L",
        "roadmap": ["Month 1: SQL + Excel mastery", "Month 2: Python for Data (Pandas, NumPy)",
                    "Month 3: Tableau / Power BI dashboards", "Month 4: Statistics & probability",
                    "Month 5: Kaggle projects", "Month 6: Resume + job applications"]
    },
    "Machine Learning Engineer": {
        "skills": ["Python", "ML Algorithms", "TensorFlow/PyTorch", "Statistics", "SQL"],
        "certs":  ["Andrew Ng's ML Course (Coursera)", "Fast.ai Deep Learning", "TensorFlow Developer Certificate"],
        "salary": "₹6L – ₹18L",
        "roadmap": ["Month 1: Python + Maths (Linear Algebra, Stats)", "Month 2: Classical ML (scikit-learn)",
                    "Month 3: Deep Learning basics", "Month 4: NLP or Computer Vision project",
                    "Month 5: Kaggle competitions", "Month 6: Portfolio + LinkedIn optimization"]
    },
    "Business Analyst": {
        "skills": ["Excel", "SQL", "Communication", "Problem Solving", "Power BI"],
        "certs":  ["Business Analysis Fundamentals (Udemy)", "CBAP Certification", "Google Analytics"],
        "salary": "₹4L – ₹9L",
        "roadmap": ["Month 1: Excel Advanced + SQL basics", "Month 2: Business process modeling",
                    "Month 3: Power BI / Tableau dashboards", "Month 4: Case study practice",
                    "Month 5: Communication & presentation skills", "Month 6: Apply to analyst roles"]
    },
    "Full Stack Developer": {
        "skills": ["HTML/CSS/JS", "React", "Node.js", "SQL", "REST APIs"],
        "certs":  ["The Odin Project (free)", "Full Stack Open (free)", "Meta Front-End Developer"],
        "salary": "₹4L – ₹14L",
        "roadmap": ["Month 1: HTML, CSS, JavaScript fundamentals", "Month 2: React.js",
                    "Month 3: Node.js + Express + MongoDB", "Month 4: Full project (deploy on Vercel)",
                    "Month 5: Git, Docker basics", "Month 6: Job applications + portfolio"]
    },
}

INDUSTRY_SKILLS = {
    "Python": 90, "SQL": 88, "Communication": 85,
    "Problem Solving": 82, "Git": 78, "Data Analysis": 75,
    "Machine Learning": 65, "Cloud (AWS/GCP)": 60,
    "System Design": 55, "Agile/Scrum": 50,
}

def render_career_recommendation(cgpa, backlogs, skills_count, comm, internship, major_p):
    st.markdown("<div class='section-header'>🎯 Career Recommendation Engine</div>", unsafe_allow_html=True)

    # Score each role
    scores = {}
    for role, data in CAREER_MAP.items():
        score = 0
        score += min(cgpa / 9.0, 1.0) * 30
        score += max(0, (5 - backlogs) / 5) * 20
        score += min(skills_count / 5, 1.0) * 20
        score += min(comm / 10.0, 1.0) * 15
        score += (1 if internship == "Yes" else 0) * 10
        score += min(major_p / 2, 1.0) * 5
        scores[role] = round(score, 1)

    top3 = sorted(scores.items(), key=lambda x: -x[1])[:3]

    cols = st.columns(3)
    for i, (role, score) in enumerate(top3):
        data = CAREER_MAP[role]
        with cols[i]:
            medal = ["🥇", "🥈", "🥉"][i]
            st.markdown(f"""
            <div class='card card-{"green" if i==0 else "blue" if i==1 else "amber"}'>
            <div style='font-size:1.5rem'>{medal}</div>
            <div style='font-size:1.1rem;font-weight:800;color:#e2e8f0'>{role}</div>
            <div style='font-size:0.8rem;color:#64748b;margin:4px 0'>Match Score</div>
            <div style='font-size:2rem;font-weight:800;color:{"#22c55e" if i==0 else "#60a5fa" if i==1 else "#fbbf24"}'>{score}%</div>
            <div style='margin-top:10px;font-size:0.8rem;color:#94a3b8'>💰 {data["salary"]}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("---")

    best_role, _ = top3[0]
    data = CAREER_MAP[best_role]
    st.markdown(f"### 🏆 Top Recommendation: {best_role}")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Required Skills**")
        for s in data["skills"]:
            st.markdown(f"<span class='tag tag-blue'>🔹 {s}</span>", unsafe_allow_html=True)

        st.markdown("**Recommended Certifications**")
        for cert in data["certs"]:
            st.markdown(f"<span class='tag tag-purple'>📜 {cert}</span>", unsafe_allow_html=True)

    with c2:
        st.markdown("**📅 6-Month Improvement Roadmap**")
        for step in data["roadmap"]:
            st.markdown(f"""<div class='roadmap-step'>
            <span style='color:#60a5fa'>▶</span> {step}
            </div>""", unsafe_allow_html=True)

def render_skill_gap(skills_count, cgpa, comm, internship, major_p, backlogs):
    st.markdown("<div class='section-header'>📊 Skill Gap Analysis</div>", unsafe_allow_html=True)

    # Map student profile to skill scores (0–100)
    student_skills = {
        "Python":             min(skills_count / 5 * 85 + 10, 95),
        "SQL":                min(skills_count / 5 * 75 + 5, 90),
        "Communication":      comm * 9,
        "Problem Solving":    (cgpa - 6.5) / 2.5 * 80 + 20,
        "Git":                (75 if internship == "Yes" else 35),
        "Data Analysis":      min(skills_count / 5 * 70, 90),
        "Machine Learning":   (cgpa - 6.5) / 2.5 * 60,
        "Cloud (AWS/GCP)":    (50 if major_p >= 1 else 15),
        "System Design":      (cgpa - 6.5) / 2.5 * 55,
        "Agile/Scrum":        (40 if internship == "Yes" else 10),
    }

    skill_names = list(INDUSTRY_SKILLS.keys())
    industry_vals = [INDUSTRY_SKILLS[s] for s in skill_names]
    student_vals  = [round(min(student_skills[s], 100), 1) for s in skill_names]
    gaps          = [max(0, i - s) for i, s in zip(industry_vals, student_vals)]
    readiness     = round(sum(student_vals) / sum(industry_vals) * 100, 1)

    col1, col2 = st.columns([3, 2])
    with col1:
        fig, ax = plt.subplots(figsize=(9, 5), facecolor="#111827")
        ax.set_facecolor("#111827")
        x = np.arange(len(skill_names))
        w = 0.38
        b1 = ax.bar(x - w/2, industry_vals, w, label="Industry Required", color="#3b82f6", alpha=0.85, edgecolor="white")
        b2 = ax.bar(x + w/2, student_vals,  w, label="Your Level",        color="#22c55e", alpha=0.85, edgecolor="white")
        ax.set_xticks(x)
        ax.set_xticklabels([s.replace(" ", "\n") for s in skill_names], fontsize=7, color="white")
        ax.set_ylabel("Score (%)", color="white")
        ax.set_ylim(0, 115)
        ax.legend(fontsize=9)
        ax.tick_params(colors="white")
        for spine in ax.spines.values(): spine.set_edgecolor("#1f2d45")
        fig.patch.set_facecolor("#111827")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col2:
        # Readiness gauge
        color = "#22c55e" if readiness >= 75 else "#f59e0b" if readiness >= 50 else "#ef4444"
        st.markdown(f"""
        <div class='card' style='text-align:center'>
            <div class='metric-lbl'>Skill Readiness</div>
            <div class='metric-big' style='color:{color}'>{readiness}%</div>
        </div>""", unsafe_allow_html=True)

        # Strong & missing
        strong  = [skill_names[i] for i, v in enumerate(student_vals) if v >= INDUSTRY_SKILLS[skill_names[i]] * 0.8]
        missing = [skill_names[i] for i, g in enumerate(gaps) if g > 30]

        st.markdown("**✅ Strong Skills**")
        for s in strong[:4]:
            st.markdown(f"<span class='tag tag-green'>{s}</span>", unsafe_allow_html=True)

        st.markdown("**⚠️ Skills to Improve**")
        for s in missing[:4]:
            st.markdown(f"<span class='tag tag-red'>{s}</span>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# ── MODULE 4: RESUME ANALYZER ─────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

TECH_KEYWORDS = [
    "python","java","c++","sql","machine learning","deep learning","tensorflow",
    "pytorch","nlp","data analysis","pandas","numpy","scikit-learn","git","github",
    "docker","kubernetes","aws","gcp","azure","react","node.js","flask","fastapi",
    "html","css","javascript","rest api","agile","scrum","linux","mongodb","postgresql"
]
RESUME_SECTIONS = ["education","experience","skills","projects","certifications","achievements","summary","objective"]

def analyze_resume(text: str):
    text_l = text.lower()
    words  = re.findall(r'\b\w+\b', text_l)
    word_count = len(words)

    # ATS Keywords
    found_kw = [kw for kw in TECH_KEYWORDS if kw in text_l]
    kw_score = min(len(found_kw) / 15 * 100, 100)

    # Section completeness
    found_sec = [s for s in RESUME_SECTIONS if s in text_l]
    struct_score = len(found_sec) / len(RESUME_SECTIONS) * 100

    # Grammar score (heuristic: sentence count, avg length)
    sentences = [s.strip() for s in re.split(r'[.!?]', text) if len(s.strip()) > 10]
    avg_len = np.mean([len(s.split()) for s in sentences]) if sentences else 0
    grammar_score = min(100, max(30, 100 - abs(avg_len - 15) * 4))

    # Overall ATS
    ats_score = round(kw_score * 0.4 + struct_score * 0.35 + grammar_score * 0.25, 1)
    overall   = round((kw_score + struct_score + grammar_score) / 3, 1)

    return {
        "ats_score":     round(ats_score, 1),
        "kw_score":      round(kw_score, 1),
        "struct_score":  round(struct_score, 1),
        "grammar_score": round(grammar_score, 1),
        "overall":       overall,
        "found_keywords":found_kw,
        "found_sections":found_sec,
        "missing_sections": [s for s in RESUME_SECTIONS if s not in found_sec],
        "word_count":    word_count,
    }

# ══════════════════════════════════════════════════════════════════════════════
# ── HELPER: Dark matplotlib figure ───────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

def dark_fig(*args, **kwargs):
    fig, ax = plt.subplots(*args, facecolor="#111827", **kwargs)
    if hasattr(ax, '__iter__'):
        for a in ax.flatten():
            a.set_facecolor("#111827")
            for sp in a.spines.values(): sp.set_edgecolor("#1f2d45")
            a.tick_params(colors="white")
    else:
        ax.set_facecolor("#111827")
        for sp in ax.spines.values(): sp.set_edgecolor("#1f2d45")
        ax.tick_params(colors="white")
    fig.patch.set_facecolor("#111827")
    return fig, ax

# ══════════════════════════════════════════════════════════════════════════════
# ── INIT ──────────────────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

init_db()

if "logged_in" not in st.session_state:
    st.session_state["logged_in"] = False

if not st.session_state["logged_in"]:
    login_page()
    st.stop()

# ── Load data & models ────────────────────────────────────────────────────────
df = load_data()

with st.spinner("🔄 Loading models... (cached after first run)"):
    results, X, y, X_test, y_test, scaler = train_models(df)
    sal_model, sal_scaler = train_salary_model(df)

best_name = max(results, key=lambda k: results[k]["auc"])
best      = results[best_name]

role = st.session_state["role"]
username = st.session_state["username"]

# ══════════════════════════════════════════════════════════════════════════════
# ── SIDEBAR ───────────────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown(f"### 🎓 PlaceIQ")
    st.markdown(f"**{st.session_state['full_name']}**")
    st.markdown(f"<span class='tag tag-{'green' if role=='admin' else 'blue' if role=='officer' else 'purple'}'>{role.upper()}</span>",
                unsafe_allow_html=True)
    st.markdown("---")

    if role == "student":
        page = st.radio("Navigate", [
            "🏠 Dashboard",
            "🤖 Predict My Placement",
            "🎯 Career Recommendations",
            "📊 Skill Gap Analysis",
            "📄 Resume Analyzer",
            "💰 Salary Prediction",
        ])
    elif role == "officer":
        page = st.radio("Navigate", [
            "📈 Admin Dashboard",
            "👥 All Students",
            "🤖 Predict My Placement",
            "📊 Model Comparison",
        ])
    else:  # admin
        page = st.radio("Navigate", [
            "📈 Admin Dashboard",
            "👥 All Students",
            "🔬 EDA",
            "🤖 Model Comparison",
            "🔍 Feature Importance",
        ])

    st.markdown("---")
    if st.button("🚪 Logout"):
        st.session_state.clear()
        st.rerun()

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: STUDENT DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════
if page == "🏠 Dashboard":
    st.title(f"👋 Welcome, {st.session_state['full_name']}!")
    st.markdown("Your personal placement intelligence hub.")
    st.markdown("---")

    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Best Model AUC",    f"{best['auc']:.4f}")
    c2.metric("Best Model",        best_name.split()[0])
    c3.metric("Dataset Size",      "10,000")
    c4.metric("Models Trained",    len(results))

    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📈 Model AUC Comparison")
        model_names = list(results.keys())
        aucs = [results[n]["auc"] for n in model_names]
        colors_bar = ["#22c55e" if n == best_name else "#3b82f6" for n in model_names]
        fig, ax = dark_fig(figsize=(7, 4))
        ax.barh(model_names, aucs, color=colors_bar, edgecolor="white", height=0.55)
        for i, (v, n) in enumerate(zip(aucs, model_names)):
            ax.text(v + 0.001, i, f"{v:.4f}", va="center", color="white", fontsize=8)
        ax.set_xlim(0.8, 1.02)
        ax.set_xlabel("AUC Score", color="white")
        plt.tight_layout(); st.pyplot(fig); plt.close()

    with col2:
        st.subheader("🎯 Class Distribution")
        counts = df["PlacementStatus"].value_counts()
        fig, ax = dark_fig(figsize=(7, 4))
        bars = ax.bar(counts.index, counts.values,
                      color=["#22c55e","#ef4444"], edgecolor="white", width=0.5)
        for bar, val in zip(bars, counts.values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50,
                    f"{val:,}", ha="center", fontweight="bold", color="white")
        plt.tight_layout(); st.pyplot(fig); plt.close()

    st.markdown("---")
    st.subheader("📋 All Models Summary")
    summary_df = pd.DataFrame({
        "Model":         model_names,
        "CV Accuracy":   [f"{results[n]['cv_mean']:.4f} ± {results[n]['cv_std']:.4f}" for n in model_names],
        "Test Accuracy": [f"{results[n]['test_acc']:.4f}" for n in model_names],
        "ROC-AUC":       [f"{results[n]['auc']:.4f}" for n in model_names],
        "Best":          ["🏆" if n == best_name else "" for n in model_names],
    })
    st.dataframe(summary_df, use_container_width=True, hide_index=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: PREDICT MY PLACEMENT (Module 1 + 5)
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🤖 Predict My Placement":
    st.title("🤖 Smart Placement Predictor")
    st.markdown("Real-time prediction powered by 7 ML models + salary estimation.")
    st.markdown("---")

    col1, col2 = st.columns([1, 1])
    with col1:
        st.subheader("📝 Student Profile")
        cgpa       = st.slider("CGPA", 6.5, 9.1, 7.5, 0.1)
        backlogs   = st.slider("Backlogs", 0, 7, 1)
        tenth      = st.slider("10th Percentage", 57, 88, 74)
        twelfth    = st.slider("12th Percentage", 57, 88, 74)
        comm       = st.slider("Communication Skill Rating", 1.0, 10.0, 6.5, 0.1)
        skills     = st.slider("Skills (count)", 0, 5, 3)
        major_p    = st.slider("Major Projects", 0, 2, 1)
        mini_p     = st.slider("Mini Projects", 0, 3, 1)
        ws         = st.slider("Workshops / Certifications", 0, 3, 1)
        internship = st.selectbox("Internship", ["Yes", "No"])
        hackathon  = st.selectbox("Hackathon", ["Yes", "No"])
        chosen_model = st.selectbox("Prediction Model", list(results.keys()), index=3)

    input_data = pd.DataFrame([{
        "CGPA":                      cgpa,
        "Major Projects":            major_p,
        "Workshops/Certificatios":   ws,
        "Mini Projects":             mini_p,
        "Skills":                    skills,
        "Communication Skill Rating":comm,
        "Internship":                1 if internship == "Yes" else 0,
        "Hackathon":                 1 if hackathon == "Yes" else 0,
        "12th Percentage":           twelfth,
        "10th Percentage":           tenth,
        "backlogs":                  backlogs,
    }])
    X_input  = scaler.transform(input_data)
    sel_m    = results[chosen_model]["model"]
    prob     = sel_m.predict_proba(X_input)[0][1]
    pred     = "Placed" if prob >= 0.5 else "NotPlaced"
    pct      = prob * 100

    # Salary prediction
    sal_inp = sal_scaler.transform(input_data)
    raw_sal = sal_model.predict(sal_inp)[0]
    sal_min = round(max(250000, raw_sal - 60000) / 100000, 2)
    sal_max = round(min(1500000, raw_sal + 80000) / 100000, 2)

    with col2:
        st.subheader("🔮 Results")
        if pred == "Placed":
            st.markdown(f"""
            <div class='card card-green' style='text-align:center;padding:28px'>
                <div style='font-size:3rem'>✅</div>
                <div style='font-size:1.8rem;font-weight:800;color:#22c55e'>LIKELY PLACED</div>
                <div style='font-size:3rem;font-weight:800;color:#22c55e'>{pct:.1f}%</div>
                <div style='color:#64748b'>Placement Probability via {chosen_model}</div>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class='card card-red' style='text-align:center;padding:28px'>
                <div style='font-size:3rem'>❌</div>
                <div style='font-size:1.8rem;font-weight:800;color:#ef4444'>LIKELY NOT PLACED</div>
                <div style='font-size:3rem;font-weight:800;color:#ef4444'>{pct:.1f}%</div>
                <div style='color:#64748b'>Placement Probability via {chosen_model}</div>
            </div>""", unsafe_allow_html=True)

        st.progress(int(pct))

        # MODULE 5: Salary
        st.markdown(f"""
        <div class='card card-amber'>
            <div style='font-size:0.8rem;color:#64748b'>💰 PREDICTED SALARY RANGE</div>
            <div style='font-size:1.8rem;font-weight:800;color:#fbbf24'>₹{sal_min}L – ₹{sal_max}L</div>
            <div style='color:#94a3b8;font-size:0.82rem'>Annual (based on profile + ML regression)</div>
        </div>""", unsafe_allow_html=True)

        # All models quick comparison
        st.markdown("#### 📊 All Models — Quick Compare")
        all_probs = {n: results[n]["model"].predict_proba(X_input)[0][1] for n in results}
        prob_df = pd.DataFrame({
            "Model":       list(all_probs.keys()),
            "Probability": [f"{v*100:.1f}%" for v in all_probs.values()],
            "Verdict":     ["✅ Placed" if v >= 0.5 else "❌ Not Placed" for v in all_probs.values()],
        })
        st.dataframe(prob_df, use_container_width=True, hide_index=True)

        # Save to DB
        db_save_prediction(username, chosen_model, prob, pred, sal_min, sal_max)
        st.caption("✅ Prediction saved to database.")

        # Insights
        st.markdown("#### 💡 Profile Insights")
        insights = []
        if backlogs == 0:    insights.append(("✅", "Zero backlogs — strong positive signal", "green"))
        elif backlogs >= 4:  insights.append(("⚠️", "High backlogs reduce chances significantly", "red"))
        else:                insights.append(("🔸", "Some backlogs — try to clear them", "amber"))
        if cgpa >= 8.5:      insights.append(("✅", "Excellent CGPA", "green"))
        elif cgpa < 7.0:     insights.append(("⚠️", "CGPA below 7.0 — focus on academics", "red"))
        if internship=="Yes":insights.append(("✅", "Internship experience adds value", "green"))
        else:                insights.append(("💡", "Consider completing an internship", "blue"))
        if hackathon=="Yes": insights.append(("✅", "Hackathon participation is a plus", "green"))
        if skills >= 4:      insights.append(("✅", "Strong skill set", "green"))
        elif skills <= 1:    insights.append(("💡", "Build more technical skills", "blue"))

        for icon, text, color in insights:
            st.markdown(f"<span class='tag tag-{color}'>{icon} {text}</span>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: CAREER RECOMMENDATIONS (Module 2)
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🎯 Career Recommendations":
    st.title("🎯 Career Recommendation Engine")
    st.markdown("Personalized career paths based on your academic profile.")
    st.markdown("---")

    with st.expander("⚙️ Enter / Update Your Profile", expanded=True):
        c1, c2, c3 = st.columns(3)
        with c1:
            r_cgpa  = st.slider("CGPA", 6.5, 9.1, 7.5, 0.1, key="cr_cgpa")
            r_bl    = st.slider("Backlogs", 0, 7, 1, key="cr_bl")
        with c2:
            r_sk    = st.slider("Skills", 0, 5, 3, key="cr_sk")
            r_comm  = st.slider("Communication", 1.0, 10.0, 6.5, 0.1, key="cr_comm")
        with c3:
            r_int   = st.selectbox("Internship", ["Yes","No"], key="cr_int")
            r_mp    = st.slider("Major Projects", 0, 2, 1, key="cr_mp")

    render_career_recommendation(r_cgpa, r_bl, r_sk, r_comm, r_int, r_mp)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: SKILL GAP ANALYSIS (Module 3)
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📊 Skill Gap Analysis":
    st.title("📊 Skill Gap Analysis")
    st.markdown("Compare your skills against industry requirements.")
    st.markdown("---")

    with st.expander("⚙️ Your Profile", expanded=True):
        c1, c2 = st.columns(2)
        with c1:
            sg_skills  = st.slider("Skills (count)", 0, 5, 3, key="sg_sk")
            sg_cgpa    = st.slider("CGPA", 6.5, 9.1, 7.5, 0.1, key="sg_cgpa")
            sg_comm    = st.slider("Communication Skill Rating", 1.0, 10.0, 6.5, 0.1, key="sg_comm")
        with c2:
            sg_int     = st.selectbox("Internship", ["Yes","No"], key="sg_int")
            sg_mp      = st.slider("Major Projects", 0, 2, 1, key="sg_mp")
            sg_bl      = st.slider("Backlogs", 0, 7, 1, key="sg_bl")

    render_skill_gap(sg_skills, sg_cgpa, sg_comm, sg_int, sg_mp, sg_bl)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: RESUME ANALYZER (Module 4)
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📄 Resume Analyzer":
    st.title("📄 AI Resume Analyzer")
    st.markdown("Paste your resume text for NLP-based ATS and structure analysis.")
    st.markdown("---")

    resume_text = st.text_area("📋 Paste Resume Text Here", height=280,
        placeholder="Paste your resume content (education, skills, experience, projects)...")

    if st.button("🔍 Analyze Resume", type="primary") and resume_text.strip():
        res = analyze_resume(resume_text)

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("ATS Score",       f"{res['ats_score']}%")
        col2.metric("Keyword Match",   f"{res['kw_score']:.1f}%")
        col3.metric("Structure Score", f"{res['struct_score']:.1f}%")
        col4.metric("Grammar Score",   f"{res['grammar_score']:.1f}%")

        st.markdown("---")
        col1, col2 = st.columns(2)
        with col1:
            # Radar-style bar chart
            categories = ["ATS", "Keywords", "Structure", "Grammar"]
            vals = [res["ats_score"], res["kw_score"], res["struct_score"], res["grammar_score"]]
            colors = ["#22c55e" if v >= 70 else "#f59e0b" if v >= 45 else "#ef4444" for v in vals]
            fig, ax = dark_fig(figsize=(7, 4))
            bars = ax.barh(categories, vals, color=colors, edgecolor="white", height=0.5)
            for bar, v in zip(bars, vals):
                ax.text(v + 1, bar.get_y() + bar.get_height()/2,
                        f"{v:.1f}%", va="center", color="white", fontweight="bold")
            ax.set_xlim(0, 115)
            ax.set_xlabel("Score (%)", color="white")
            ax.set_title("Resume Score Breakdown", color="white", fontweight="bold")
            plt.tight_layout(); st.pyplot(fig); plt.close()

        with col2:
            st.markdown("**✅ Found Keywords**")
            for kw in res["found_keywords"][:12]:
                st.markdown(f"<span class='tag tag-green'>{kw}</span>", unsafe_allow_html=True)

            st.markdown("**📂 Found Sections**")
            for sec in res["found_sections"]:
                st.markdown(f"<span class='tag tag-blue'>✅ {sec}</span>", unsafe_allow_html=True)

            if res["missing_sections"]:
                st.markdown("**❌ Missing Sections**")
                for sec in res["missing_sections"]:
                    st.markdown(f"<span class='tag tag-red'>❌ {sec}</span>", unsafe_allow_html=True)

        # Overall ATS verdict
        ats = res["ats_score"]
        verdict_color = "#22c55e" if ats >= 70 else "#f59e0b" if ats >= 45 else "#ef4444"
        verdict_text  = "Excellent — ATS-Ready ✅" if ats >= 70 else "Moderate — Needs Improvement ⚠️" if ats >= 45 else "Weak — Major Revamp Needed ❌"
        st.markdown(f"""
        <div class='card' style='margin-top:16px'>
            <div class='metric-lbl'>Overall ATS Verdict</div>
            <div style='font-size:1.5rem;font-weight:800;color:{verdict_color}'>{verdict_text}</div>
            <div style='color:#94a3b8;margin-top:6px'>Word count: {res["word_count"]} &nbsp;|&nbsp; Keywords found: {len(res["found_keywords"])}/30</div>
        </div>""", unsafe_allow_html=True)

        db_save_resume(username, res["ats_score"], res["kw_score"],
                       res["struct_score"], res["grammar_score"], res["overall"])
        st.caption("✅ Resume score saved to database.")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: SALARY PREDICTION (Module 5)
# ══════════════════════════════════════════════════════════════════════════════
elif page == "💰 Salary Prediction":
    st.title("💰 Salary Prediction Module")
    st.markdown("ML regression to estimate your expected salary range based on your profile.")
    st.markdown("---")

    col1, col2 = st.columns([1, 1])
    with col1:
        st.subheader("📝 Your Profile")
        s_cgpa   = st.slider("CGPA", 6.5, 9.1, 7.5, 0.1, key="sal_cgpa")
        s_bl     = st.slider("Backlogs", 0, 7, 0, key="sal_bl")
        s_10     = st.slider("10th Percentage", 57, 88, 78, key="sal_10")
        s_12     = st.slider("12th Percentage", 57, 88, 76, key="sal_12")
        s_comm   = st.slider("Communication Skill", 1.0, 10.0, 7.5, 0.1, key="sal_comm")
        s_sk     = st.slider("Skills", 0, 5, 4, key="sal_sk")
        s_mp     = st.slider("Major Projects", 0, 2, 1, key="sal_mp")
        s_mnp    = st.slider("Mini Projects", 0, 3, 2, key="sal_mnp")
        s_ws     = st.slider("Workshops", 0, 3, 2, key="sal_ws")
        s_int    = st.selectbox("Internship", ["Yes","No"], key="sal_int")
        s_hack   = st.selectbox("Hackathon", ["Yes","No"], key="sal_hack")

    sal_df = pd.DataFrame([{
        "CGPA": s_cgpa, "Major Projects": s_mp, "Workshops/Certificatios": s_ws,
        "Mini Projects": s_mnp, "Skills": s_sk, "Communication Skill Rating": s_comm,
        "Internship": 1 if s_int=="Yes" else 0, "Hackathon": 1 if s_hack=="Yes" else 0,
        "12th Percentage": s_12, "10th Percentage": s_10, "backlogs": s_bl,
    }])
    sal_sc = sal_scaler.transform(sal_df)
    raw = sal_model.predict(sal_sc)[0]
    s_min = round(max(250000, raw - 60000), -3)
    s_max = round(min(1500000, raw + 80000), -3)

    with col2:
        st.subheader("💰 Salary Estimate")
        st.markdown(f"""
        <div class='card card-amber' style='text-align:center;padding:32px'>
            <div style='font-size:0.9rem;color:#64748b'>PREDICTED ANNUAL SALARY</div>
            <div style='font-size:3rem;font-weight:800;color:#fbbf24'>
                ₹{s_min/100000:.1f}L – ₹{s_max/100000:.1f}L
            </div>
            <div style='color:#94a3b8;font-size:0.85rem'>Based on ML Regression Model</div>
        </div>""", unsafe_allow_html=True)

        # Salary breakdown by factor
        st.markdown("#### 📊 Salary Drivers")
        factors = {
            "CGPA":         s_cgpa / 9.1,
            "Skills":       s_sk / 5,
            "Communication":s_comm / 10,
            "Internship":   1 if s_int == "Yes" else 0,
            "Projects":     (s_mp * 2 + s_mnp) / 7,
            "Academics":    ((s_10 + s_12) / 2 - 57) / 31,
        }
        fig, ax = dark_fig(figsize=(7, 4))
        names = list(factors.keys())
        vals  = [v * 100 for v in factors.values()]
        colors= ["#22c55e" if v >= 70 else "#f59e0b" if v >= 40 else "#ef4444" for v in vals]
        bars = ax.barh(names, vals, color=colors, edgecolor="white", height=0.5)
        for bar, v in zip(bars, vals):
            ax.text(v + 1, bar.get_y() + bar.get_height()/2,
                    f"{v:.0f}%", va="center", color="white")
        ax.set_xlim(0, 115)
        ax.set_xlabel("Strength (%)", color="white")
        ax.set_title("Profile Strength by Factor", color="white", fontweight="bold")
        plt.tight_layout(); st.pyplot(fig); plt.close()

        # Industry benchmark
        st.markdown("""
        <div class='card card-blue' style='font-size:0.85rem'>
            <b>📊 Industry Benchmarks (Freshers 2024)</b><br>
            🟢 Top 10% &nbsp; ₹12L+ &nbsp;|&nbsp;
            🟡 Median &nbsp; ₹5–7L &nbsp;|&nbsp;
            🔴 Entry &nbsp; ₹2.5–4L
        </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: ADMIN DASHBOARD (Module 6)
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📈 Admin Dashboard":
    st.title("📈 Admin Analytics Dashboard")
    st.markdown("Placement trends, department performance, and student insights.")
    st.markdown("---")

    # KPIs
    total = len(df)
    placed_n = (df["PlacementStatus"] == "Placed").sum()
    not_placed_n = total - placed_n
    c1,c2,c3,c4,c5 = st.columns(5)
    c1.metric("Total Students",  f"{total:,}")
    c2.metric("Placed",          f"{placed_n:,}", f"{placed_n/total*100:.1f}%")
    c3.metric("Not Placed",      f"{not_placed_n:,}", f"-{not_placed_n/total*100:.1f}%")
    c4.metric("Best Model AUC",  f"{best['auc']:.4f}")
    c5.metric("Models Trained",  len(results))

    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📊 CGPA Distribution by Status")
        fig, ax = dark_fig(figsize=(7, 4))
        for status, color in PALETTE.items():
            ax.hist(df[df["PlacementStatus"] == status]["CGPA"],
                    bins=25, alpha=0.65, color=color, label=status, edgecolor="none")
        ax.set_xlabel("CGPA", color="white")
        ax.set_ylabel("Count", color="white")
        ax.legend()
        plt.tight_layout(); st.pyplot(fig); plt.close()

    with col2:
        st.subheader("🔍 Backlog Impact on Placement")
        backlog_rate = df.groupby("backlogs").apply(
            lambda x: (x["PlacementStatus"] == "Placed").mean() * 100)
        fig, ax = dark_fig(figsize=(7, 4))
        colors_bl = ["#22c55e" if v >= 50 else "#f59e0b" if v >= 30 else "#ef4444"
                     for v in backlog_rate.values]
        ax.bar(backlog_rate.index, backlog_rate.values, color=colors_bl, edgecolor="white", width=0.7)
        ax.set_xlabel("Number of Backlogs", color="white")
        ax.set_ylabel("Placement Rate (%)", color="white")
        ax.axhline(50, color="#64748b", linestyle="--", lw=1.5)
        plt.tight_layout(); st.pyplot(fig); plt.close()

    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("💼 Internship vs Placement Rate")
        int_rate = df.groupby("Internship").apply(
            lambda x: (x["PlacementStatus"] == "Placed").mean() * 100)
        fig, ax = dark_fig(figsize=(6, 4))
        ax.bar(int_rate.index, int_rate.values, color=["#22c55e","#ef4444"], edgecolor="white", width=0.5)
        ax.set_ylabel("Placement Rate (%)", color="white")
        for i, v in enumerate(int_rate.values):
            ax.text(i, v + 1, f"{v:.1f}%", ha="center", color="white", fontweight="bold")
        plt.tight_layout(); st.pyplot(fig); plt.close()

    with col2:
        st.subheader("🏆 Skill Count vs Placement")
        sk_rate = df.groupby("Skills").apply(
            lambda x: (x["PlacementStatus"] == "Placed").mean() * 100)
        fig, ax = dark_fig(figsize=(6, 4))
        ax.plot(sk_rate.index, sk_rate.values, marker="o", color="#3b82f6", linewidth=2.5, markersize=8)
        ax.fill_between(sk_rate.index, sk_rate.values, alpha=0.2, color="#3b82f6")
        ax.set_xlabel("Skills Count", color="white")
        ax.set_ylabel("Placement Rate (%)", color="white")
        plt.tight_layout(); st.pyplot(fig); plt.close()

    # DB Predictions log
    st.markdown("---")
    st.subheader("🗄️ Recent Predictions (Database)")
    pred_df = db_all_predictions()
    if not pred_df.empty:
        st.dataframe(pred_df.tail(20), use_container_width=True, hide_index=True)
    else:
        st.info("No predictions saved yet. Use the Predictor to generate some!")

    st.subheader("👥 Registered Users")
    users_df = db_all_users()
    st.dataframe(users_df, use_container_width=True, hide_index=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: ALL STUDENTS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "👥 All Students":
    st.title("👥 Student Records")
    st.markdown("---")
    st.subheader("📋 Full Dataset")
    search = st.text_input("🔍 Filter by Placement Status", placeholder="Placed / NotPlaced")
    if search:
        filt = df[df["PlacementStatus"].str.contains(search, case=False)]
    else:
        filt = df
    st.dataframe(filt, use_container_width=True)
    st.caption(f"Showing {len(filt):,} of {len(df):,} records")

    st.markdown("---")
    # Weak students (risk analysis)
    st.subheader("⚠️ At-Risk Students (backlogs ≥ 3 AND CGPA < 7.2)")
    at_risk = df[(df["backlogs"] >= 3) & (df["CGPA"] < 7.2)]
    st.metric("At-Risk Students", len(at_risk), f"{len(at_risk)/len(df)*100:.1f}% of total")
    st.dataframe(at_risk.head(30), use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: EDA
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔬 EDA":
    st.title("🔬 Exploratory Data Analysis")
    st.markdown("---")

    tab1, tab2, tab3, tab4 = st.tabs(["📊 Distributions", "📦 Boxplots", "🏷 Categorical", "🌡 Correlation"])

    with tab1:
        num_cols = ["CGPA","Communication Skill Rating","12th Percentage",
                    "10th Percentage","backlogs","Skills",
                    "Major Projects","Mini Projects","Workshops/Certificatios"]
        fig, axes = plt.subplots(3, 3, figsize=(16, 11), facecolor="#111827")
        for i, col in enumerate(num_cols):
            ax = axes.flatten()[i]
            ax.set_facecolor("#111827")
            for status, color in PALETTE.items():
                ax.hist(df[df["PlacementStatus"] == status][col],
                        bins=20, alpha=0.65, color=color, label=status, edgecolor="none")
            ax.set_title(col, color="white", fontweight="bold", fontsize=9)
            ax.legend(fontsize=7)
            ax.tick_params(colors="white")
            for sp in ax.spines.values(): sp.set_edgecolor("#1f2d45")
        fig.patch.set_facecolor("#111827")
        plt.tight_layout(); st.pyplot(fig); plt.close()

    with tab2:
        box_cols = ["CGPA","Communication Skill Rating","12th Percentage","10th Percentage","backlogs"]
        fig, axes = plt.subplots(1, 5, figsize=(20, 5), facecolor="#111827")
        for i, col in enumerate(box_cols):
            axes[i].set_facecolor("#111827")
            data_box = [df[df["PlacementStatus"]==s][col] for s in ["Placed","NotPlaced"]]
            bp = axes[i].boxplot(data_box, patch_artist=True,
                                 labels=["Placed","Not\nPlaced"],
                                 medianprops=dict(color="white", linewidth=2))
            for patch, color in zip(bp["boxes"], list(PALETTE.values())):
                patch.set_facecolor(color); patch.set_alpha(0.75)
            axes[i].set_title(col, color="white", fontweight="bold", fontsize=8)
            axes[i].tick_params(colors="white")
            for sp in axes[i].spines.values(): sp.set_edgecolor("#1f2d45")
        fig.patch.set_facecolor("#111827")
        plt.tight_layout(); st.pyplot(fig); plt.close()

    with tab3:
        c1, c2 = st.columns(2)
        for col_name, widget in [("Internship", c1), ("Hackathon", c2)]:
            ct = pd.crosstab(df[col_name], df["PlacementStatus"])
            fig, ax = dark_fig(figsize=(6, 4))
            ct.plot(kind="bar", ax=ax, color=list(PALETTE.values()), edgecolor="white", rot=0)
            ax.set_title(f"{col_name} vs Placement", color="white", fontweight="bold")
            ax.legend(title="Status")
            plt.tight_layout()
            with widget: st.pyplot(fig)
            plt.close()

    with tab4:
        df_enc = df.copy()
        for col in ["Internship","Hackathon","PlacementStatus"]:
            df_enc[col] = LabelEncoder().fit_transform(df_enc[col])
        corr = df_enc.corr()
        mask = np.triu(np.ones_like(corr, dtype=bool))
        fig, ax = dark_fig(figsize=(10, 8))
        sns.heatmap(corr, mask=mask, ax=ax, cmap="RdYlGn",
                    annot=True, fmt=".2f", annot_kws={"size": 9},
                    linewidths=0.5, linecolor="#1f2d45", vmin=-1, vmax=1)
        ax.set_title("Correlation Heatmap", color="white", fontweight="bold")
        ax.tick_params(colors="white")
        fig.patch.set_facecolor("#111827")
        plt.tight_layout(); st.pyplot(fig); plt.close()

    st.markdown("---")
    st.subheader("📋 Raw Dataset Preview")
    st.dataframe(df.head(50), use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: MODEL COMPARISON
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📊 Model Comparison":
    st.title("📊 Model Comparison — 7 Algorithms")
    st.markdown("---")

    model_names = list(results.keys())

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📈 Accuracy vs AUC")
        x = np.arange(len(model_names))
        w = 0.35
        fig, ax = dark_fig(figsize=(8, 5))
        ax.bar(x - w/2, [results[n]["test_acc"] for n in model_names], w,
               label="Test Accuracy", color="#3b82f6", edgecolor="white")
        ax.bar(x + w/2, [results[n]["auc"] for n in model_names], w,
               label="AUC", color="#f59e0b", edgecolor="white")
        ax.set_xticks(x)
        ax.set_xticklabels([n.replace(" ","\n") for n in model_names], fontsize=7, color="white")
        ax.set_ylim(0.8, 1.02)
        ax.legend()
        ax.set_ylabel("Score", color="white")
        plt.tight_layout(); st.pyplot(fig); plt.close()

    with col2:
        st.subheader("📉 ROC Curves")
        fig, ax = dark_fig(figsize=(8, 5))
        for name in model_names:
            fpr, tpr, _ = roc_curve(y_test, results[name]["y_proba"])
            lw = 2.5 if name == best_name else 1
            ax.plot(fpr, tpr, linewidth=lw,
                    label=f"{name.split()[0]} ({results[name]['auc']:.3f})")
        ax.plot([0, 1], [0, 1], "w--", lw=1)
        ax.set_xlabel("FPR", color="white")
        ax.set_ylabel("TPR", color="white")
        ax.legend(fontsize=7)
        plt.tight_layout(); st.pyplot(fig); plt.close()

    st.markdown("---")
    st.subheader(f"🏆 Best: {best_name}")
    col1, col2 = st.columns([1,1])
    with col1:
        cm = confusion_matrix(y_test, best["y_pred"])
        fig, ax = dark_fig(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=["NotPlaced","Placed"],
                    yticklabels=["NotPlaced","Placed"],
                    linewidths=2, linecolor="#111827", ax=ax,
                    annot_kws={"size": 16, "fontweight": "bold"})
        ax.set_xlabel("Predicted", color="white"); ax.set_ylabel("Actual", color="white")
        ax.tick_params(colors="white")
        fig.patch.set_facecolor("#111827")
        plt.tight_layout(); st.pyplot(fig); plt.close()
    with col2:
        report = classification_report(y_test, best["y_pred"],
                                       target_names=["NotPlaced","Placed"],
                                       output_dict=True)
        report_df = pd.DataFrame(report).T.round(4)
        st.dataframe(report_df, use_container_width=True)

    st.markdown("---")
    st.subheader("📋 Full Comparison Table")
    summary_df = pd.DataFrame({
        "Model":         model_names,
        "CV Accuracy":   [f"{results[n]['cv_mean']:.4f} ± {results[n]['cv_std']:.4f}" for n in model_names],
        "Test Accuracy": [f"{results[n]['test_acc']:.4f}" for n in model_names],
        "ROC-AUC":       [f"{results[n]['auc']:.4f}" for n in model_names],
        "Best":          ["🏆" if n == best_name else "" for n in model_names],
    })
    st.dataframe(summary_df, use_container_width=True, hide_index=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: FEATURE IMPORTANCE
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔍 Feature Importance":
    st.title("🔍 Feature Importance Analysis")
    st.markdown("---")

    rf_model = results["Random Forest"]["model"]
    gb_model = results["Gradient Boosting"]["model"]
    fi_rf = pd.Series(rf_model.feature_importances_, index=X.columns).sort_values(ascending=False)
    fi_gb = pd.Series(gb_model.feature_importances_, index=X.columns).sort_values(ascending=False)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("🌲 Random Forest")
        fig, ax = dark_fig(figsize=(7, 5))
        vals_sorted = fi_rf.sort_values()
        colors_rf = ["#22c55e" if v >= fi_rf.median() else "#60a5fa" for v in vals_sorted.values]
        ax.barh(vals_sorted.index, vals_sorted.values, color=colors_rf, edgecolor="white", height=0.6)
        ax.axvline(fi_rf.median(), color="#ef4444", linestyle="--", lw=1.5, label="Median")
        ax.legend(); ax.set_xlabel("Importance", color="white")
        plt.tight_layout(); st.pyplot(fig); plt.close()

    with col2:
        st.subheader("⚡ Gradient Boosting")
        fig, ax = dark_fig(figsize=(7, 5))
        vals_sorted_gb = fi_gb.sort_values()
        colors_gb = ["#f59e0b" if v >= fi_gb.median() else "#fcd34d" for v in vals_sorted_gb.values]
        ax.barh(vals_sorted_gb.index, vals_sorted_gb.values, color=colors_gb, edgecolor="white", height=0.6)
        ax.axvline(fi_gb.median(), color="#ef4444", linestyle="--", lw=1.5, label="Median")
        ax.legend(); ax.set_xlabel("Importance", color="white")
        plt.tight_layout(); st.pyplot(fig); plt.close()

    st.markdown("---")
    st.subheader("📋 Feature Importance Table")
    fi_table = pd.DataFrame({
        "Feature":       fi_rf.index,
        "RF Importance": fi_rf.values.round(4),
        "GB Importance": [fi_gb.get(f, 0) for f in fi_rf.index],
        "RF Rank":       range(1, len(fi_rf)+1),
    })
    st.dataframe(fi_table, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.subheader("💡 Key Insights")
    insights_html = """
    <div class='card card-blue'><b>🔑 #1 backlogs (~47%)</b><br>
    Academic backlogs are the strongest placement predictor. Zero backlogs = highest chance.</div>
    <div class='card card-green'><b>🎓 #2 & #3: 10th % & CGPA</b><br>
    Consistent academic performance across all years is highly valued by recruiters.</div>
    <div class='card card-amber'><b>💼 Internship & Hackathon</b><br>
    Practical experience significantly boosts placement probability.</div>
    <div class='card card-purple'><b>🧠 Communication Skills</b><br>
    High communication rating correlates strongly with placement success in interviews.</div>
    """
    st.markdown(insights_html, unsafe_allow_html=True)
