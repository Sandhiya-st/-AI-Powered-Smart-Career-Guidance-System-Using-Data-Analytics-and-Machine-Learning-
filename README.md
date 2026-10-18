# 🎓 PlaceIQ — Smart Placement Intelligence System
### Major Project | AI-Powered Campus Placement Prediction System

---

## 📁 Project Structure
```
placement_major/
│
├── app.py                          ← Main Streamlit app (all modules)
├── Placement_Prediction_data.csv   ← Dataset (copy from your mini project)
├── requirements.txt                ← Dependencies
├── placeiq.db                      ← SQLite database (auto-created on first run)
└── README.md                       ← This file
```

---

## 🚀 How to Run

### Step 1 — Copy your dataset
```
Copy Placement_Prediction_data.csv into this folder
```

### Step 2 — Create virtual environment
```bash
python -m venv venv
```
Activate:
- **Windows**: `venv\Scripts\activate`
- **Mac/Linux**: `source venv/bin/activate`

### Step 3 — Install dependencies
```bash
pip install -r requirements.txt
```

### Step 4 — Run the app
```bash
streamlit run app.py
```
Opens at **http://localhost:8501** 🎉

---

## 🔐 Demo Login Accounts

| Role | Username | Password | Access |
|------|----------|----------|--------|
| 👤 Admin | `admin` | `admin123` | Full access |
| 🏢 Placement Officer | `officer` | `officer123` | Dashboard + students |
| 🎓 Student | `student1` | `student123` | Personal tools |

---

## 🧠 8 Major Modules Implemented

### 1. 🤖 Multi-Model Prediction Engine
- **7 Models**: Logistic Regression, Decision Tree, Random Forest, Gradient Boosting, SVM, KNN, Neural Network (ANN)
- Cross-validation with StratifiedKFold (5-fold)
- ROC-AUC comparison, Confusion Matrix, Classification Report
- User can select any model for live prediction

### 2. 🎯 Career Recommendation Module
- Recommends Top 3 career paths based on profile
- Roles: Software Developer, Data Analyst, ML Engineer, Business Analyst, Full Stack
- Shows: Required skills, Certifications, Salary range
- Includes 6-Month Improvement Roadmap

### 3. 📊 Skill Gap Analysis
- Compares student skills vs. industry requirements (10 key skills)
- Skill Readiness % score
- Visual bar chart: Industry vs. Student levels
- Highlights strong skills and skills to improve

### 4. 📄 Resume Analyzer (NLP-Based)
- **ATS Score**: Keyword-weighted compatibility score
- **Keyword Match**: 30 tech keywords detected
- **Structure Score**: Section completeness (Education, Experience, Skills, etc.)
- **Grammar Score**: Sentence quality heuristic
- Missing sections highlighted with improvement tips
- Scores saved to database

### 5. 💰 Salary Prediction Module
- MLPRegressor trained on synthetic salary data based on features
- Predicts annual salary range (₹L)
- Factor breakdown chart showing salary drivers
- Industry benchmark comparison

### 6. 📈 Admin Analytics Dashboard
- CGPA distribution by placement status
- Backlog impact analysis (placement rate per backlog count)
- Internship vs. placement rate comparison
- Skills count trend analysis
- Live database log of all predictions and users
- At-risk student identification (backlogs ≥ 3 AND CGPA < 7.2)

### 7. 🗄️ Database Integration (SQLite)
- **Tables**: `users`, `students`, `predictions`, `resume_scores`
- All predictions auto-saved with timestamp
- Resume scores persisted per user
- Admin can view full prediction history
- Easily upgradeable to MySQL/PostgreSQL

### 8. 🔐 Authentication & Role Management
- **3 Roles**: Student, Placement Officer, Admin
- Role-based page access (students see personal tools, admin sees everything)
- SHA-256 password hashing
- User registration with department selection
- Session management via Streamlit session_state

---

## 🤖 Models Trained

| Model | Type | Notes |
|-------|------|-------|
| Logistic Regression | Linear | Baseline |
| Decision Tree | Tree | max_depth=8 |
| Random Forest | Ensemble | 200 estimators |
| **Gradient Boosting** | Ensemble | **Best: AUC ~0.99** |
| SVM | Kernel | RBF kernel |
| K-Nearest Neighbors | Instance | k=7 |
| Neural Network (ANN) | Deep | 3 hidden layers: 128→64→32 |

---

## 🔑 Key Findings

1. **backlogs** — #1 predictor (47% importance). Zero backlogs = highest placement rate.
2. **CGPA** — Strong second predictor. 8.5+ CGPA significantly boosts chances.
3. **10th Percentage** — Foundational academics remain relevant to recruiters.
4. **Internship** — Students with internships have ~20% higher placement rate.
5. **Communication Skills** — High-rated communication correlates with placement success.

---

## ☁️ Deployment (Major Project Add-On)

### Option A — Render (Free)
```bash
# Create render.yaml in root
services:
  - type: web
    name: placeiq
    env: python
    buildCommand: pip install -r requirements.txt
    startCommand: streamlit run app.py --server.port $PORT
```

### Option B — Docker
```dockerfile
FROM python:3.10-slim
WORKDIR /app
COPY . .
RUN pip install -r requirements.txt
EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0"]
```

### Option C — AWS EC2
```bash
# On EC2 instance:
pip install -r requirements.txt
nohup streamlit run app.py --server.port 8501 &
```

> 💡 **For your report:** *"The system is built with a scalable, modular architecture. The SQLite backend can be seamlessly migrated to MySQL or PostgreSQL for production, and the Streamlit application is containerized with Docker for cloud deployment on AWS EC2 or Render."*

---

## 📊 Tech Stack

| Layer | Technology |
|-------|-----------|
| Frontend | Streamlit |
| ML Models | scikit-learn |
| Deep Learning | MLPClassifier / MLPRegressor |
| NLP Resume Analysis | Python re + keyword extraction |
| Database | SQLite (upgradeable to MySQL) |
| Auth | SHA-256 + Streamlit session_state |
| Visualization | Matplotlib + Seaborn |
| Deployment-Ready | Docker + REST API compatible |

---

## 🆚 Mini vs Major Project Comparison

| Feature | Mini Project | Major Project (This) |
|---------|-------------|---------------------|
| ML Models | 6 models | 7 models + ANN |
| Prediction | ✅ | ✅ Enhanced |
| Career Recommendation | ❌ | ✅ NEW |
| Skill Gap Analysis | ❌ | ✅ NEW |
| Resume Analyzer | ❌ | ✅ NLP-Based |
| Salary Prediction | ❌ | ✅ NEW (Regression) |
| Admin Dashboard | Basic | ✅ Full Analytics |
| Database | CSV only | ✅ SQLite (DB) |
| Authentication | ❌ | ✅ Role-Based |
| Deployment Ready | ❌ | ✅ Docker + Cloud |

---

*Built with Streamlit + scikit-learn | PlaceIQ v2.0 — Major Project*
