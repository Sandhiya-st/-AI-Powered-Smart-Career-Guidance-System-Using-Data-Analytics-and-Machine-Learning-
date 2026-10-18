# 🎓 PlaceIQ – AI-Powered Smart Career Guidance System

**MCA Data Science Major Project**

PlaceIQ is a machine learning-based placement prediction and career guidance system designed to help students understand their placement readiness, identify skill gaps, explore career opportunities, and improve their employability.

## 🚀 Features

- **Placement Prediction:** Uses multiple machine learning models to predict student placement outcomes.
- **Model Evaluation:** Compares models using accuracy, cross-validation, confusion matrices, classification reports, and ROC-AUC.
- **Career Recommendations:** Suggests suitable career paths based on student profiles.
- **Skill Gap Analysis:** Identifies strengths and areas for improvement.
- **Resume Analyzer:** Provides an estimated ATS compatibility score, keyword matching, and resume structure feedback.
- **Salary Estimation:** Estimates potential salary using a regression model trained on generated salary targets.
- **Analytics Dashboard:** Visualizes placement trends and student performance.
- **Database Integration:** Uses SQLite to store user information, predictions, and resume scores.
- **Role-Based Access:** Includes student, placement officer, and administrator interfaces.

## 🛠️ Technologies Used

- Python
- Streamlit
- Pandas and NumPy
- Scikit-learn
- Matplotlib and Seaborn
- SQLite
- Natural Language Processing (NLP) techniques

## 🤖 Machine Learning Models

The project includes seven classification models:

1. Logistic Regression
2. Decision Tree
3. Random Forest
4. Gradient Boosting
5. Support Vector Machine (SVM)
6. K-Nearest Neighbors (KNN)
7. Artificial Neural Network (ANN)

The models are evaluated to compare their performance on the placement prediction task.

## 📂 Project Structure

```text
placement_major_project/
├── app.py
├── Placement_Prediction_data.csv
├── requirements.txt
└── README.md
```

The SQLite database (`placeiq.db`) is created automatically when the application initializes.

## ⚙️ Installation and Setup

### 1. Clone the repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd YOUR_REPOSITORY_FOLDER
```

Replace the example values with your actual GitHub repository URL and folder name.

### 2. Create a virtual environment

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the application

```bash
streamlit run app.py
```

Open the local URL displayed in your terminal, usually `http://localhost:8501`.

## 📊 Dataset

The project uses `Placement_Prediction_data.csv`, which contains student academic, skills, internship, project, and placement-related information.

The dataset is used to train and evaluate machine learning models for placement outcome prediction.

## 🎯 Project Objectives

- Predict student placement outcomes using machine learning.
- Help students understand their career readiness.
- Identify skill gaps and suggest areas for improvement.
- Provide resume feedback to support job applications.
- Present placement-related insights through interactive visualizations.

## 🔮 Future Enhancements

- Deploy the application online.
- Integrate a more advanced career recommendation engine.
- Improve resume analysis using advanced NLP techniques.
- Evaluate salary prediction using real salary data.
- Add more datasets and validate model performance on unseen data.

## 👩‍💻 Author

**Sandhiya S**  
MCA – Data Science Graduate  
Takshashila University

---

*Developed as an academic major project combining machine learning, data analytics, and career guidance.*
