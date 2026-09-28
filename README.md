# TrustML — An ML Model That Knows When Not to Predict

> **TrustML is a machine-learning reliability system designed to make predictions only when they are sufficiently reliable.**

## 📌 Overview

Machine-learning models can produce predictions even when they are uncertain or when they encounter inputs that are significantly different from the data they were trained on.

TrustML investigates whether the reliability of an ML system can be improved by allowing it to **reject predictions when it is uncertain or unfamiliar with the input**.

The system combines:

* A conventional machine-learning prediction model
* Probability/confidence estimation
* Probability calibration
* Out-of-distribution (OOD) / anomaly detection
* A prediction rejection mechanism
* Explainability using SHAP
* A Streamlit interface

The project is designed as a genuine ML system rather than a chatbot or LLM-based application.

---

## 🎯 Research Question

> **Can we improve the reliability of an ML system by allowing it to reject predictions when it is uncertain or unfamiliar with the input?**

The main experiment compares:

1. A normal ML model
2. An ML model with calibrated probabilities
3. TrustML with calibration, OOD detection, and a reject option

The key question is:

> **When TrustML rejects uncertain or unfamiliar cases, does the error rate among accepted predictions decrease?**

---

## 🧠 System Architecture

```text
                    ┌─────────────────┐
                    │    Input Data   │
                    └────────┬────────┘
                             ↓
                    ┌─────────────────┐
                    │ Preprocessing   │
                    └────────┬────────┘
                             ↓
                    ┌─────────────────┐
                    │   ML Model      │
                    └────────┬────────┘
                             ↓
                    ┌─────────────────┐
                    │ Raw Probability │
                    └────────┬────────┘
                             ↓
                    ┌─────────────────┐
                    │   Calibration   │
                    └────────┬────────┘
                             ↓
              ┌──────────────┴──────────────┐
              ↓                             ↓
       Confidence                    OOD Detection
              │                             │
              └──────────────┬──────────────┘
                             ↓
                    ┌─────────────────┐
                    │ Reject Decision │
                    └────────┬────────┘
                             ↓
                  ┌──────────┴──────────┐
                  ↓                     ↓
               ACCEPT                 REJECT
                  ↓                     ↓
          Prediction +            Unreliable
          Confidence              + Reason
                  │
                  ↓
             Explainability
                  │
                  ↓
               Streamlit
```

---

## 📊 Dataset

**Dataset:** AI4I 2020 Predictive Maintenance Dataset

The project uses a real-world predictive-maintenance problem involving machine operating conditions and machine failures.

### Prediction task

The model will predict whether a machine failure occurs.

```text
0 → No Failure
1 → Failure
```

### Features

The project will investigate machine and operating characteristics such as:

* Machine type
* Air temperature
* Process temperature
* Rotational speed
* Torque
* Tool wear

Features that directly reveal the target or create data leakage will not be used as predictive inputs.

> Dataset details and final feature selection will be documented after the EDA stage.

---

## 🤖 Machine Learning Approach

### 1. Baseline Model

The first stage is to train conventional ML models and establish a reliable baseline.

Candidate models include:

* Logistic Regression
* Random Forest
* XGBoost
* SVM where appropriate

The final baseline model will be selected based on experimental performance and suitability for the problem.

---

### 2. Probability Calibration

A model's predicted probability should not automatically be interpreted as its actual probability of being correct.

TrustML will compare:

```text
Uncalibrated Model
        vs
Calibrated Model
```

Possible calibration methods include:

* Sigmoid / Platt scaling
* Isotonic regression

Calibration will be evaluated using measures such as:

* Calibration curves
* Reliability diagrams
* Brier score
* Confidence distributions

---

### 3. OOD / Anomaly Detection

TrustML will investigate whether a new input is sufficiently similar to the training distribution.

The initial approach will use an anomaly/OOD detection method such as:

* Isolation Forest
* Distance-based detection where appropriate

The project will avoid unnecessary algorithmic complexity and focus on experimentally evaluating a practical approach.

---

### 4. Prediction Rejection

The final TrustML decision combines confidence and input familiarity.

Conceptually:

```text
IF input is familiar
AND calibrated confidence >= threshold:

    ACCEPT prediction

ELSE:

    REJECT prediction
```

A rejected prediction will include a reason such as:

```text
Prediction: NORMAL
Confidence: 61%
Reliability: LOW
Status: REJECTED

Reason:
Input is significantly different from the training distribution.
```

---

## 📈 Evaluation

TrustML will compare three systems:

### Model A — Normal ML

```text
Input → ML Model → Prediction
```

### Model B — Calibrated ML

```text
Input → ML Model → Calibration → Prediction
```

### Model C — TrustML

```text
Input
  ↓
ML Model
  ↓
Calibration
  ↓
OOD Detection
  ↓
Reject / Accept
```

### Metrics

For classification, the project will evaluate:

* Accuracy
* Precision
* Recall
* F1-score
* ROC-AUC where appropriate
* PR-AUC
* Confusion matrix
* Calibration quality
* Coverage
* Error rate among accepted predictions
* Rejected/uncertain percentage

### Selective prediction

A major evaluation will measure whether rejecting uncertain predictions improves reliability among accepted predictions.

```text
Coverage =
Accepted Predictions / Total Predictions
```

The project will investigate the trade-off between:

```text
Higher Coverage
      ↕
Prediction Reliability
```

---

## 🔬 Threshold Experiment

TrustML will experiment with different confidence/rejection thresholds.

Example:

```text
0.50
0.60
0.70
0.80
0.90
```

For each threshold, the project will measure:

* Coverage
* Accuracy of accepted predictions
* Rejected percentage

The goal is to identify a sensible operating point rather than simply maximizing accuracy.

---

## 🔎 Explainability

SHAP will be used to help explain model predictions.

For accepted predictions, TrustML will show features that contributed most to the model's prediction.

For rejected predictions, the system will provide an understandable reason for rejection, such as:

* Low calibrated confidence
* Input significantly different from training data

SHAP explanations will be treated as explanations of model behavior and **not as proof of causation**.

---

## 🖥️ Streamlit Application

The final application will provide a simple interface where users can enter or upload model input features.

The application will display:

```text
Prediction: NORMAL

Confidence: 93%

Reliability: HIGH

Status: ✓ Prediction accepted

Top contributing features:
1. Temperature
2. Pressure
3. Vibration
```

For an unreliable input:

```text
Prediction: NORMAL

Confidence: 61%

Reliability: LOW

Status: ⚠ Prediction rejected

Reason:
Input is significantly different from
the training distribution.

Recommendation:
Do not rely solely on this prediction.
```

The interface will prioritize clarity rather than visual complexity.

---

## 📁 Project Structure

```text
Trust-ML/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_baseline_models.ipynb
│   ├── 03_calibration.ipynb
│   └── 04_ood_detection.ipynb
│
├── src/
│   ├── __init__.py
│   ├── data_preprocessing.py
│   ├── feature_engineering.py
│   ├── train.py
│   ├── calibration.py
│   ├── ood_detection.py
│   ├── evaluation.py
│   └── predict.py
│
├── models/
│
├── results/
│
├── app/
│   └── app.py
│
├── tests/
│
├── requirements.txt
├── .gitignore
└── README.md
```

### Directory purposes

| Directory/File     | Purpose                                |
| ------------------ | -------------------------------------- |
| `data/raw/`        | Original dataset                       |
| `data/processed/`  | Cleaned/processed data                 |
| `notebooks/`       | EDA and experiments                    |
| `src/`             | Reusable ML source code                |
| `models/`          | Saved trained models                   |
| `results/`         | Metrics, graphs and experiment results |
| `app/`             | Streamlit application                  |
| `tests/`           | Automated tests                        |
| `requirements.txt` | Python dependencies                    |
| `.gitignore`       | Files excluded from Git                |
| `README.md`        | Project documentation                  |

---

## 🧪 Testing

Tests will cover:

* Data preprocessing
* Feature engineering
* Model prediction
* Probability calibration
* OOD detection
* Rejection logic

Test cases will include:

* Normal input
* High-confidence input
* Low-confidence input
* Unfamiliar input
* Missing input
* Invalid input

---

## 🗓️ Development Plan

### Week 1 — Data + Baseline ML

* Dataset analysis
* EDA
* Missing-value analysis
* Duplicate/outlier analysis
* Feature engineering
* Train/validation/test split
* Baseline ML models
* Initial evaluation

### Week 2 — Model Reliability

* Probability/confidence analysis
* Probability calibration
* Calibration comparison
* Reliability diagrams
* Brier score
* Confidence analysis

### Week 3 — OOD + Rejection

* OOD/anomaly detection
* Isolation Forest or another suitable method
* Rejection mechanism
* Threshold experiments
* Coverage analysis
* Explainability

### Week 4 — Finalization

* Final model
* Final experiments
* Streamlit application
* Testing
* Documentation
* Architecture diagram
* Results
* Screenshots
* Final presentation/demo

---

## 👥 Team

### Developer A — ML / Data

Responsible primarily for:

* Dataset
* EDA
* Preprocessing
* Feature engineering
* Baseline models
* Model tuning
* Calibration
* Evaluation
* Final technical results

### Developer B — Reliability / Engineering

Responsible primarily for:

* Project setup
* GitHub workflow
* Pipeline support
* Calibration implementation support
* OOD detection
* Rejection mechanism
* Explainability
* Streamlit
* Testing
* Documentation

Both developers will understand the complete system and contribute meaningful commits.

---

## 🛠️ Tech Stack

* Python 3.11
* Pandas
* NumPy
* Scikit-learn
* XGBoost
* Matplotlib
* Seaborn
* SciPy
* SHAP
* Streamlit
* Jupyter Notebook
* Git
* GitHub

---

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/prarthanaa11/Trust-ML.git
cd Trust-ML
```

### 2. Create a virtual environment

Windows PowerShell:

```powershell
py -3.11 -m venv .venv
```

### 3. Activate it

```powershell
.\.venv\Scripts\Activate.ps1
```

### 4. Install dependencies

```powershell
pip install -r requirements.txt
```

### 5. Run the application

```powershell
streamlit run app/app.py
```

> The application will be completed during the development process.

---

## 🔀 Git Workflow

The project uses a simple feature-branch workflow:

```text
main
 ↓
feature branch
 ↓
Pull Request
 ↓
Code Review
 ↓
Merge
```

Example branches:

```text
feature/eda
feature/baseline-model
feature/calibration
feature/ood-detection
feature/streamlit
feature/testing
```

Major milestones:

```text
v0.1-baseline
v0.2-calibration
v0.3-trustml
v1.0-final
```

---

## ⚠️ Project Principles

TrustML will:

* Compare multiple models
* Use proper train/validation/test separation
* Prevent data leakage
* Evaluate more than accuracy
* Perform reproducible experiments
* Document failures and limitations
* Explain design decisions
* Keep the system focused on ML reliability

TrustML will **not**:

* Treat confidence as equivalent to correctness
* Train and test on the same data
* Leak test data into training
* Rely on an LLM as the main AI system
* Build only a user interface
* Implement unnecessary algorithms
* Spend excessive time on frontend design

---

## 📌 Current Status

**Project stage:** Initial setup

* [x] GitHub repository created
* [x] Python environment configured
* [x] Project repository initialized
* [x] GitHub remote configured
* [x] Initial project pushed to GitHub
* [ ] Dataset finalized
* [ ] EDA
* [ ] Baseline model
* [ ] Calibration
* [ ] OOD detection
* [ ] Rejection mechanism
* [ ] Explainability
* [ ] Streamlit application
* [ ] Testing
* [ ] Final evaluation

---

## 🚀 Final Goal

By the end of the project, TrustML should demonstrate:

> **A machine-learning system that does not blindly make predictions, but evaluates whether a prediction should be trusted before presenting it to the user.**
