# Diabetes Predictor Model

A machine learning project for predicting whether a patient is likely to be **diabetic** or **not diabetic** based on common medical input features. The project includes a trained classification pipeline and a **Gradio** web interface for interactive testing.

## Live Demo

Hugging Face Space: [Diabetes Predictor Model](https://huggingface.co/spaces/mastermindfa/Diabetes_Predictor_Model)

## Project Overview

This project uses the **Pima Indians Diabetes dataset** style feature set to build a binary classification model.

The app takes the following patient information as input:

- Pregnancies
- Glucose
- Blood Pressure
- Skin Thickness
- Insulin
- BMI
- Diabetes Pedigree Function
- Age

Then it predicts:

- **Diabetic**
- **Not Diabetic**

It also shows the **probability of diabetes** and uses a learned **decision threshold** instead of relying only on the default `0.50` cutoff.

## Main Improvements in This Version

This version keeps the original structure mostly unchanged, but fixes the core prediction logic where needed:

- Uses `predict_proba()` for probability-based prediction
- Applies a learned threshold selected from the validation/test evaluation stage
- Fixes the issue where predictions were always leaning toward **Not Diabetic**
- Uses **authentic random samples** from the dataset in the Gradio interface
- Avoids unrealistic random values by selecting medically valid rows for the sample-fill button

## Model Pipeline

The model is built with a Scikit-learn pipeline that includes:

1. **Zero-to-NaN conversion** for medically invalid zero values in selected columns
2. **Missing value imputation** using median values
3. **Feature engineering**, including:
   - `BMI_Age`
   - `Glucose_BMI`
   - `Preg_Age_Ratio`
4. **Outlier capping** using 1st and 99th percentiles
5. **Feature scaling** with `StandardScaler`
6. **Classification** using `LogisticRegression`

## Training and Evaluation

The notebook/script includes:

- Train-test split with stratification
- 5-fold stratified cross-validation
- Hyperparameter tuning with `GridSearchCV`
- Evaluation using:
  - Accuracy
  - Precision
  - Recall
  - F1-score
  - ROC-AUC
  - Confusion Matrix
  - Classification Report

## Gradio Interface

The web app provides:

- Slider-based patient input
- A **Fill Random Sample** button
- Prediction output with:
  - Final class label
  - Probability of diabetes
  - Decision threshold used

The **Fill Random Sample** button uses actual dataset rows instead of fake/randomly generated values.

## Files

A typical project structure is:

```bash
.
├── app.py
├── diabetes.csv
├── diabetes_model_pipeline.pkl
├── requirements.txt
└── README.md
```

If you are using the updated training script version locally, you may also have:

```bash
./diabetespredictionmodel_fixed.py
```

## Installation

Clone the repository and install dependencies:

```bash
git clone <https://github.com/mastermind-fa/DiabetesPredictionModel>
cd <DiabetesPredictionModel>
pip install -r requirements.txt
```

## Run Locally

If your app entry file is `app.py`:

```bash
python app.py
```

If you are running the updated local script directly:

```bash
python diabetespredictionmodel_fixed.py
```

## Requirements

Main libraries used:

- pandas
- numpy
- matplotlib
- scikit-learn
- gradio
- joblib

## Use Case

This project is useful for:

- Learning end-to-end ML workflow
- Practicing healthcare classification problems
- Demonstrating model deployment with Gradio
- Hosting ML apps on Hugging Face Spaces

## Important Note

This project is for **educational and demonstration purposes only**. It is **not a medical diagnosis tool** and should not be used as a substitute for professional clinical judgment.

## Author

**Farhana Alam / mastermindfa**

## Hugging Face Space

You can view the deployed version here:

[https://huggingface.co/spaces/mastermindfa/Diabetes_Predictor_Model](https://huggingface.co/spaces/mastermindfa/Diabetes_Predictor_Model)
