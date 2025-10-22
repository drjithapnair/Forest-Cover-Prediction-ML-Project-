# 🌲 Forest Cover Type Prediction using Machine Learning

This project predicts the **forest cover type** (the type of forest vegetation) based on cartographic variables such as elevation, slope, soil type, and wilderness area.  
It demonstrates an end-to-end **Data Science and Machine Learning workflow** — from data exploration and visualization to model training, evaluation, and saving the best model.

---

## 📘 Table of Contents

- [Project Overview](#project-overview)
- [Dataset Description](#dataset-description)
- [Project Workflow](#project-workflow)
- [Visualizations](#visualizations)
- [Model Training](#model-training)
- [Model Evaluation](#model-evaluation)
- [Saving the Best Model](#saving-the-best-model)
- [Requirements](#requirements)
- [How to Run](#how-to-run)
- [Results](#results)
- [Future Enhancements](#future-enhancements)
- [Author](#author)

---

## 🌳 Project Overview

The goal of this project is to build a machine learning model that can **accurately classify forest cover types** using various environmental and geographical factors.  
This problem is inspired by the **U.S. Forest Service’s dataset**, which includes observations from wilderness areas in Colorado.

---

## 📊 Dataset Description

- **File Name:** `train.csv`  
- **Target Variable:** `Cover_Type` (integer from 1 to 7)
- **Features:**
  - Elevation  
  - Aspect  
  - Slope  
  - Horizontal Distance to Hydrology  
  - Vertical Distance to Hydrology  
  - Horizontal Distance to Roadways  
  - Hillshade features  
  - Soil Type (categorical)  
  - Wilderness Area (categorical)

Each record represents a **30m x 30m patch** of forest land.

---

## 🧭 Project Workflow

1. **Data Loading and Inspection**
   - Load dataset using Pandas
   - Handle missing values and duplicates

2. **Data Cleaning**
   - Remove unnecessary or constant columns
   - Encode categorical variables if required

3. **Exploratory Data Analysis (EDA)**
   - Generate statistical summaries
   - Visualize feature distributions
   - Correlation analysis

4. **Feature Selection**
   - Apply `SelectKBest` with ANOVA F-test to select top features

5. **Model Building**
   - Train multiple machine learning algorithms:
     - Logistic Regression  
     - Random Forest  
     - Gradient Boosting  
     - Support Vector Machine (SVM)  
     - K-Nearest Neighbors (KNN)

6. **Model Evaluation**
   - Compare models using Accuracy Score

7. **Model Saving**
   - Save the best model using `joblib` for reuse

---

## 📈 Visualizations

Visualizations created using **Matplotlib** and **Seaborn** include:

- Histogram distributions for numeric features  
- Count plots for categorical features  
- Correlation heatmap  
- Pair plots for top correlated variables  
- Boxplots for outlier detection  
- Feature importance visualization (Random Forest)

---

## 🧠 Model Training

| Model | Library | Description |
|--------|----------|-------------|
| Logistic Regression | `sklearn.linear_model` | Linear baseline model |
| Random Forest | `sklearn.ensemble` | Bagging-based ensemble model |
| Gradient Boosting | `sklearn.ensemble` | Boosting-based ensemble model |
| SVM | `sklearn.svm` | Kernel-based classification |
| KNN | `sklearn.neighbors` | Distance-based classification |

**Train-Test Split:** 80% training, 20% testing

---

## 📊 Model Evaluation

The models were evaluated using **Accuracy Score**.  
Example output:

Logistic Regression: 0.72
Random Forest: 0.88
Gradient Boosting: 0.86
SVM: 0.70
KNN: 0.75

✅ Best Model: Random Forest with Accuracy = 0.88
🏁 Results

Best Model: Random Forest Classifier

Accuracy: Around 88–90%

Key Predictors: Elevation, Soil Type, Hillshade features, and Aspect.

🚀 Future Enhancements

Add XGBoost and LightGBM models

Implement GridSearchCV for hyperparameter tuning

Add Cross-validation for robust model evaluation

Deploy the trained model using Flask or Streamlit

Automate EDA and feature engineering steps
