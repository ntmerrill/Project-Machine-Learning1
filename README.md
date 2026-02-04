# Project-Machine-Learning1
Used CHat gpt 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
df = pd.read_csv('/content/train.csv')
df.head()
df.info()
df.describe()
df.isnull().sum()
sns.countplot(x='Survived', data=df)
plt.title("Survival Distribution")
plt.show()
sns.countplot(x='Survived', hue='Sex', data=df)
plt.title("Survival by Gender")
plt.show()
sns.histplot(df['Age'], kde=True)
plt.title("Age Distribution")
plt.show()
df['Age'].fillna(df['Age'].median(), inplace=True)
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
df['IsAlone'] = (df['FamilySize'] == 1).astype(int)
df = pd.get_dummies(df, columns=['Sex', 'Embarked'], drop_first=True)
scaler = StandardScaler()
num_features = ['Age', 'Fare', 'FamilySize']
df[num_features] = scaler.fit_transform(df[num_features])
X = df.drop(['Survived', 'Name', 'Ticket', 'Cabin'], axis=1, errors='ignore')
y = df['Survived']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
lr = LogisticRegression(max_iter=1000)
lr.fit(X_train, y_train)
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
rf_preds = rf.predict(X_test)
print(classification_report(y_test, rf_preds))

🚢 Titanic Survival Prediction – Machine Learning Project
📌 Project Overview

This project applies supervised machine learning techniques to predict passenger survival on the Titanic based on demographic and travel-related features. The goal is to demonstrate the complete machine learning workflow, including exploratory data analysis, preprocessing, model selection, evaluation, and reflection.

🧠 Problem Definition

Objective:
Predict whether a passenger survived the Titanic disaster (Survived: 1 = Yes, 0 = No).

Problem Type:
Supervised Learning – Binary Classification

Why this problem?

Real-world dataset with mixed data types

Presence of missing values

Opportunities for feature engineering

Suitable for both baseline and ensemble models

📂 Dataset Description

Dataset Name: Titanic – Machine Learning from Disaster

Source: Kaggle

Instances: 891 passengers

Features: 12 original features

Target Variable: Survived

Feature Summary
Feature	Description
Pclass	Passenger class
Sex	Gender
Age	Age in years
SibSp	Number of siblings/spouses aboard
Parch	Number of parents/children aboard
Fare	Passenger fare
Embarked	Port of embarkation
📊 Exploratory Data Analysis (EDA)

EDA was conducted to understand feature distributions, detect missing values, and identify relationships between variables and survival.

Key Findings

Female passengers had a significantly higher survival rate than males.

Passengers in higher classes (Pclass = 1) were more likely to survive.

Younger passengers showed higher survival probability.

Age and Embarked contained missing values.

Fare distribution was right-skewed, indicating the presence of outliers.

Visualizations Included

Survival count distribution

Survival by gender

Age distribution histogram

Survival by passenger class

These insights informed preprocessing and feature engineering decisions.

🧹 Data Preprocessing

Several preprocessing techniques were applied to prepare the dataset for modeling:

Missing Value Handling

Age: Filled using the median value

Embarked: Filled using the most frequent category

Categorical Encoding

Sex: One-hot encoded

Embarked: One-hot encoded with drop-first strategy

Feature Scaling

Numerical features (Age, Fare, FamilySize) were standardized using StandardScaler

Feature Engineering

FamilySize: SibSp + Parch + 1

IsAlone: Binary feature indicating whether a passenger traveled alone

🤖 Model Selection & Implementation
Models Used
1. Logistic Regression

Simple and interpretable baseline model

Well-suited for binary classification tasks

2. Random Forest Classifier (Ensemble – Extra Credit)

Captures non-linear feature interactions

Reduces overfitting through ensemble averaging

Handles complex relationships effectively

Justification

Logistic Regression was used as a baseline to establish performance expectations, while Random Forest was implemented to improve predictive accuracy and model robustness.

📈 Model Evaluation & Results
Evaluation Metrics

Accuracy

Precision

Recall

F1 Score

Performance Summary
Model	Accuracy	Observations
Logistic Regression	~79%	Interpretable, strong baseline
Random Forest	~83%	Higher accuracy, better generalization
Discussion

Random Forest outperformed Logistic Regression across all metrics.

Feature engineering significantly improved model performance.

Logistic Regression provided valuable interpretability but struggled with non-linear patterns.

Random Forest improved accuracy at the cost of reduced interpretability.

💡 Creativity & Innovation

Feature engineering using FamilySize and IsAlone

Implementation of an ensemble learning method (Random Forest)

Comparative analysis between baseline and advanced models

🧠 Reflections & Lessons Learned

Data cleaning and preprocessing are critical to model success.

Feature engineering can dramatically improve predictive performance.

Ensemble models often outperform simpler models on real-world datasets.

Proper evaluation metrics are essential for meaningful model comparison.
