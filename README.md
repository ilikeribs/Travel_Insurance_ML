# Module 3 - Sprint 1 Supervised Machine Learning Fundamentals 

This project is a part of the Turing College Data Science learning programme at (https://www.turingcollege.com/data-science). Projects outline can be found at the [main GitHub repo](https://github.com/TuringCollegeSubmissions/vbeino-ML.1.5.git).

#### -- Project Status: [Active]

## Project Intro/Objective
The purpose of this project is to develop and practice supervised machine learning skills as part of the Data Science programme curriculum. For the provided Travel Insurance dataset a task was provided to generate an efficient machine learning model which would assist in classifying potential customers for Travel  Insurance.

### Technologies
* Python
* Pandas, Jupyter
* Scikit-Learn

## Project Description
For the given Travel Insurance dataset (https://www.kaggle.com/datasets/tejashvi14travel-insurance-prediction-data) EDA analysis was performed, following hypotheses tested using Chi2 method:

- Is there statistically significant relationship between the presence of chronic diseases and travel insurance purchase features.
- Is there a relationship between graduate status and travel insurance purchase features.

Pipelines were created for following supervised classification algorithms tests using default values: (Logistic Regression, Decision trees, Random Forest, Support Vector Machines (SVM), K-Nearest Neighbors (KNN), Naive Bayes). 

Random Forest, KNN and Radial-SVM algorithms were further optimized with hyperparameter tuning and assembled into a Voting Classifier ensemble model. The ensemble model was further tuned and tested on the holdout dataset portion with following macro results:

- Precision: 82%
- Recall: 77%
- F1: 78%
- Accuracy:81% 

## Needs of this project

- Learning purposes

## Getting Started

1. Clone this repo (for help see this [tutorial](https://github.com/TuringCollegeSubmissions/vbeino-ML.1.5.git)).
2. Pip install requirements.txt
3. Run Travel_Insurance.ipynb


## Author 

**Lead : [Vytautas Beinoravicius ]**

