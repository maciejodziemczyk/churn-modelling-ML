# Churn modelling (Machine Learning)
Project created for *Practical Machine Learning in Python* (org. *Praktyczny Macine Learning w Pythonie*) classes at WNE UW

Language:
 * Polish - classes, notebook

Semester: II (MA studies)

## About
The main objective of this project was to gain some experience, practice and perform own experiments with Machine Learning models learned during classes. The idea was to show my ML journey with this dataset. Moreover I gained some expert knowledge about churn events while working on this project, which is very cool for me (mostly during feature engineering part). The project was divided into 3 parts + one presentation notebook (used during project defense).<br>
1. In the first part you can see:
 - basic data preprocessing and analysis (s.t data structure, variables, target balance, correlation, null/missing check, variables mean equality between target classes, label encoding for tree-based models)
 - first Logit, Random Forest and XGBoost with hyperparameters optimization using hand coded random search in k-Fold Cross Validation wrapper
 - basic feature importance analysis and feature engineering
2. In the second part you can see:
 - complex EDA and feature engineering (distribution analysis, non linear transformations, quantile splitting, normalization or unitarization for continuous variables, cross tables for discrete variables)
 - feature selection using Mutual Information, Spearman Correlation coef, General to Specyfic procedure 
 - more feature engineering (interactions)
 - logit reestimation (random forest performance level after EDA)
 - some experiments with SVM and kNN
3. In the third part you can see:
 - Neural Network (Feedforward MLP) development (optimizers, batch size, activation functions, regularization, dropout, experiments with architecture, cross validation)
 - the best models comparison
 - logit y~X, where y is target variable and X is all models predictions matrix to find the best model and gain an intuition about ensembling and votings weights
 - bootstrap simulation to find best ensembling weights (voting)
4. In the "zero" -th notebook you can see main conclusions and stuff used during defense.

Findings:
 - EDA is such a complex process but very important, even simple models with deep EDA and feature engineering can obtain good results
 - tree-based models are robust to monotonic transformations and basic feature engineering (reasonable)
 - XGBoost is extremely good algorithm in this problem, even ensembling gave worse results
 - tree-based models are very easy to overfit
 - neural networks are extremely powerful but require more experience and knowledge 

In this project I learnd a ton of machine learning models and end-to-end process, data analysis, feature engineering and how to think about it. I gained more experience and practice with Python. It is very cool project I think, and my first such a bich ML analysis. This classes was one of the best I have ever participated.  

## Repository description
 - 0 - Prezentacja.ipynb - Jupyter notebook with project brief presentation
 - 1 - Pierwszy Logit, Random Forest, XGB.ipynb - Jupyter notebook with first part of the analysis
 - 2 - Pogłębione EDA, feature engineering, powrót do logitu, RF, XGB.ipynb - Jupyter notebook with second part of the analysis
 - 3 - Sieci neuronowe i model finalny.ipynb - Jupyter notebook with third part of the analysis
 - Churn_Modelling.csv - dataset found on kaggle.com
 - df.p and dffin.p - different variants of dataset (depending on EDA and Feature engineering)
 - hpl.ipynb and hpl.py - helper functions used during analysis (in notebook format and python script)
 - model_XX_.py - saved models (dictionary of specifications and predictions)
 - symulacja.p - bootstrap ensembling simulation results

## Technologies
 - Python (numpy, pandas, matplotlib.pyplot, seabor, statsmodels, scikit-learn, scipy, keras, xgboost)
 - Jupyter Notebook

## Author
Maciej Odziemczyk

##
Note
Some of the scripts and functions or ideas used in this project was developed by my teacher (dr Maciej Wilamowski)
