# Loan Approval Prediction

Big Data Analysis course project (solo). A **learning prototype** that predicts historical loan approval
outcomes, compares interpretable classifiers, and audits class imbalance and fairness.

> Not a real lending decision system. It learns from past decisions, which may contain bias.

## Folder structure
```
loan-approval-prediction/
├── README.md
├── requirements.txt
├── .gitignore
├── data/            dataset (loan_data.csv) and its description
├── notebooks/       Colab/Jupyter notebooks (01_eda.ipynb, ...)
├── app/             Streamlit app (app.py)
├── images/          plots and screenshots for the report
└── report/          report and slides
```

## Dataset
Kaggle "Loan Prediction Problem Dataset", 614 rows, 13 columns. See `data/README.md`.

## Run the app
```
pip install -r requirements.txt
streamlit run app/app.py
```

## App features
Prediction with an action plan, model comparison, feature importance, fairness audit, cost-sensitive
threshold, proxy detection, bias mitigation, scalability benchmark, model card, research-paper methods
(KNN, SVM, AdaBoost, Naive Bayes, stacking, K-Best/RFE, cross-validation), and optional voice input/output.

## Progress
- [ ] 1. EDA (`notebooks/01_eda.ipynb`)
- [ ] 2. Cleaning and feature engineering
- [ ] 3. Models and evaluation
- [ ] 4. Feature importance, imbalance, fairness
- [ ] 5. App demo
- [ ] 6. Report and slides

## Results
(Add your results table and key plots here.)
