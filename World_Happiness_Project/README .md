# World Happiness Score Prediction using Regression

SI Lab Project, Semester V, B.Tech. CSE (AI & Data Science), Ramdeobaba University, Nagpur (2026-27)

**Student:** Sanika Bhoyar | **Roll No.:** 22

## About the project
A machine learning project that estimates a country's happiness score from the indicators in the
World Happiness Report 2015 (158 countries). It covers web scraping, data cleaning, descriptive and
inferential statistics, visualisation, regression and classification models, model saving, and a
Streamlit dashboard.

**Features used:** Economy (GDP per Capita), Family, Health (Life Expectancy), Freedom,
Trust (Government Corruption), Generosity, Dystopia Residual.
**Target:** Happiness Score (regression) and High/Low happiness, split at the median (classification).

## Results (80:20 split, random_state=42, 32 test countries)

| Regression model | R² | MAE | RMSE |
|---|---|---|---|
| Linear Regression (selected) | 1.0000 | 0.0002 | 0.0003 |
| Ridge | 0.9971 | 0.0493 | 0.0643 |
| Gradient Boosting | 0.9648 | 0.1773 | 0.2236 |
| Random Forest | 0.9402 | 0.2224 | 0.2912 |
| Decision Tree | 0.8163 | 0.4101 | 0.5106 |
| Lasso | -0.0051 | 1.0290 | 1.1942 |

| Classification model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| Random Forest (selected) | 0.906 | 0.92 | 0.91 | 0.91 |
| Logistic Regression | 0.875 | 0.90 | 0.88 | 0.87 |
| Decision Tree | 0.781 | 0.78 | 0.78 | 0.78 |

> Note: in the 2015 report the happiness score is the sum of its component columns, so a linear model
> recovers it almost exactly. This confirms how the index is built; it is not a forecast for new years.

## Repository structure
```
world-happiness-prediction/
├── README.md
├── requirements.txt
├── .gitignore
├── notebook/
│   └── SI_World_happiness_project.ipynb
├── app/
│   └── happiness_dashboard_v2.py
├── data/
│   ├── 2015_world_happiness.csv
│   ├── 2015-selected-columns.csv
│   └── happiness_scraped.csv
├── models/
│   └── happiness_model.pkl
├── report/
│   └── World_Happiness_SI_Lab_Report.docx
└── screenshots/
    ├── dashboard_input.png
    └── dashboard_prediction.png
```

## How to run

1. Clone and install
```bash
git clone <your-repo-url>
cd world-happiness-prediction
python -m venv venv
venv\Scripts\activate          # Windows  (Linux/Mac: source venv/bin/activate)
pip install -r requirements.txt
```

2. Notebook: open `notebook/SI_World_happiness_project.ipynb` in Jupyter or Google Colab. Upload the
CSV files from `data/` to the working folder first (the notebook reads them by file name).
Colab link: <paste your Colab link here>

3. Dashboard: the app reads `2015-selected-columns.csv` from the folder it is run in, so run it from `data/`:
```bash
cd data
streamlit run ../app/happiness_dashboard_v2.py
```
Alternatively, run it from anywhere and upload the CSV using the sidebar uploader.

## Dashboard
The Streamlit app ("Happiness Insights") has five tabs: Overview & Graphs, Regression Comparison,
High/Low Classification (with confusion matrix), Predict Happiness (Linear Regression) and Dataset.
It trains its own models at start-up and does not load `happiness_model.pkl`; that file is the model
saved in the notebook (trained on the scraped Wikipedia table).

## Tech stack
Python, Pandas, NumPy, SciPy, scikit-learn, Matplotlib, Seaborn, Requests, Joblib, Streamlit, Google Colab.

## Data sources
- World Happiness Report 2015 dataset: https://www.kaggle.com/datasets/unsdsn/world-happiness
- World Happiness Report (Wikipedia, scraped table): https://en.wikipedia.org/wiki/World_Happiness_Report
