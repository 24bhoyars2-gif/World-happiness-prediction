import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso, LogisticRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor
from sklearn.metrics import (
    r2_score, mean_absolute_error, mean_squared_error,
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

# -------------------- Page setup --------------------
st.set_page_config(
    page_title="Happiness Insights | World Happiness 2015",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------- Styling --------------------
st.markdown("""
<style>
    .stApp { background: #0b1220; color: #f3f4f6; }
    [data-testid="stSidebar"] { background: #111b2e; }
    h1, h2, h3, p, label { color: #f3f4f6; }
    .hero {
        padding: 1.3rem 1.5rem; border-radius: 18px;
        background: linear-gradient(115deg, #182a49, #243b55);
        border: 1px solid #334b6b; margin-bottom: 1rem;
    }
    .hero p { color: #cbd5e1; margin-bottom: 0; }
    div[data-testid="stMetric"] {
        background: #142238; border: 1px solid #2c405c;
        padding: 14px 16px; border-radius: 14px;
    }
    div[data-testid="stMetricLabel"] p { color: #cbd5e1 !important; }
    div[data-testid="stMetricValue"] { color: #ffffff !important; }
    div[data-testid="stMetricDelta"] { color: #a7f3d0 !important; }
    .small-note { color: #aebdd0; font-size: 0.9rem; }
    div[data-testid="stTabs"] button { color: #dbeafe; }
</style>
""", unsafe_allow_html=True)

# -------------------- Load dataset --------------------
@st.cache_data
def load_data(uploaded_file=None):
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
    else:
        # Works when the CSV is in the same folder as this app.
        data = pd.read_csv("2015_world_happiness.csv")

    data.columns = data.columns.str.strip()
    data = data.drop_duplicates()
    data = data.dropna(subset=["Happiness Score"])
    numeric_cols = data.select_dtypes(include="number").columns
    for col in numeric_cols:
        data[col] = data[col].fillna(data[col].median())
    return data

st.sidebar.title("🌍 Happiness Insights")
st.sidebar.caption("World Happiness Report • 2015")
uploaded = st.sidebar.file_uploader(
    "Upload the CSV if it is not in the app folder",
    type=["csv"]
)

try:
    df = load_data(uploaded)
except FileNotFoundError:
    st.title("🌍 Happiness Insights")
    st.error("Dataset file not found. Keep `2015_world_happiness.csv` in the same folder as `happiness_dashboard.py`, or upload it using the sidebar.")
    st.stop()
except Exception as e:
    st.error(f"Could not load the dataset: {e}")
    st.stop()

required = ["Country", "Region", "Happiness Rank", "Happiness Score"]
missing = [c for c in required if c not in df.columns]
if missing:
    st.error("Required columns missing from dataset: " + ", ".join(missing))
    st.stop()

# Use only factors that exist in the user's selected-columns CSV.
candidate_features = [
    "Economy (GDP per Capita)", "Family", "Health (Life Expectancy)",
    "Freedom", "Trust (Government Corruption)", "Generosity",
    "Dystopia Residual"
]
features = [c for c in candidate_features if c in df.columns]
if not features:
    st.error("No supported happiness-factor columns were found in the CSV.")
    st.stop()

# -------------------- Train models dynamically --------------------
@st.cache_resource
def train_models(data, feature_cols):
    X = data[feature_cols].copy()
    y_reg = data["Happiness Score"].astype(float)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_reg, test_size=0.2, random_state=42
    )

    regressors = {
        "Linear Regression": LinearRegression(),
        "Ridge": Ridge(),
        "Lasso": Lasso(max_iter=10000),
        "Decision Tree": DecisionTreeRegressor(random_state=42),
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
        "Gradient Boosting": GradientBoostingRegressor(random_state=42)
    }
    reg_rows, reg_fitted = [], {}
    for name, model in regressors.items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        reg_rows.append({
            "Model": name,
            "R²": r2_score(y_test, pred),
            "MAE": mean_absolute_error(y_test, pred),
            "RMSE": np.sqrt(mean_squared_error(y_test, pred))
        })
        reg_fitted[name] = model
    reg_results = pd.DataFrame(reg_rows).sort_values("R²", ascending=False).reset_index(drop=True)

    # High/Low is defined from the full dataset median, matching the notebook logic.
    threshold = float(data["Happiness Score"].median())
    y_class = (data["Happiness Score"] >= threshold).astype(int)
    Xc_train, Xc_test, yc_train, yc_test = train_test_split(
        X, y_class, test_size=0.2, random_state=42, stratify=y_class
    )
    classifiers = {
        "Logistic Regression": LogisticRegression(max_iter=2000),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42)
    }
    cls_rows, cls_fitted, cls_predictions = [], {}, {}
    for name, model in classifiers.items():
        model.fit(Xc_train, yc_train)
        pred = model.predict(Xc_test)
        cls_rows.append({
            "Model": name,
            "Accuracy (%)": accuracy_score(yc_test, pred) * 100,
            "Precision": precision_score(yc_test, pred, zero_division=0),
            "Recall": recall_score(yc_test, pred, zero_division=0),
            "F1-score": f1_score(yc_test, pred, zero_division=0)
        })
        cls_fitted[name] = model
        cls_predictions[name] = (yc_test, pred)
    cls_results = pd.DataFrame(cls_rows).sort_values("Accuracy (%)", ascending=False).reset_index(drop=True)
    return reg_results, reg_fitted, cls_results, cls_fitted, cls_predictions, threshold, X_test, y_test

reg_results, reg_models, cls_results, cls_models, cls_predictions, threshold, X_test, y_test = train_models(df, features)

# -------------------- Header and filters --------------------
st.markdown("""
<div class="hero">
  <h1>🌍 Happiness Insights</h1>
  <p>Explore country happiness, compare machine-learning models, and estimate a happiness score using selected factors.</p>
</div>
""", unsafe_allow_html=True)

regions = ["All regions"] + sorted(df["Region"].dropna().unique().tolist())
selected_region = st.sidebar.selectbox("Filter by region", regions)
view = df if selected_region == "All regions" else df[df["Region"] == selected_region]
st.sidebar.markdown("---")
st.sidebar.caption("Model evaluation uses an 80/20 train-test split and random_state=42.")

# -------------------- Overview metrics --------------------
avg_score = float(view["Happiness Score"].mean())
median_score = float(view["Happiness Score"].median())
top_country = view.loc[view["Happiness Score"].idxmax(), "Country"]
low_country = view.loc[view["Happiness Score"].idxmin(), "Country"]

m1, m2, m3, m4 = st.columns(4)
m1.metric("Countries", f"{len(view):,}")
m2.metric("Average happiness", f"{avg_score:.2f}/10")
m3.metric("Highest score", f"{view['Happiness Score'].max():.2f}", str(top_country))
m4.metric("Median score", f"{median_score:.2f}/10")

tabs = st.tabs([
    "📊 Overview & Graphs",
    "🤖 Regression Comparison",
    "🏷️ High / Low Classification",
    "🎯 Predict Happiness",
    "📋 Dataset"
])

# -------------------- Tab 1: graphs --------------------
with tabs[0]:
    st.subheader("Explore happiness patterns")
    a, b = st.columns(2)

    with a:
        fig, ax = plt.subplots(figsize=(7, 4))
        sns.histplot(view["Happiness Score"], bins=15, kde=True, ax=ax, color="#60a5fa")
        ax.set_title("Distribution of Happiness Scores")
        ax.set_xlabel("Happiness Score")
        ax.set_ylabel("Number of countries")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    with b:
        fig, ax = plt.subplots(figsize=(7, 4))
        sns.boxplot(x=view["Happiness Score"], ax=ax, color="#34d399")
        ax.set_title("Spread and Outliers")
        ax.set_xlabel("Happiness Score")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    a, b = st.columns(2)
    with a:
        top = view.nlargest(10, "Happiness Score").sort_values("Happiness Score")
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.barh(top["Country"], top["Happiness Score"], color="#60a5fa")
        ax.set_title("Top 10 Happiest Countries")
        ax.set_xlabel("Happiness Score")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    with b:
        if "Economy (GDP per Capita)" in view.columns:
            fig, ax = plt.subplots(figsize=(7, 5))
            ax.scatter(view["Economy (GDP per Capita)"], view["Happiness Score"], alpha=0.8, color="#fbbf24")
            ax.set_title("Economy vs Happiness")
            ax.set_xlabel("Economy (GDP per Capita)")
            ax.set_ylabel("Happiness Score")
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

    a, b = st.columns(2)
    with a:
        fig, ax = plt.subplots(figsize=(7, 4))
        counts = view["Region"].value_counts()
        ax.bar(counts.index, counts.values, color="#a78bfa")
        ax.set_title("Countries by Region")
        ax.set_ylabel("Number of countries")
        ax.tick_params(axis="x", rotation=35)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    with b:
        corr_cols = ["Happiness Score"] + [c for c in features if c in view.columns]
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.heatmap(view[corr_cols].corr(), annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax)
        ax.set_title("Correlation Heatmap")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    st.subheader("Country ranking table")
    st.dataframe(
        view[["Country", "Region", "Happiness Rank", "Happiness Score"]]
        .sort_values("Happiness Rank"),
        use_container_width=True,
        hide_index=True
    )

# -------------------- Tab 2: regression comparison --------------------
with tabs[1]:
    st.subheader("Regression model comparison")
    st.write("Regression predicts a numerical Happiness Score. Higher R² and lower MAE/RMSE are generally better.")
    best_reg = reg_results.iloc[0]
    c1, c2, c3 = st.columns(3)
    c1.metric("Top R² model", str(best_reg["Model"]))
    c2.metric("Best R²", f"{best_reg['R²']:.4f}")
    c3.metric("Its MAE", f"{best_reg['MAE']:.4f}")

    fig, ax = plt.subplots(figsize=(10, 4.8))
    plot_df = reg_results.sort_values("R²", ascending=True)
    ax.barh(plot_df["Model"], plot_df["R²"], color="#60a5fa")
    ax.set_title("Regression Models — R² Score")
    ax.set_xlabel("R² (higher is better)")
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    a, b = st.columns(2)
    with a:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.bar(reg_results["Model"], reg_results["MAE"], color="#fbbf24")
        ax.set_title("Mean Absolute Error (lower is better)")
        ax.set_ylabel("MAE")
        ax.tick_params(axis="x", rotation=35)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
    with b:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.bar(reg_results["Model"], reg_results["RMSE"], color="#f87171")
        ax.set_title("Root Mean Squared Error (lower is better)")
        ax.set_ylabel("RMSE")
        ax.tick_params(axis="x", rotation=35)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    st.dataframe(reg_results.round(4), use_container_width=True, hide_index=True)
    st.info("Interpretation: R² closer to 1 and smaller MAE/RMSE indicate a closer fit on this test split. Features in this report are calculated contributors to the score, so very high results may partly reflect how the report's score is constructed.")

# -------------------- Tab 3: classification --------------------
with tabs[2]:
    st.subheader("High vs Low Happiness Classification")
    st.write("The target label is created from the median Happiness Score: score ≥ median = High; score < median = Low.")
    c1, c2, c3 = st.columns(3)
    c1.metric("Median threshold", f"{threshold:.2f}")
    c2.metric("High class", f"{int((df['Happiness Score'] >= threshold).sum())} countries")
    c3.metric("Low class", f"{int((df['Happiness Score'] < threshold).sum())} countries")

    best_cls = cls_results.iloc[0]
    st.success(f"Best accuracy on this test split: {best_cls['Model']} — {best_cls['Accuracy (%)']:.1f}%")

    fig, ax = plt.subplots(figsize=(9, 4.5))
    plot_df = cls_results.sort_values("Accuracy (%)", ascending=True)
    bars = ax.barh(plot_df["Model"], plot_df["Accuracy (%)"], color="#34d399")
    ax.set_xlim(0, 100)
    ax.set_xlabel("Accuracy (%)")
    ax.set_title("Classification Model Accuracy Comparison")
    for bar, val in zip(bars, plot_df["Accuracy (%)"]):
        ax.text(val + 1, bar.get_y() + bar.get_height()/2, f"{val:.1f}%", va="center")
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.dataframe(cls_results.round(3), use_container_width=True, hide_index=True)

    chosen_model = st.selectbox("View confusion matrix for", cls_results["Model"].tolist())
    actual, predicted = cls_predictions[chosen_model]
    cm = confusion_matrix(actual, predicted, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=["Predicted Low", "Predicted High"],
                yticklabels=["Actual Low", "Actual High"], ax=ax)
    ax.set_title(f"Confusion Matrix — {chosen_model}")
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("Actual label")
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.markdown("**Metrics meaning:** Accuracy = overall correct predictions; Precision = how many predicted High cases were actually High; Recall = how many actual High cases were found; F1-score balances precision and recall.")
    st.info("These are results from one train-test split, not a guarantee for future countries or years. The model uses happiness-report contribution columns as features.")

# -------------------- Tab 4: predictor --------------------
with tabs[3]:
    st.subheader("Estimate a Happiness Score")
    st.write("Enter values for the factors available in your CSV. The prediction uses a Linear Regression model trained on an 80/20 split's training data.")
    X = df[features]
    y = df["Happiness Score"].astype(float)
    X_train_pred, X_test_pred, y_train_pred, y_test_pred = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    prediction_model = LinearRegression().fit(X_train_pred, y_train_pred)

    with st.form("predict_form"):
        inputs = {}
        left, right = st.columns(2)
        for i, feature in enumerate(features):
            default = float(df[feature].median())
            min_val = float(df[feature].min())
            max_val = float(df[feature].max())
            # Give number_input a valid range even for constant columns.
            if min_val == max_val:
                min_val -= 0.01
                max_val += 0.01
            target_col = left if i % 2 == 0 else right
            inputs[feature] = target_col.number_input(
                feature,
                min_value=min_val,
                max_value=max_val,
                value=min(max(default, min_val), max_val),
                step=max((max_val-min_val)/100, 0.01),
                format="%.4f"
            )
        submitted = st.form_submit_button("Predict Happiness Score")
    if submitted:
        input_df = pd.DataFrame([inputs], columns=features)
        score = float(prediction_model.predict(input_df)[0])
        st.metric("Estimated Happiness Score", f"{score:.2f} / 10")
        st.caption("This is a model estimate based on the selected features, not an official score or a causal conclusion.")

# -------------------- Tab 5: dataset --------------------
with tabs[4]:
    st.subheader("Dataset preview")
    st.write(f"Rows: **{len(df)}** · Columns: **{len(df.columns)}**")
    st.dataframe(df, use_container_width=True, hide_index=True)
    st.subheader("Column summary")
    st.dataframe(
        pd.DataFrame({
            "Column": df.columns,
            "Data type": [str(df[c].dtype) for c in df.columns],
            "Missing values": [int(df[c].isna().sum()) for c in df.columns]
        }),
        use_container_width=True,
        hide_index=True
    )

st.markdown("---")
st.markdown('<p class="small-note">World Happiness Insights • Educational project • Models are evaluated on the selected dataset and split.</p>', unsafe_allow_html=True)
