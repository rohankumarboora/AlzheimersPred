# app.py
# Streamlit Dashboard: Alzheimer’s (OASIS Longitudinal) – EDA, Training, and Single-Case Prediction
# Run: streamlit run app.py

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import altair as alt
import streamlit as st

from typing import Tuple, Dict, List
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, confusion_matrix, classification_report,
    RocCurveDisplay
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

st.set_page_config(page_title="Alzheimer’s ML Dashboard", layout="wide", page_icon="🧠")

# ---------------------- Utilities ----------------------
DEFAULT_PATH = "oasis_longitudinal_expanded.csv"  # fallback to your expanded CSV

PRIORITY_LABELS = ["group", "target", "label", "class", "diagnosis", "dx", "status", "outcome", "cdr", "dementia", "y"]
ID_LIKE_HINTS = ["id", "subject", "mri", "visit"]

def pick_label_column(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lowermap = {c: c.lower() for c in cols}
    # Priority names
    for c in cols:
        if lowermap[c] in PRIORITY_LABELS and df[c].nunique(dropna=True) > 1:
            return c
    # Low-cardinality categorical-ish
    candidates = []
    for c in cols:
        nunq = df[c].nunique(dropna=True)
        if 1 < nunq <= 10 and nunq < len(df) * 0.5:
            candidates.append((nunq, c))
    if candidates:
        candidates.sort(key=lambda x: x[0])
        return candidates[0][1]
    # Fallback: last col
    return cols[-1]

def suggest_drop_cols(df: pd.DataFrame) -> List[str]:
    # Drop obviously ID-like and nearly-unique columns
    drops = []
    n = len(df)
    for c in df.columns:
        lc = c.lower()
        if any(h in lc for h in ID_LIKE_HINTS):
            drops.append(c)
        elif df[c].nunique(dropna=True) > 0.9 * n:
            drops.append(c)
    return sorted(list(set(drops)))

@st.cache_data(show_spinner=False)
def load_data(uploaded_file) -> pd.DataFrame:
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
    else:
        if os.path.exists(DEFAULT_PATH):
            df = pd.read_csv(DEFAULT_PATH)
        else:
            st.stop()
    return df

def split_features_target(
    df: pd.DataFrame,
    target_col: str,
    drop_cols: List[str]
) -> Tuple[pd.DataFrame, pd.Series]:
    cols = [c for c in df.columns if c != target_col and c not in drop_cols]
    X = df[cols].copy()
    y = df[target_col].copy()
    return X, y

def type_columns(X: pd.DataFrame) -> Tuple[List[str], List[str]]:
    cat_cols = [c for c in X.columns if X[c].dtype == "object" or str(X[c].dtype) == "category" or X[c].dtype == "bool"]
    num_cols = [c for c in X.columns if c not in cat_cols]
    return num_cols, cat_cols

def make_pipeline(num_cols: List[str], cat_cols: List[str], model_name: str, params: Dict):
    numeric = Pipeline(steps=[("scaler", StandardScaler())])
    categorical = Pipeline(steps=[("onehot", OneHotEncoder(handle_unknown="ignore"))])
    pre = ColumnTransformer(
        transformers=[
            ("num", numeric, num_cols),
            ("cat", categorical, cat_cols),
        ]
    )

    if model_name == "Logistic Regression":
        clf = LogisticRegression(max_iter=params.get("max_iter", 200), C=params.get("C", 1.0), n_jobs=None, multi_class="auto")
    elif model_name == "Random Forest":
        clf = RandomForestClassifier(
            n_estimators=params.get("n_estimators", 300),
            max_depth=params.get("max_depth", None),
            random_state=42
        )
    elif model_name == "Gradient Boosting":
        clf = GradientBoostingClassifier(
            n_estimators=params.get("n_estimators", 300),
            learning_rate=params.get("learning_rate", 0.05),
            max_depth=params.get("max_depth", 3),
            random_state=42
        )
    else:
        raise ValueError("Unknown model")
    pipe = Pipeline(steps=[("pre", pre), ("clf", clf)])
    return pipe

def safe_roc_auc(model, X_val, y_val):
    # Works for binary or multi-class
    try:
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X_val)
            # If multiclass, use 'ovr' macro average
            if proba.shape[1] > 2:
                return roc_auc_score(y_val, proba, multi_class="ovr")
            else:
                return roc_auc_score(y_val, proba[:, 1])
        else:
            return np.nan
    except Exception:
        return np.nan

@st.cache_resource(show_spinner=False)
def train_model(X, y, model_name, params, test_size, random_state):
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    num_cols, cat_cols = type_columns(X_train)
    pipe = make_pipeline(num_cols, cat_cols, model_name, params)
    pipe.fit(X_train, y_train)
    yhat = pipe.predict(X_val)
    metrics = {
        "accuracy": accuracy_score(y_val, yhat),
        "f1_macro": f1_score(y_val, yhat, average="macro"),
        "roc_auc": safe_roc_auc(pipe, X_val, y_val),
        "report": classification_report(y_val, yhat, digits=3)
    }
    cm = confusion_matrix(y_val, yhat, labels=np.unique(y))
    labels = list(np.unique(y))
    return pipe, metrics, cm, labels, (X_train, X_val, y_train, y_val)

# ====== UPDATED: inline range (close to label) + Age as whole number ======
def auto_build_single_input(df: pd.DataFrame, X_cols: List[str]) -> Dict[str, any]:
    """
    Build UI inputs for a single-case form using training column types & ranges.
    - Shows allowed range inline RIGHT NEXT to label.
    - 'Age' is forced to whole numbers (step=1, integer bounds).
    - Other numeric fields remain float inputs with bounds.
    """
    st.subheader("Enter values")
    values = {}
    for c in X_cols:
        series = df[c]

        # -------- Categorical ----------
        if series.dtype == "object" or str(series.dtype) == "category" or series.dtype == "bool":
            opts = sorted([o for o in series.dropna().unique().tolist()])
            values[c] = st.selectbox(f"{c}", options=opts if len(opts) > 0 else [""])
            continue

        # -------- Numeric ----------
        s = pd.to_numeric(series, errors="coerce").dropna()
        if len(s) == 0:
            # Fallback when no stats are available
            st.markdown(f"**{c}**")
            values[c] = st.number_input("", value=0.0, key=f"{c}_fallback")
            continue

        mn, mx = float(s.min()), float(s.max())
        mean_val = float(s.mean())
        if mn == mx:  # guard degenerate ranges
            mn, mx = mn - 1.0, mx + 1.0

        is_age = c.strip().lower() == "age"

        if is_age:
            # Whole-number Age with integer bounds
            mn_i, mx_i = int(np.floor(mn)), int(np.ceil(mx))
            default_i = int(np.clip(int(round(mean_val)), mn_i, mx_i))
            # Inline label + whole-number range right next to it
            st.markdown(
                f"**{c}** <span style='color:gray'>(Allowed: [{mn_i}, {mx_i}])</span>",
                unsafe_allow_html=True
            )
            values[c] = st.number_input(
                "",
                min_value=mn_i,
                max_value=mx_i,
                value=default_i,
                step=1,
                key=f"{c}_int",
            )
        else:
            # Float bounds; inline range right next to label
            st.markdown(
                f"**{c}** <span style='color:gray'>(Allowed: [{mn:.2f}, {mx:.2f}])</span>",
                unsafe_allow_html=True
            )
            step = (mx - mn) / 200.0 if mx > mn else 0.01
            default_f = float(np.clip(mean_val, mn, mx))
            values[c] = st.number_input(
                "",
                min_value=float(mn),
                max_value=float(mx),
                value=default_f,
                step=float(step),
                key=f"{c}_float",
            )

    return values
# ====== /UPDATED ======

# ---------------------- Sidebar / Navigation ----------------------
st.sidebar.title("🧠 Alzheimer’s ML Dashboard")
uploaded = st.sidebar.file_uploader("Upload CSV (optional)", type=["csv"])
df = load_data(uploaded)

st.sidebar.markdown("---")
auto_label = pick_label_column(df)
label_col = st.sidebar.selectbox("Target column", options=list(df.columns), index=list(df.columns).index(auto_label) if auto_label in df.columns else 0)
st.sidebar.caption(f"Auto-detected: **{auto_label}**")
drop_suggest = suggest_drop_cols(df)
drop_cols = st.sidebar.multiselect("Columns to drop (IDs, nearly-unique)", options=list(df.columns), default=[c for c in drop_suggest if c != label_col])

st.sidebar.markdown("---")
page = st.sidebar.radio("Navigate", ["Overview", "Explore (EDA)", "Train & Evaluate", "Single-Case Prediction"])

# Derive X/y
if label_col not in df.columns:
    st.error("Selected target column not in dataframe.")
    st.stop()
X, y = split_features_target(df, label_col, drop_cols)

# ---------------------- Pages ----------------------
if page == "Overview":
    st.title("Overview")
    c1, c2 = st.columns([2, 1], gap="large")

    with c1:
        st.markdown("### Dataset Snapshot")
        st.dataframe(df.head(20), use_container_width=True)
        st.markdown(f"- **Rows:** {len(df)}  \n- **Columns:** {len(df.columns)}")
        st.markdown(f"- **Target:** `{label_col}`  \n- **Dropped:** {', '.join(drop_cols) if drop_cols else 'None'}")

    with c2:
        st.markdown("### Class Distribution")
        vc = y.value_counts(dropna=False)
        st.write(vc)
        cd = pd.DataFrame({"class": vc.index.astype(str), "count": vc.values})
        chart = (
            alt.Chart(cd)
            .mark_bar()
            .encode(x=alt.X("class:N", sort="-y", title="Class"),
                    y=alt.Y("count:Q", title="Count"),
                    tooltip=["class", "count"])
            .properties(height=300)
        )
        st.altair_chart(chart, use_container_width=True)

    st.markdown("### Column Types")
    num_cols, cat_cols = type_columns(X)
    col1, col2 = st.columns(2)
    with col1:
        st.write("**Numeric:**", num_cols if num_cols else "—")
    with col2:
        st.write("**Categorical:**", cat_cols if cat_cols else "—")

elif page == "Explore (EDA)":
    st.title("Explore (EDA)")
    st.markdown("Select columns to visualize distributions and relationships.")

    col = st.selectbox("Choose a column", options=list(X.columns))
    if col:
        if X[col].dtype == "object" or str(X[col].dtype) == "category" or X[col].dtype == "bool":
            st.markdown(f"#### Categorical: {col}")
            vc = X[col].value_counts(dropna=False)
            cat_df = pd.DataFrame({col: vc.index.astype(str), "count": vc.values})
            chart = (
                alt.Chart(cat_df)
                .mark_bar()
                .encode(x=alt.X(f"{col}:N", sort="-y"), y="count:Q", tooltip=[col, "count"])
                .properties(height=350)
            )
            st.altair_chart(chart, use_container_width=True)
            # Stacked vs target
            st.markdown("#### Cross by Target")
            cross = pd.crosstab(X[col].astype(str), y)
            st.dataframe(cross, use_container_width=True)
        else:
            st.markdown(f"#### Numeric: {col}")
            c1, c2 = st.columns(2)
            with c1:
                hist = alt.Chart(pd.DataFrame({col: X[col]})).mark_bar().encode(
                    alt.X(f"{col}:Q", bin=alt.Bin(maxbins=30)),
                    y="count()"
                ).properties(height=350)
                st.altair_chart(hist, use_container_width=True)
            with c2:
                box = alt.Chart(pd.DataFrame({col: X[col], "target": y})).mark_boxplot().encode(
                    x="target:N", y=f"{col}:Q"
                ).properties(height=350)
                st.altair_chart(box, use_container_width=True)

    st.markdown("---")
    st.markdown("#### Correlation (numeric only)")
    num_cols, _ = type_columns(X)
    if len(num_cols) >= 2:
        corr = X[num_cols].corr(numeric_only=True)
        st.dataframe(corr.style.background_gradient(cmap="Blues"), use_container_width=True)
    else:
        st.info("Not enough numeric columns for correlation heatmap.")

elif page == "Train & Evaluate":
    st.title("Train & Evaluate")
    st.markdown("Tune settings, train a model, and review performance.")

    with st.form("train_form"):
        c1, c2, c3 = st.columns(3)
        with c1:
            model_name = st.selectbox("Model", ["Logistic Regression", "Random Forest", "Gradient Boosting"])
            test_size = st.slider("Test size", 0.1, 0.4, 0.2, 0.01)
        with c2:
            random_state = st.number_input("Random state", min_value=0, value=42, step=1)
        with c3:
            st.caption("Hyperparameters")
            params = {}
            if model_name == "Logistic Regression":
                params["C"] = st.number_input("C (inverse regularization)", min_value=0.001, value=1.0, step=0.1)
                params["max_iter"] = st.number_input("max_iter", min_value=50, value=200, step=50)
            elif model_name == "Random Forest":
                params["n_estimators"] = st.number_input("n_estimators", min_value=50, value=300, step=50)
                max_depth = st.text_input("max_depth (blank = None)", value="")
                params["max_depth"] = None if max_depth.strip() == "" else int(max_depth)
            elif model_name == "Gradient Boosting":
                params["n_estimators"] = st.number_input("n_estimators", min_value=50, value=300, step=50)
                params["learning_rate"] = st.number_input("learning_rate", min_value=0.001, value=0.05, step=0.01)
                params["max_depth"] = st.number_input("max_depth (trees)", min_value=1, value=3, step=1)

        submitted = st.form_submit_button("Train")
    if submitted:
        with st.spinner("Training..."):
            model, metrics, cm, labels, splits = train_model(X, y, model_name, params, test_size, random_state)

        st.success("Training complete.")
        m1, m2, m3 = st.columns(3)
        m1.metric("Accuracy", f"{metrics['accuracy']:.3f}")
        m2.metric("F1 (macro)", f"{metrics['f1_macro']:.3f}")
        m3.metric("ROC AUC", f"{metrics['roc_auc']:.3f}" if not np.isnan(metrics['roc_auc']) else "—")

        st.markdown("#### Confusion Matrix")
        fig, ax = plt.subplots()
        im = ax.imshow(cm, cmap="Blues")
        ax.set_xticks(range(len(labels))); ax.set_yticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha="right"); ax.set_yticklabels(labels)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, cm[i, j], ha="center", va="center", color="black")
        ax.set_xlabel("Predicted"); ax.set_ylabel("True")
        st.pyplot(fig, clear_figure=True)

        st.markdown("#### Classification Report")
        st.code(metrics["report"])

        # ROC curve (only if binary or we can show OvR curves)
        from sklearn.preprocessing import label_binarize
        X_train, X_val, y_train, y_val = splits
        if hasattr(model, "predict_proba"):
            try:
                proba = model.predict_proba(X_val)
                fig2, ax2 = plt.subplots()
                if proba.shape[1] == 2:
                    RocCurveDisplay.from_predictions(y_val, proba[:, 1], name="ROC", ax=ax2)
                else:
                    classes = model.classes_
                    y_bin = label_binarize(y_val, classes=classes)
                    for i, cls in enumerate(classes):
                        RocCurveDisplay.from_predictions(y_bin[:, i], proba[:, i], name=f"ROC: {cls}", ax=ax2)
                st.pyplot(fig2, clear_figure=True)
            except Exception:
                st.info("Could not render ROC curves for this configuration.")

        # Store model in session for prediction page
        st.session_state["trained_model"] = model
        st.session_state["feature_columns"] = list(X.columns)
        st.session_state["train_df"] = pd.concat([X, y.rename(label_col)], axis=1)

elif page == "Single-Case Prediction":
    st.title("Single-Case Prediction")
    # Require a trained model
    if "trained_model" not in st.session_state or "feature_columns" not in st.session_state:
        st.warning("Please train a model first on the **Train & Evaluate** page.")
    else:
        model = st.session_state["trained_model"]
        feat_cols = st.session_state["feature_columns"]
        train_df = st.session_state.get("train_df", pd.concat([X, y.rename(label_col)], axis=1))

        with st.form("single_case"):
            inputs = auto_build_single_input(train_df, feat_cols)
            predict_btn = st.form_submit_button("Predict")
        if predict_btn:
            x_row = pd.DataFrame([inputs])[feat_cols]
            pred = model.predict(x_row)[0]
            st.markdown(
                f"""
                <div style="padding:16px; border-radius:12px; background:#F3F8FF; border:1px solid #D6E4FF;">
                    <div style="font-size:22px; color:#1F3B7B; font-weight:700; margin-bottom:6px;">
                        Predicted class
                    </div>
                    <div style="font-size:34px; font-weight:800; color:#0B63F6;">
                        {pred}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(x_row).flatten()
                prob_df = pd.DataFrame({"class": list(model.classes_), "probability": proba})
                prob_df["probability"] = prob_df["probability"].round(4)
                st.markdown("#### Class Probabilities")
                chart = (
                    alt.Chart(prob_df.sort_values("probability", ascending=False))
                    .mark_bar()
                    .encode(
                        x=alt.X("probability:Q", title="Probability", scale=alt.Scale(domain=[0, 1])),
                        y=alt.Y("class:N", sort="-x", title="Class"),
                        tooltip=["class", alt.Tooltip("probability:Q", format=".4f")],
                    )
                    .properties(height=300)
                )
                st.altair_chart(chart, use_container_width=True)
                st.dataframe(prob_df, use_container_width=True)
            else:
                st.info("This model does not provide predict_proba; probabilities not available.")
