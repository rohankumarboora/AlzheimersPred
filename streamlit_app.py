# app.py
# Streamlit Dashboard: Alzheimer’s (OASIS Longitudinal) – EDA, Training, and Single-Case Prediction
# Run: streamlit run app.py

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import altair as alt
import streamlit as st

from typing import Tuple, Dict, List, Optional, Tuple as Tup
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, label_binarize
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, confusion_matrix, classification_report,
    RocCurveDisplay
)
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    AdaBoostClassifier,
    ExtraTreesClassifier,
    VotingClassifier
)

st.set_page_config(page_title="Alzheimer’s ML Dashboard", layout="wide", page_icon="🧠")

# ---------------------- Utilities ----------------------
DEFAULT_PATH = "oasis_longitudinal_expanded.csv"  # fallback to your expanded CSV

PRIORITY_LABELS = ["group", "target", "label", "class", "diagnosis", "dx", "status", "outcome", "cdr", "dementia", "y"]
ID_LIKE_HINTS = ["id", "subject", "mri", "visit"]

def pick_label_column(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lowermap = {c: c.lower() for c in cols}
    for c in cols:
        if lowermap[c] in PRIORITY_LABELS and df[c].nunique(dropna=True) > 1:
            return c
    candidates = []
    for c in cols:
        nunq = df[c].nunique(dropna=True)
        if 1 < nunq <= 10 and nunq < len(df) * 0.5:
            candidates.append((nunq, c))
    if candidates:
        candidates.sort(key=lambda x: x[0])
        return candidates[0][1]
    return cols[-1]

def suggest_drop_cols(df: pd.DataFrame) -> List[str]:
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

def _preprocessor(num_cols, cat_cols):
    numeric = Pipeline(steps=[("scaler", StandardScaler())])
    categorical = Pipeline(steps=[("onehot", OneHotEncoder(handle_unknown="ignore"))])
    return ColumnTransformer(
        transformers=[("num", numeric, num_cols), ("cat", categorical, cat_cols)]
    )

def build_soft_voter(weights: Optional[List[float]] = None) -> VotingClassifier:
    rf = RandomForestClassifier(n_estimators=300, random_state=42)
    gb = GradientBoostingClassifier(n_estimators=300, learning_rate=0.05, max_depth=3, random_state=42)
    ab = AdaBoostClassifier(n_estimators=200, learning_rate=0.5, random_state=42)
    et = ExtraTreesClassifier(n_estimators=400, random_state=42)
    voter = VotingClassifier(
        estimators=[("rf", rf), ("gb", gb), ("ab", ab), ("et", et)],
        voting="soft",
        weights=weights
    )
    return voter

def make_pipeline(num_cols: List[str], cat_cols: List[str], model_name: str, params: Dict, voter_weights: Optional[List[float]] = None):
    pre = _preprocessor(num_cols, cat_cols)

    if model_name == "Random Forest":
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
    elif model_name == "AdaBoost":
        clf = AdaBoostClassifier(
            n_estimators=params.get("n_estimators", 200),
            learning_rate=params.get("learning_rate", 0.5),
            random_state=42
        )
    elif model_name == "Extra Trees":
        clf = ExtraTreesClassifier(
            n_estimators=params.get("n_estimators", 400),
            max_depth=params.get("max_depth", None),
            random_state=42
        )
    elif model_name == "Soft Voting (RF+GB+Ada+ET)":
        clf = build_soft_voter(weights=voter_weights)
    else:
        raise ValueError("Unknown model")

    return Pipeline(steps=[("pre", pre), ("clf", clf)])

def make_soft_voter_pipeline(num_cols: List[str], cat_cols: List[str], weights: Optional[List[float]] = None) -> Pipeline:
    """Used on the Single-Case Prediction page; respects stored weights."""
    pre = _preprocessor(num_cols, cat_cols)
    voter = build_soft_voter(weights=weights)
    return Pipeline(steps=[("pre", pre), ("voter", voter)])

def safe_roc_auc(model, X_val, y_val):
    try:
        # route to the last step if pipeline
        last = model
        if hasattr(model, "named_steps"):
            if "clf" in model.named_steps:
                last = model.named_steps["clf"]
            elif "voter" in model.named_steps:
                last = model.named_steps["voter"]

        if not hasattr(last, "predict_proba"):
            return np.nan
        proba = model.predict_proba(X_val)
        if proba.ndim == 2 and proba.shape[1] == 2:
            return roc_auc_score(y_val, proba[:, 1])
        if proba.ndim == 2 and proba.shape[1] > 2:
            return roc_auc_score(y_val, proba, multi_class="ovr")
        return np.nan
    except Exception:
        return np.nan

def soft_components_val_probs(pipe: Pipeline, X_val: pd.DataFrame) -> Optional[Tup[Dict[str, np.ndarray], np.ndarray]]:
    """If pipe is soft voter, return dict of per-model probs on X_val (preprocessed) and classes."""
    if not hasattr(pipe, "named_steps") or "clf" not in pipe.named_steps:
        return None
    voter = pipe.named_steps["clf"]
    pre = pipe.named_steps["pre"]
    if not isinstance(voter, VotingClassifier):
        return None
    # transformed features
    Xv = pre.transform(X_val)
    names = [name for name, _ in voter.estimators]
    fitted = voter.estimators_  # fitted base estimators
    probs = {}
    for name, est in zip(names, fitted):
        if hasattr(est, "predict_proba"):
            probs[name] = est.predict_proba(Xv)
    classes = voter.classes_
    return probs, classes

@st.cache_resource(show_spinner=False)
def train_model(X, y, model_name, params, test_size, random_state, voter_weights=None):
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    num_cols, cat_cols = type_columns(X_train)
    pipe = make_pipeline(num_cols, cat_cols, model_name, params, voter_weights)
    pipe.fit(X_train, y_train)
    yhat = pipe.predict(X_val)
    metrics = {
        "accuracy": accuracy_score(y_val, yhat),
        "f1_macro": f1_score(y_val, yhat, average="macro"),
        "roc_auc": safe_roc_auc(pipe, X_val, y_val),
        "report": classification_report(y_val, yhat, digits=3, zero_division=0)
    }
    cm = confusion_matrix(y_val, yhat, labels=np.unique(y))
    labels = list(np.unique(y))
    return pipe, metrics, cm, labels, (X_train, X_val, y_train, y_val)

# ---------------------- Sidebar / Navigation ----------------------
st.sidebar.title("🧠 Alzheimer’s ML Dashboard")
uploaded = st.sidebar.file_uploader("Upload CSV (optional)", type=["csv"])
df = load_data(uploaded)

st.sidebar.markdown("---")
auto_label = pick_label_column(df)
label_col = st.sidebar.selectbox(
    "Target column",
    options=list(df.columns),
    index=list(df.columns).index(auto_label) if auto_label in df.columns else 0
)
st.sidebar.caption(f"Auto-detected: **{auto_label}**")
drop_suggest = suggest_drop_cols(df)
drop_cols = st.sidebar.multiselect(
    "Columns to drop (IDs, nearly-unique)",
    options=list(df.columns),
    default=[c for c in drop_suggest if c != label_col]
)

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
            .encode(
                x=alt.X("class:N", sort="-y", title="Class"),
                y=alt.Y("count:Q", title="Count"),
                tooltip=["class", "count"]
            )
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
            model_name = st.selectbox(
                "Model",
                ["Random Forest", "Gradient Boosting", "AdaBoost", "Extra Trees", "Soft Voting (RF+GB+Ada+ET)"]
            )
            test_size = st.slider("Test size", 0.1, 0.4, 0.2, 0.01)
            train_all = st.checkbox("Train all models (compare, includes Soft Voting)")
        with c2:
            random_state = st.number_input("Random state", min_value=0, value=42, step=1)
        with c3:
            st.caption("Hyperparameters (for selected model)")
            params = {}
            voter_weights = None
            show_per_model = False

            if model_name == "Random Forest":
                params["n_estimators"] = st.number_input("n_estimators", min_value=50, value=300, step=50)
                max_depth = st.text_input("max_depth (blank = None)", value="")
                params["max_depth"] = None if max_depth.strip() == "" else int(max_depth)

            elif model_name == "Gradient Boosting":
                params["n_estimators"] = st.number_input("n_estimators", min_value=50, value=300, step=50)
                params["learning_rate"] = st.number_input("learning_rate", min_value=0.001, value=0.05, step=0.01)
                params["max_depth"] = st.number_input("max_depth (trees)", min_value=1, value=3, step=1)

            elif model_name == "AdaBoost":
                params["n_estimators"] = st.number_input("n_estimators", min_value=50, value=200, step=50)
                params["learning_rate"] = st.number_input("learning_rate", min_value=0.01, value=0.5, step=0.01)

            elif model_name == "Extra Trees":
                params["n_estimators"] = st.number_input("n_estimators", min_value=50, value=400, step=50)
                max_depth = st.text_input("max_depth (blank = None)", value="")
                params["max_depth"] = None if max_depth.strip() == "" else int(max_depth)

            elif model_name == "Soft Voting (RF+GB+Ada+ET)":
                st.caption("Soft Voting uses RF, GB, Ada, ET with configurable weights.")
                c_w1, c_w2, c_w3, c_w4 = st.columns(4)
                w_rf = c_w1.number_input("RF weight", min_value=0.0, value=float(st.session_state.get("sv_w_rf", 1.0)), step=0.1)
                w_gb = c_w2.number_input("GB weight", min_value=0.0, value=float(st.session_state.get("sv_w_gb", 1.0)), step=0.1)
                w_ab = c_w3.number_input("Ada weight", min_value=0.0, value=float(st.session_state.get("sv_w_ab", 1.0)), step=0.1)
                w_et = c_w4.number_input("ET weight", min_value=0.0, value=float(st.session_state.get("sv_w_et", 1.0)), step=0.1)
                voter_weights = [w_rf, w_gb, w_ab, w_et]
                show_per_model = st.checkbox("Show per-model validation probabilities & AUCs")

        submitted = st.form_submit_button("Train")

    if submitted and not train_all:
        # ---------------- Single model flow ----------------
        with st.spinner("Training..."):
            model, metrics, cm, labels, splits = train_model(X, y, model_name, params, test_size, random_state, voter_weights=voter_weights)

        st.success("Training complete.")
        m1, m2, m3 = st.columns(3)
        m1.metric("Accuracy", f"{metrics['accuracy']:.3f}")
        m2.metric("F1 (macro)", f"{metrics['f1_macro']:.3f}")
        m3.metric("ROC AUC", f"{metrics['roc_auc']:.3f}" if not np.isnan(metrics['roc_auc']) else "—")

        st.markdown("#### Classification Report")
        st.code(metrics["report"])

        st.markdown("#### Confusion Matrix")
        fig, ax = plt.subplots()
        ax.imshow(cm, cmap="Blues")
        ax.set_xticks(range(len(labels))); ax.set_yticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha="right"); ax.set_yticklabels(labels)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, cm[i, j], ha="center", va="center", color="black")
        ax.set_xlabel("Predicted"); ax.set_ylabel("True")
        st.pyplot(fig, clear_figure=True)

        # ROC curves
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
            except Exception as e:
                st.info(f"Could not render ROC curves: {e}")

        # If soft voting and requested, show per-model validation probabilities & AUCs
        if model_name == "Soft Voting (RF+GB+Ada+ET)" and show_per_model:
            out = soft_components_val_probs(model, X_val)
            if out is not None:
                comp_probs, classes = out
                # AUC per component
                auc_rows = []
                if len(classes) == 2:
                    # binary
                    for name, p in comp_probs.items():
                        try:
                            auc_val = roc_auc_score(y_val, p[:, 1])
                        except Exception:
                            auc_val = np.nan
                        auc_rows.append({"Estimator": name, "Val ROC AUC": auc_val})
                else:
                    # multiclass OvR
                    for name, p in comp_probs.items():
                        try:
                            auc_val = roc_auc_score(y_val, p, multi_class="ovr")
                        except Exception:
                            auc_val = np.nan
                        auc_rows.append({"Estimator": name, "Val ROC AUC": auc_val})
                st.markdown("#### Soft Voting — Per-Model Validation AUC")
                st.dataframe(pd.DataFrame(auc_rows), use_container_width=True)

                # Show first 10 rows of per-model probabilities (or class-wise means if multiclass)
                if len(classes) == 2:
                    show_n = min(10, len(y_val))
                    df_list = []
                    for name, p in comp_probs.items():
                        df_list.append(pd.DataFrame({
                            "estimator": name,
                            "y_val": list(y_val)[:show_n],
                            f"P({classes[1]})": p[:show_n, 1]
                        }))
                    st.markdown("#### Per-Model Validation Probabilities (first 10 rows)")
                    st.dataframe(pd.concat(df_list, ignore_index=True), use_container_width=True)
                else:
                    mean_rows = []
                    for name, p in comp_probs.items():
                        row = {"estimator": name}
                        for i, cls in enumerate(classes):
                            row[f"mean_P({cls})"] = float(np.mean(p[:, i]))
                        mean_rows.append(row)
                    st.markdown("#### Per-Model Mean Validation Probabilities (multiclass)")
                    st.dataframe(pd.DataFrame(mean_rows), use_container_width=True)
            else:
                st.info("Per-model probabilities unavailable (not a soft voting pipeline).")

        # Save (and remember voter weights if any)
        st.session_state["trained_model"] = model
        st.session_state["feature_columns"] = list(X.columns)
        st.session_state["train_df"] = pd.concat([X, y.rename(label_col)], axis=1)
        st.session_state["label_col"] = label_col
        if voter_weights is not None:
            st.session_state["sv_w_rf"], st.session_state["sv_w_gb"], st.session_state["sv_w_ab"], st.session_state["sv_w_et"] = voter_weights

    if submitted and train_all:
        # ---------------- Train ALL models and compare (including Soft Voting) ----------------
        with st.spinner("Training all models..."):
            default_weights = [
                float(st.session_state.get("sv_w_rf", 1.0)),
                float(st.session_state.get("sv_w_gb", 1.0)),
                float(st.session_state.get("sv_w_ab", 1.0)),
                float(st.session_state.get("sv_w_et", 1.0)),
            ]
            all_cfgs = [
                ("Random Forest", {"n_estimators": 300, "max_depth": None}, None),
                ("Gradient Boosting", {"n_estimators": 300, "learning_rate": 0.05, "max_depth": 3}, None),
                ("AdaBoost", {"n_estimators": 200, "learning_rate": 0.5}, None),
                ("Extra Trees", {"n_estimators": 400, "max_depth": None}, None),
                ("Soft Voting (RF+GB+Ada+ET)", {}, default_weights),
            ]

            results = []
            for name, p, w in all_cfgs:
                mdl, mtr, cm, lbls, splits = train_model(X, y, name, p, test_size, random_state, voter_weights=w)
                results.append({
                    "Model": name,
                    "Accuracy": mtr["accuracy"],
                    "F1 (macro)": mtr["f1_macro"],
                    "ROC AUC": mtr["roc_auc"],
                    "Report": mtr["report"],
                    "CM": cm,
                    "Labels": lbls,
                    "Splits": splits,
                    "Estimator": mdl,
                })

        st.success("Training complete (all models).")
        summary_df = pd.DataFrame(
            [{k: v for k, v in r.items() if k in ["Model", "Accuracy", "F1 (macro)", "ROC AUC"]} for r in results]
        ).sort_values("Accuracy", ascending=False)
        summary_df[["Accuracy", "F1 (macro)", "ROC AUC"]] = summary_df[["Accuracy", "F1 (macro)", "ROC AUC"]].applymap(
            lambda x: np.nan if pd.isna(x) else float(x)
        )
        st.markdown("### Metrics Summary")
        st.dataframe(summary_df.reset_index(drop=True), use_container_width=True)

        tabs = st.tabs([r["Model"] for r in results])
        for tab, r in zip(tabs, results):
            with tab:
                st.markdown(f"#### {r['Model']} — Classification Report")
                st.code(r["Report"])

                st.markdown("#### Confusion Matrix")
                fig, ax = plt.subplots()
                ax.imshow(r["CM"], cmap="Blues")
                ax.set_xticks(range(len(r["Labels"]))); ax.set_yticks(range(len(r["Labels"])))
                ax.set_xticklabels(r["Labels"], rotation=45, ha="right"); ax.set_yticklabels(r["Labels"])
                for i in range(r["CM"].shape[0]):
                    for j in range(r["CM"].shape[1]):
                        ax.text(j, i, r["CM"][i, j], ha="center", va="center", color="black")
                ax.set_xlabel("Predicted"); ax.set_ylabel("True")
                st.pyplot(fig, clear_figure=True)

                # ROC curves
                model = r["Estimator"]
                X_train, X_val, y_train, y_val = r["Splits"]
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
                    except Exception as e:
                        st.info(f"Could not render ROC curves: {e}")

        # Keep best by accuracy for convenience
        best_row = max(results, key=lambda r: r["Accuracy"])
        st.session_state["trained_model"] = best_row["Estimator"]
        st.session_state["feature_columns"] = list(X.columns)
        st.session_state["train_df"] = pd.concat([X, y.rename(label_col)], axis=1)
        st.session_state["label_col"] = label_col
        st.info(f"Best model by Accuracy: **{best_row['Model']}**")

elif page == "Single-Case Prediction":
    st.title("Single-Case Prediction")
    if "feature_columns" not in st.session_state or "train_df" not in st.session_state or "label_col" not in st.session_state:
        st.warning("Please train at least once on the **Train & Evaluate** page.")
    else:
        feat_cols = st.session_state["feature_columns"]
        train_df_full = st.session_state["train_df"].copy()
        tgt_col = st.session_state["label_col"]

        # Pull last-used soft-voter weights (defaults 1.0)
        sv_w_rf = float(st.session_state.get("sv_w_rf", 1.0))
        sv_w_gb = float(st.session_state.get("sv_w_gb", 1.0))
        sv_w_ab = float(st.session_state.get("sv_w_ab", 1.0))
        sv_w_et = float(st.session_state.get("sv_w_et", 1.0))

        with st.form("single_case"):
            st.subheader("Enter values (Soft Voting across RF/GB/Ada/ExtraTrees)")
            c_w1, c_w2, c_w3, c_w4 = st.columns(4)
            sv_w_rf = c_w1.number_input("RF weight", min_value=0.0, value=sv_w_rf, step=0.1)
            sv_w_gb = c_w2.number_input("GB weight", min_value=0.0, value=sv_w_gb, step=0.1)
            sv_w_ab = c_w3.number_input("Ada weight", min_value=0.0, value=sv_w_ab, step=0.1)
            sv_w_et = c_w4.number_input("ET weight", min_value=0.0, value=sv_w_et, step=0.1)

            values = {}
            for c in feat_cols:
                series = train_df_full[c]
                if series.dtype == "object" or str(series.dtype) == "category" or series.dtype == "bool":
                    opts = sorted([o for o in series.dropna().unique().tolist()])
                    values[c] = st.selectbox(c, options=opts if len(opts) > 0 else [""], key=f"{c}_sb")
                    continue

                s = pd.to_numeric(series, errors="coerce").dropna()
                if len(s) == 0:
                    st.markdown(f"**{c}**")
                    values[c] = st.number_input("", value=0.0, key=f"{c}_ni_fallback")
                    continue

                mn, mx = float(s.min()), float(s.max())
                mn_i, mx_i = int(np.floor(mn)), int(np.ceil(mx))

                if c.strip().lower() == "age":
                    default_i = int(np.clip(int(round(float(s.mean()))), mn_i, mx_i))
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
                    default_f = float(np.clip(float(s.mean()), mn, mx))
                    step = (mx - mn) / 100.0 if mx > mn else 0.01
                    st.markdown(
                        f"**{c}** <span style='color:gray'>(Allowed: [{mn_i}, {mx_i}])</span>",
                        unsafe_allow_html=True
                    )
                    values[c] = st.number_input(
                        "",
                        min_value=float(mn),
                        max_value=float(mx),
                        value=default_f,
                        step=float(step),
                        key=f"{c}_float",
                    )
            predict_btn = st.form_submit_button("Predict")

        if predict_btn:
            X_all = train_df_full[feat_cols]
            y_all = train_df_full[tgt_col]
            num_cols, cat_cols = type_columns(X_all)
            soft_voter = make_soft_voter_pipeline(num_cols, cat_cols, weights=[sv_w_rf, sv_w_gb, sv_w_ab, sv_w_et])
            soft_voter.fit(X_all, y_all)

            x_row = pd.DataFrame([values])[feat_cols]
            pred = soft_voter.predict(x_row)[0]
            st.markdown(
                f"""
                <div style="padding:16px; border-radius:12px; background:#F3F8FF; border:1px solid #D6E4FF;">
                    <div style="font-size:22px; color:#1F3B7B; font-weight:700; margin-bottom:6px;">
                        Predicted class (Soft Voting)
                    </div>
                    <div style="font-size:34px; font-weight:800; color:#0B63F6;">
                        {pred}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            if hasattr(soft_voter, "predict_proba"):
                proba = soft_voter.predict_proba(x_row).flatten()
                prob_df = pd.DataFrame({"class": list(soft_voter.named_steps["voter"].classes_), "probability": proba})
                prob_df["probability"] = prob_df["probability"].round(4)
                st.markdown("#### Class Probabilities (Soft Voting)")
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

            # remember latest weights
            st.session_state["sv_w_rf"], st.session_state["sv_w_gb"], st.session_state["sv_w_ab"], st.session_state["sv_w_et"] = sv_w_rf, sv_w_gb, sv_w_ab, sv_w_et
