import os
from itertools import combinations
import numpy as np
import pandas as pd
import streamlit as st
import statsmodels.api as sm
import matplotlib.pyplot as plt
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.diagnostic import het_breuschpagan, linear_reset
from statsmodels.stats.stattools import durbin_watson
import json
import anthropic
from scipy import stats
import math
import ui
import io
from openpyxl import Workbook
from openpyxl.styles import Font, Alignment
from openpyxl.drawing.image import Image as XLImage


st.set_page_config(page_title="AI-Guided Regression Tool", layout="wide")



# ---------- Settings ----------
MAX_LEVELS = 15  # text columns with more levels than this are dropped by default
STAGES = [
    "1. Upload & Complete",
    "2. Data Picture",
    "3. Baseline Regression",
    "4. Claude Suggestions",
    "5. Pool & Search",
    "6. Results & Export",
]
MODEL_BRIEF = "claude-sonnet-5-5"  # switch to "claude-sonnet-5-5" for final runs
ALPHA = 0.05  # significance level used everywhere (confidence level = 1 - ALPHA)
CONF_LEVEL = int(round((1 - ALPHA) * 100))

# ---------- Regression functions (match Excel ToolPak layout) ----------
def run_regression(df, y_name, x_names):
    """Fit OLS with an intercept. Returns the statsmodels results object."""
    X = sm.add_constant(df[x_names].astype(float))
    y = df[y_name].astype(float)
    return sm.OLS(y, X).fit()


def toolpak_tables(model):
    """Return (regression_stats, anova, coefficients) DataFrames in Excel's layout."""
    reg = pd.DataFrame(
        {
            "Value": [
                np.sqrt(model.rsquared),
                model.rsquared,
                model.rsquared_adj,
                np.sqrt(model.mse_resid),
                int(model.nobs),
            ]
        },
        index=["Multiple R", "R Square", "Adjusted R Square", "Standard Error", "Observations"],
    )
    anova = pd.DataFrame(
        {
            "df": [model.df_model, model.df_resid, model.df_model + model.df_resid],
            "SS": [model.ess, model.ssr, model.ess + model.ssr],
            "MS": [model.ess / model.df_model, model.mse_resid, np.nan],
            "F": [model.fvalue, np.nan, np.nan],
            "Significance F": [model.f_pvalue, np.nan, np.nan],
        },
        index=["Regression", "Residual", "Total"],
    )
    ci = model.conf_int(alpha=ALPHA)
    coef = pd.DataFrame(
        {
            "Coefficients": model.params,
            "Standard Error": model.bse,
            "t Stat": model.tvalues,
            "P-value": model.pvalues,
            f"Lower {CONF_LEVEL}%": ci[0],
            f"Upper {CONF_LEVEL}%": ci[1],
        }
    )
    coef.index = ["Intercept"] + list(coef.index[1:])
    return reg, anova, coef


# ---------- Cleaning functions ----------
def load_file(uploaded):
    if uploaded.name.lower().endswith(".csv"):
        return pd.read_csv(uploaded)
    return pd.read_excel(uploaded)


def is_text(series):
    return not pd.api.types.is_numeric_dtype(series)


def flag_id_columns(df):
    """Likely ID/label columns: all-unique integers, or names like Year/ID/Index."""
    flagged = []
    for c in df.columns:
        name = str(c).lower()
        if name in ("year", "id", "index", "date") or name.endswith("_id"):
            flagged.append(c)
        elif pd.api.types.is_integer_dtype(df[c]) and df[c].is_unique and len(df) > 1:
            flagged.append(c)
    return flagged


def runnable_check(df):
    """Every condition must be True before regression is allowed."""
    numeric = all(pd.api.types.is_numeric_dtype(df[c]) for c in df.columns)
    checks = {"All columns numeric": numeric}
    if numeric:
        arr = df.to_numpy(dtype=float)
        checks["No missing values"] = not np.isnan(arr).any()
        checks["No infinite values"] = not np.isinf(arr).any()
        checks["No constant columns"] = bool((df.nunique() > 1).all())
    checks["More rows than columns"] = len(df) > df.shape[1]
    return checks


def clean_data(df, y_name, exclude, encode_high):
    """Make the table regression-ready. Returns a dict with the clean table, report, and recipe."""
    report = []
    recipe = {"target": y_name, "excluded_by_user": list(exclude)}
    work = df.drop(columns=list(exclude), errors="ignore").copy()

    # 1. Missing/infinite Y
    work[y_name] = work[y_name].replace([np.inf, -np.inf], np.nan)
    n_before = len(work)
    work = work[work[y_name].notna()]
    recipe["rows_dropped_missing_y"] = n_before - len(work)
    report.append(f"Dropped {n_before - len(work)} rows with missing {y_name}. {len(work):,} rows remain.")

    # 2. Text columns: one-hot encode (low cardinality) or drop (high cardinality)
    text_cols = [c for c in work.columns if c != y_name and is_text(work[c])]
    low = [c for c in text_cols if work[c].nunique() <= MAX_LEVELS or c in encode_high]
    high = [c for c in text_cols if c not in low]
    work[low] = work[low].fillna("Missing").astype(str)
    cat_original = work[low].copy()  # kept for categorical-vs-numeric charts in Tab 2
    work = work.drop(columns=high)
    if low:
        work = pd.get_dummies(work, columns=low, drop_first=True, dtype=int)
    recipe["one_hot_encoded"] = low
    recipe["dropped_high_cardinality"] = high
    if low:
        report.append(f"One-hot encoded (drop_first): {', '.join(map(str, low))}.")
    if high:
        report.append(f"Dropped high-cardinality text columns: {', '.join(map(str, high))}.")

    # 3. Booleans to 0/1, infinities to NaN, median imputation
    for c in work.columns:
        if pd.api.types.is_bool_dtype(work[c]):
            work[c] = work[c].astype(int)
    x_cols = [c for c in work.columns if c != y_name]
    work[x_cols] = work[x_cols].replace([np.inf, -np.inf], np.nan)
    all_nan = [c for c in x_cols if work[c].isna().all()]
    work = work.drop(columns=all_nan)
    x_cols = [c for c in x_cols if c not in all_nan]
    medians = work[x_cols].median()
    imputed = {str(c): int(work[c].isna().sum()) for c in x_cols if work[c].isna().any()}
    work[x_cols] = work[x_cols].fillna(medians)
    recipe["dropped_all_missing"] = all_nan
    recipe["median_imputed_counts"] = imputed
    if all_nan:
        report.append(f"Dropped columns that were entirely missing: {', '.join(map(str, all_nan))}.")
    if imputed:
        report.append("Filled missing values with column medians: " + ", ".join(f"{k} ({v})" for k, v in imputed.items()) + ".")

    # 4. Dummy pairs that always sum to 1 (keep only one)
    binary = [c for c in x_cols if set(work[c].unique()) <= {0, 1}]
    pair_drop = []
    for a, b in combinations(binary, 2):
        if a in pair_drop or b in pair_drop:
            continue
        if (work[a] + work[b] == 1).all():
            pair_drop.append(b)
    work = work.drop(columns=pair_drop)
    x_cols = [c for c in x_cols if c not in pair_drop]
    recipe["dropped_dummy_pair_duplicates"] = pair_drop
    if pair_drop:
        report.append(f"Dropped one column from each dummy pair that sums to 1: {', '.join(map(str, pair_drop))}.")

    # 5. Constant columns
    constant = [c for c in x_cols if work[c].nunique() <= 1]
    work = work.drop(columns=constant)
    x_cols = [c for c in x_cols if c not in constant]
    recipe["dropped_constant"] = constant
    if constant:
        report.append(f"Dropped constant columns: {', '.join(map(str, constant))}.")

    # 6. Perfect collinearity: keep a column only if it raises the matrix rank
    collinear, kept = [], []
    mat = np.ones((len(work), 1))
    rank = 1
    for c in x_cols:
        v = work[c].to_numpy(dtype=float)
        z = (v - v.mean()) / v.std()
        trial = np.column_stack([mat, z])
        r = np.linalg.matrix_rank(trial)
        if r > rank:
            mat, rank = trial, r
            kept.append(c)
        else:
            collinear.append(c)
    work = work[[y_name] + kept]
    recipe["dropped_perfectly_collinear"] = collinear
    if collinear:
        report.append(f"Dropped perfectly collinear columns: {', '.join(map(str, collinear))}.")

    recipe["final_columns"] = [str(c) for c in work.columns]
    report.append(f"Final table: {len(work):,} rows, 1 target + {len(kept)} predictors.")
    raw = df.loc[work.index].copy()  # original columns, same rows as the cleaned table
    return {"df": work, "raw": raw, "cat_original": cat_original, "report": report, "recipe": recipe}


# ---------- Stage navigation (session state) ----------
st.session_state.setdefault("unlocked", 0)
st.session_state.setdefault("stage_nav", STAGES[0])
st.session_state.setdefault("clean", None)
st.session_state.setdefault("brief", None)
st.session_state.setdefault("base_model", None)

NOTE_KEYS = ["notes_stage1", "notes_stage2", "notes_stage3", "notes_stage4", "notes_stage5","notes_stage6"]
st.session_state.setdefault("baseline_brief", None)

def reset_state():
    """Runs when a new file is uploaded or removed."""
    st.session_state.unlocked = 0
    st.session_state.stage_nav = STAGES[0]
    st.session_state.clean = None
    st.session_state.brief = None
    st.session_state.base_model = None
    st.session_state.pop("base_x", None)
    st.session_state.pop("seg_current", None)
    st.session_state.pop("seg_col", None)
    for k in NOTE_KEYS:
        st.session_state.pop(k, None)
    st.session_state.pop("seg_brief", None)
    st.session_state.baseline_brief = None
    reset_features()


def approve_stage(i):
    """Unlock and jump to the next stage."""
    st.session_state.unlocked = max(st.session_state.unlocked, i + 1)
    st.session_state.stage_nav = STAGES[i + 1]


def notes_box(label, key):
    """Free-text analyst notes, saved into the model recipe on every rerun."""
    text = st.text_area(label, key=key, height=100)
    st.session_state.clean["recipe"].setdefault("analyst_notes", {})[key] = text

















# ---------- Stage 1 ----------
def show_stage_1():
    uploaded = st.file_uploader("Upload a .csv or .xlsx file", type=["csv", "xlsx"], on_change=reset_state)
    if uploaded is None:
        ui.first_look_hint()
        return
    df = load_file(uploaded)

    # --- Headline strip: what did we just load? ---
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    if not numeric_cols:
        st.error("No numeric columns found, so there is no valid target variable.")
        return
    n_text = sum(is_text(df[c]) for c in df.columns)
    miss = df.isna().sum()
    miss = miss[miss > 0].sort_values(ascending=False)
    pct_missing = miss.sum() / df.size * 100 if df.size else 0
    ui.metric_strip([
        ("Rows", f"{len(df):,}"),
        ("Columns", f"{df.shape[1]}", "Every column in the file, before any cleaning."),
        ("Numeric / text", f"{len(numeric_cols)} / {n_text}", "Text columns are encoded or dropped in the completeness check."),
        ("Missing cells", f"{pct_missing:.1f}%", f"{int(miss.sum()):,} empty cells across {len(miss)} column(s)."),
    ])
    with st.expander("Preview the first 10 rows"):
        st.dataframe(df.head(10), use_container_width=True)
    if len(miss):
        with st.expander(f"Which columns have gaps? ({len(miss)})"):
            st.dataframe(
                pd.DataFrame({"Missing": miss, "Percent": miss / len(df) * 100}).style.format({"Missing": "{:,.0f}", "Percent": "{:.1f}%"}),
                use_container_width=True,
            )

    # --- Setup: two controls side by side ---
    st.divider()
    ui.section("Set up the analysis", "Pick the column to predict. Likely ID or label columns are pre-selected for exclusion.")
    left, right = st.columns([1, 2])
    with left:
        y_name = st.selectbox("Target variable (Y)", numeric_cols)
    others = [c for c in df.columns if c != y_name]
    flagged = [c for c in flag_id_columns(df) if c != y_name]
    with right:
        exclude = st.multiselect("Columns to exclude", others, default=flagged,
                                 help="ID, row-number or label columns carry no predictive meaning.")

    high = [c for c in others if c not in exclude and is_text(df[c]) and df[c].nunique() > MAX_LEVELS]
    encode_high = []
    if high:
        with st.expander(f"{len(high)} text column(s) will be dropped: too many levels"):
            st.caption(f"Text columns with more than {MAX_LEVELS} levels are dropped by default: {', '.join(map(str, high))}. "
                       "Pick any below to one-hot encode anyway (adds many columns).")
            encode_high = st.multiselect("One-hot encode anyway", high)

    if st.button("Run Completeness Check", type="primary"):
        st.session_state.clean = clean_data(df, y_name, exclude, encode_high)
        st.session_state.brief = None
        st.session_state.base_model = None
        st.session_state.pop("base_x", None)
        st.session_state.pop("seg_current", None)
        st.session_state.pop("seg_brief", None)
        reset_features()

    result = st.session_state.clean
    if result is None:
        return

    # --- Result: verdict, four numbers, detail one click away ---
    st.divider()
    ui.section("Completeness report", "What the app changed to make your table runnable.")
    rec, clean_df = result["recipe"], result["df"]
    checks = runnable_check(clean_df)
    removed = sum(len(rec[k]) for k in ("excluded_by_user", "dropped_high_cardinality", "dropped_all_missing",
                                        "dropped_dummy_pair_duplicates", "dropped_constant", "dropped_perfectly_collinear"))
    ui.metric_strip([
        ("Rows kept", f"{len(clean_df):,}", f"{rec['rows_dropped_missing_y']:,} row(s) dropped for a missing target value."),
        ("Predictors ready", f"{clean_df.shape[1] - 1}", "Numeric columns available for the regression."),
        ("Values filled", f"{sum(rec['median_imputed_counts'].values()):,}", "Missing predictor values replaced by the column median."),
        ("Columns removed", f"{removed}", "Excluded, too many levels, all missing, duplicate dummy, constant, or perfectly collinear."),
    ])
    if all(checks.values()):
        st.success(f"All {len(checks)} runnable-table checks passed. The table is ready for regression.")
    else:
        st.error("The table is not runnable yet: " + ", ".join(k for k, ok in checks.items() if not ok))
    with st.expander("See each check"):
        for name, ok in checks.items():
            st.write(("✅ " if ok else "❌ ") + name)
    with st.expander("What changed, step by step"):
        st.markdown("\n".join(f"- {line}" for line in result["report"]))
    with st.expander("Preview the cleaned table"):
        st.dataframe(clean_df.head(10), use_container_width=True)

    notes_box("Analyst notes (anything you noticed about the raw data or cleaning)", "notes_stage1")
    with st.expander("Model recipe so far"):
        st.json(rec)

    st.button(
        "Approve Completeness report data →",
        on_click=approve_stage,
        args=(0,),
        disabled=not all(checks.values()),
        type="primary",
    )



# ---------- Profile functions ----------
DESC_LABELS = [
    "Mean", "Standard Error", "Median", "Mode", "Standard Deviation", "Sample Variance",
    "Kurtosis", "Skewness", "Range", "Minimum", "Maximum", "Sum", "Count",
]


def descriptive_stats(df):
    """Excel Descriptive Statistics layout. Mode is NaN when no value repeats (Excel shows #N/A)."""
    cols = {}
    for c in df.columns:
        s = df[c].astype(float)
        counts = s.value_counts()
        if counts.max() > 1:
            tied = counts[counts == counts.max()].index
            mode = s[s.isin(tied)].iloc[0]  # first one appearing in the data, like Excel
        else:
            mode = np.nan
        cols[c] = [
            s.mean(), s.std() / np.sqrt(len(s)), s.median(), mode, s.std(), s.var(),
            s.kurt(), s.skew(), s.max() - s.min(), s.min(), s.max(), s.sum(), s.count(),
        ]
    return pd.DataFrame(cols, index=DESC_LABELS)


def compute_vif(df, x_cols):
    """Variance inflation factor for each predictor (constant added, then skipped)."""
    X = sm.add_constant(df[x_cols].astype(float)).to_numpy()
    out = {}
    for i, c in enumerate(x_cols, start=1):
        try:
            out[c] = variance_inflation_factor(X, i)
        except Exception:
            out[c] = np.inf
    return pd.Series(out)


def hist_fig(series, title):
    fig, ax = plt.subplots(figsize=(5, 3.2))
    ax.hist(series.dropna(), bins=30, color="#4C78A8", edgecolor="#16213A")
    ax.set_title(title)
    fig.tight_layout()
    return fig


def scatter_fig(df, x, y):
    fig, ax = plt.subplots(figsize=(5, 3.2))
    ax.scatter(df[x], df[y], s=8, alpha=0.4, color="#4C78A8")
    slope, intercept = np.polyfit(df[x], df[y], 1)
    xs = np.linspace(df[x].min(), df[x].max(), 50)
    ax.plot(xs, slope * xs + intercept, color="#E45756")
    ax.set_xlabel(str(x))
    ax.set_ylabel(str(y))
    fig.tight_layout()
    return fig


def heatmap_fig(corr):
    n = len(corr)
    size = min(14, max(6, n * 0.45))
    fig, ax = plt.subplots(figsize=(size, size * 0.85))
    im = ax.imshow(corr.to_numpy(), cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(corr.columns, rotation=90, fontsize=7)
    ax.set_yticklabels(corr.index, fontsize=7)
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    return fig


def box_fig(cat, y_values, cat_name, y_name):
    groups = [y_values[cat == lvl] for lvl in sorted(cat.unique())]
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.boxplot(groups, tick_labels=sorted(cat.unique()))
    ax.set_title(f"{y_name} by {cat_name}")
    ax.tick_params(axis="x", rotation=45, labelsize=8)
    fig.tight_layout()
    return fig


def freq_table(series):
    counts = series.value_counts()
    return pd.DataFrame({"Count": counts, "Percent": counts / counts.sum() * 100})


def bar_fig(ft, name):
    fig, ax = plt.subplots(figsize=(5, 3.2))
    ax.bar(ft.index.astype(str), ft["Count"], color="#4C78A8")
    ax.set_title(f"Count by {name}")
    ax.tick_params(axis="x", rotation=45, labelsize=8)
    fig.tight_layout()
    return fig


def five_number(df):
    out = df.quantile([0, 0.25, 0.5, 0.75, 1]).T
    out.columns = ["Min", "Q1", "Median", "Q3", "Max"]
    return out









# ---------- Claude brief + Base Data Model ----------
def get_api_key():
    """Reads the sidebar box (created once in Main), then the environment variable."""
    typed = st.session_state.get("api_key_box", "")
    return typed or os.environ.get("ANTHROPIC_API_KEY", "")


def num(v, digits=3):
    """JSON-safe number (no NaN or infinity)."""
    return round(float(np.nan_to_num(v, nan=0.0, posinf=1e6, neginf=-1e6)), digits)


def build_profile(df, recipe, summary, corr, cat_original):
    """Compact numeric profile for Claude. No raw rows are ever included."""
    y = recipe["target"]
    x_cols = [c for c in df.columns if c != y]
    pairs = []
    for a, b in combinations(x_cols, 2):
        if abs(corr.loc[a, b]) > 0.8:
            pairs.append({"a": str(a), "b": str(b), "r": num(corr.loc[a, b])})
    pairs = sorted(pairs, key=lambda p: -abs(p["r"]))[:10]

    categorical = {}
    for c in cat_original.columns:
        g = df[y].groupby(cat_original[c]).agg(["mean", "count"])
        categorical[str(c)] = {str(k): {"mean_y": num(r["mean"]), "n": int(r["count"])} for k, r in g.iterrows()}

    return {
        "target": str(y),
        "rows": int(len(df)),
        "n_predictors": len(x_cols),
        "target_summary": {
            "mean": num(df[y].mean()), "sd": num(df[y].std()),
            "min": num(df[y].min()), "max": num(df[y].max()), "skew": num(df[y].skew()),
        },
        "predictors_by_abs_corr_with_y": [
            {"name": str(c), "corr_y": num(r["Corr with Y"]), "skew": num(r["Skew"]), "vif": num(r["VIF"], 1)}
            for c, r in summary.head(25).iterrows()
        ],
        "high_vif_over_10": [str(c) for c in summary.index[summary["VIF"] > 10]],
        "highly_correlated_predictor_pairs": pairs,
        "categorical_y_means": categorical,
        "dropped_high_cardinality_text": [str(c) for c in recipe["dropped_high_cardinality"]],
        "cleaning_recipe": {k: v for k, v in recipe.items() if k != "final_columns"},
    }


BRIEF_SYSTEM = (
    "You are a statistics tutor helping an introductory college statistics student. "
    "You receive a compact JSON profile of a Completeness report dataset prepared for multiple linear regression. "
    "Write a plain-English brief with these short sections: "
    "1) Key relationships with the target (numeric and categorical), "
    "2) Anomalies or oddities, 3) Skew and possible log transforms, "
    "4) Collinearity concerns and which variables to consider dropping or combining, "
    "5) Possible panel or time-series issues, "
    "6) Which charts to look at first, "
    "7) Whether the table looks safe to regress. "
    "8) suggest 1 or at most 4 interesting interactions with the data, ie: varible that splits the data in two might offer an intersting take away"
    "Be concise, use only the numbers provided, and do not invent facts about the data."
)


def ask_claude_brief(profile, api_key, system=BRIEF_SYSTEM, teach=True):
    client = anthropic.Anthropic(api_key=api_key)
    msg = client.messages.create(
        model=MODEL_BRIEF,
        max_tokens=10000,
        system=system + (TEACH_LINE if teach else""),
        messages=[{"role": "user", "content": json.dumps(profile)}],
    )
    text = "".join(b.text for b in msg.content if b.type == "text")
    if not text:
        raise ValueError(f"Claude returned no text (stop reason: {msg.stop_reason}).")
    return text


def approve_base_model():
    """Save the Base Data Model choices to the recipe and unlock Stage 3."""
    keep = list(st.session_state.base_x)
    result = st.session_state.clean
    st.session_state.base_model = {"target": result["recipe"]["target"], "x_cols": keep}
    result["recipe"]["base_model_columns"] = keep
    st.session_state.baseline_brief = None
    reset_features()
    approve_stage(1)









# ---------- Segment Explorer ----------
SMALL_GROUP = 30  # groups smaller than this get a warning


def split_candidates(raw, y_name):
    """Columns that can define groups: text with 2..MAX_LEVELS levels, or numeric with 2+ values."""
    out = []
    for c in raw.columns:
        if c == y_name:
            continue
        n = raw[c].nunique()
        if is_text(raw[c]):
            if 2 <= n <= MAX_LEVELS:
                out.append(c)
        elif n >= 2:
            out.append(c)
    return out


def make_groups(raw, col, median_split):
    """Group label for every row (same index as raw). Missing values get their own group."""
    s = raw[col]
    if median_split:
        cut = s.median()
        labels = pd.Series(np.where(s <= cut, f"low (<= {cut:g})", f"high (> {cut:g})"), index=raw.index)
        return labels.where(s.notna(), "Missing")
    return s.astype(str).where(s.notna(), "Missing")


def segment_tables(df, y_name, x_cols, groups):
    """Per-group summary of Y, and per-group correlation of each predictor with Y plus the gap."""
    levels = sorted(groups.unique())
    summary = pd.DataFrame(
        {
            lvl: {
                "n": (groups == lvl).sum(),
                "Mean of Y": df.loc[groups == lvl, y_name].mean(),
                "SD of Y": df.loc[groups == lvl, y_name].std(),
            }
            for lvl in levels
        }
    ).T
    corr = pd.DataFrame(
        {lvl: df.loc[groups == lvl, x_cols].corrwith(df.loc[groups == lvl, y_name]) for lvl in levels}
    )
    corr["Gap"] = corr.max(axis=1) - corr.min(axis=1)
    return summary, corr.sort_values("Gap", ascending=False)


def split_label(col, median_split):
    return f"{col} (median split)" if median_split else f"{col} (each value)"


def overlay_fig(df, x, y, groups, max_groups=6):
    """X vs Y colored by group, with one trend line per group (largest groups only)."""
    levels = groups.value_counts().index[:max_groups]
    colors = plt.cm.tab10.colors
    s_all = df[x]
    use_log = s_all.min() >= 0 and abs(s_all.skew()) > 1
    xv = np.log1p(s_all) if use_log else s_all
    fig, ax = plt.subplots(figsize=(5, 3.2))
    for i, lvl in enumerate(levels):
        m = groups == lvl
        xs, ys = xv[m], df.loc[m, y]
        ax.scatter(xs, ys, s=8, alpha=0.3, color=colors[i], label=str(lvl))
        if len(xs) > 2 and xs.nunique() > 1:
            slope, intercept = np.polyfit(xs, ys, 1)
            line = np.linspace(xs.min(), xs.max(), 50)
            ax.plot(line, slope * line + intercept, color=colors[i], linewidth=2)
    ax.set_xlabel(f"ln(1+{x})" if use_log else str(x))
    ax.set_ylabel(str(y))
    ax.legend(fontsize=7)
    fig.tight_layout()
    return fig


def group_regressions(df, y_name, x_cols, groups):
    """Fit OLS inside each group. Predictors constant within a group are skipped for that group."""
    coefs, pvals, stats = {}, {}, {}
    for lvl in sorted(groups.unique()):
        sub = df[groups == lvl]
        use = [c for c in x_cols if sub[c].nunique() > 1]
        row = {"n": len(sub), "Predictors used": len(use), "R Square": np.nan, "Adjusted R Square": np.nan}
        if use and len(sub) > len(use) + 1:
            try:
                m = run_regression(sub, y_name, use)
                coefs[lvl] = m.params.drop("const")
                pvals[lvl] = m.pvalues.drop("const")
                row["R Square"], row["Adjusted R Square"] = m.rsquared, m.rsquared_adj
            except Exception:
                pass
        stats[lvl] = row
    return (
        pd.DataFrame(coefs).reindex(x_cols),
        pd.DataFrame(pvals).reindex(x_cols),
        pd.DataFrame(stats).T,
    )


SEGMENT_SYSTEM = (
    "You are a statistics tutor helping an introductory college statistics student. "
    "You receive a compact JSON summary of one way of splitting a regression dataset into groups. "
    "Write a plain-English reply with these short sections: "
    "1) Do the groups differ in average Y? "
    "2) Where does a predictor relate to Y differently across groups (use the correlation gaps)? "
    "3) Cautions: small groups, possible panel data, and how many splits have been explored "
    "(more splits means more differences will appear by chance). "
    "4) The options the analyst has: keep one pooled model, add an interaction term later, "
    "or consider separate models. Do not choose for them. "
    "5) Up to three other columns from other_split_candidates that might be worth splitting by, and why. "
    "Be concise, use only the numbers provided, treat differences as hypotheses, and do not invent facts. "
    "A null correlation means it could not be computed."
)


def num_or_none(v):
    return None if pd.isna(v) else num(v)


def build_segment_profile(cur, summary, corr, recipe, cands):
    """Compact numeric summary of the current split. No raw rows."""
    levels = [c for c in corr.columns if c != "Gap"]
    return {
        "split_column": str(cur["col"]),
        "split_method": "median split" if cur["median"] else "one group per value",
        "target": str(recipe["target"]),
        "groups": {
            str(l): {"n": int(r["n"]), "mean_y": num_or_none(r["Mean of Y"]), "sd_y": num_or_none(r["SD of Y"])}
            for l, r in summary.iterrows()
        },
        "small_groups": [str(l) for l in summary.index[summary["n"] < SMALL_GROUP]],
        "top_correlation_gaps": [
            {
                "predictor": str(p),
                "corr_with_y_by_group": {str(l): num_or_none(r[l]) for l in levels},
                "gap": num_or_none(r["Gap"]),
            }
            for p, r in corr.head(10).iterrows()
        ],
        "possible_panel_data": bool(recipe["dropped_high_cardinality"]),
        "splits_explored_so_far": len(recipe.get("splits_explored", [])),
        "other_split_candidates": [str(c) for c in cands if c != cur["col"]][:20],
    }


def run_split(col, median_split):
    """Button callback: remember the current split and count it once in the recipe."""
    st.session_state.seg_current = {"col": col, "median": median_split}
    label = split_label(col, median_split)
    explored = st.session_state.clean["recipe"].setdefault("splits_explored", [])
    if label not in explored:
        explored.append(label)


def show_segment_explorer(result):
    df, raw, recipe = result["df"], result["raw"], result["recipe"]
    y = recipe["target"]
    x_cols = [c for c in df.columns if c != y]

    ui.section("Segment Explorer", "Read-only: split the rows into groups and see whether Y, or how each predictor relates to Y, differs between them. Nothing here changes your data.")
    cands = split_candidates(raw, y)
    if not cands:
        st.info("No columns in this file can be used to split the data.")
        return
    left, right = st.columns([2, 1])
    with left:
        col = st.selectbox("Split the data by", cands, key="seg_col")
    numeric = not is_text(raw[col])
    n_levels = raw[col].nunique()
    with right:
        if numeric and n_levels > MAX_LEVELS:
            median = True
            st.caption(f"{col} has {n_levels} distinct values, so it is split at its median.")
        elif numeric and n_levels > 2:
            median = st.checkbox("Median split instead of one group per value", key=f"seg_median_{col}")
        else:
            median = False
    st.button("Compare groups", on_click=run_split, args=(col, median), type="primary")

    cur = st.session_state.get("seg_current")
    if cur is None or cur["col"] not in raw.columns:
        st.caption(f"Splits explored so far: {len(recipe.get('splits_explored', []))}")
        return
    groups = make_groups(raw, cur["col"], cur["median"])
    if groups.nunique() < 2:
        st.warning("That split produced only one group. Try another column.")
        return
    summary, corr = segment_tables(df, y, x_cols, groups)
    small = summary.index[summary["n"] < SMALL_GROUP].tolist()
    gaps = corr["Gap"].dropna()

    # --- Headline strip ---
    ui.metric_strip([
        ("Split", str(cur["col"]), "Median split" if cur["median"] else "One group per value"),
        ("Groups", f"{len(summary)}", f"Smallest group has {int(summary['n'].min()):,} rows."),
        ("Biggest gap", f"{gaps.iloc[0]:.2f}" if len(gaps) else "n/a",
         (f"{gaps.index[0]}: its correlation with Y differs most between groups." if len(gaps) else "No correlation gap could be computed.")),
        ("Splits explored", f"{len(recipe.get('splits_explored', []))}",
         "Trying many splits turns up some differences by chance, so treat findings as hypotheses."),
    ])

    # --- Group table + cautions ---
    st.markdown(f"**Groups by {cur['col']}**")
    st.dataframe(
        summary.style.format({"n": "{:,.0f}", "Mean of Y": "{:.3f}", "SD of Y": "{:.3f}"}),
        use_container_width=True,
    )
    if small:
        st.warning(f"Small group(s) (under {SMALL_GROUP} rows), so correlations here are unreliable: " + ", ".join(map(str, small)))
    if recipe["dropped_high_cardinality"]:
        st.caption("Rows repeat the same entities (panel data), so group differences can look stronger than they are.")

    # --- Correlation gaps ---
    st.markdown("**Correlation of each predictor with Y, by group** (largest gaps first)")
    st.dataframe(
        corr.style.format("{:.3f}", na_rep="n/a").background_gradient(subset=["Gap"], cmap="Oranges"),
        use_container_width=True,
    )
    st.caption("A big gap means the predictor relates to Y differently in the groups. Treat it as a hypothesis, not a proven finding.")

    # --- Overlay charts for the biggest gaps ---
    binary = [c for c in x_cols if df[c].nunique() <= 2]
    strength = corr.drop(columns="Gap").abs().max(axis=1)
    top = [c for c in corr["Gap"].dropna().index if c not in binary and strength[c] >= 0.2][:4]
    st.markdown("**Biggest-gap predictors vs Y, colored by group**")
    if groups.nunique() > 6:
        st.caption("Only the 6 largest groups are drawn.")
    if not top:
        st.caption("No non-binary predictor has a correlation of 0.2 or more with Y in any group, so there is nothing useful to plot.")
    cols = st.columns(2)
    for i, c in enumerate(top):
        with cols[i % 2]:
            fig = overlay_fig(df, c, y, groups)
            st.pyplot(fig)
            plt.close(fig)
    if top:
        st.caption("Lines with different slopes or heights mean the groups behave differently. Axes use ln(1+x) when a predictor is strongly skewed.")

    # --- Separate regressions (one click away) ---
    with st.expander("Fit the regression separately in each group"):
        if st.checkbox("Run it (all cleaned predictors)", key="seg_reg"):
            coef, pval, gstats = group_regressions(df, y, x_cols, groups)
            st.markdown("**Coefficients by group**")
            st.dataframe(coef.style.format("{:.4f}", na_rep="n/a"), use_container_width=True)
            st.dataframe(
                gstats.style.format({"n": "{:,.0f}", "Predictors used": "{:.0f}", "R Square": "{:.3f}", "Adjusted R Square": "{:.3f}"}, na_rep="n/a"),
                use_container_width=True,
            )
            st.markdown("**P-values by group**")
            st.dataframe(pval.style.format("{:.4f}", na_rep="n/a"), use_container_width=True)
            st.caption("n/a means the predictor did not vary inside that group, or the group was too small to fit. Coefficients are in each predictor's own units, so compare sign and size across groups for the same predictor.")

    # --- Claude's read ---
    st.markdown("**Claude's read on this split**")
    api_key = get_api_key()
    if not api_key:
        st.caption("Enter your API key in the sidebar to enable this.")
    label = split_label(cur["col"], cur["median"])
    profile = build_segment_profile(cur, summary, corr, recipe, cands)
    if st.button("Ask Claude about this split", disabled=not api_key):
        with st.spinner("Claude is reading the split..."):
            try:
                text = ask_claude_brief(profile, api_key, SEGMENT_SYSTEM)
                st.session_state.seg_brief = {"label": label, "text": text}
                recipe.setdefault("segment_findings", {})[label] = text
            except Exception as e:
                st.error(f"Claude call failed: {e}")
    sb = st.session_state.get("seg_brief")
    if sb and sb["label"] == label:
        with st.container(border=True):
            st.markdown(sb["text"])
    with st.expander("Exactly what Claude was sent (this split)"):
        st.json(profile)






# ---------- Stage 2 ----------
def show_stage_2():
    result = st.session_state.clean
    df, recipe = result["df"], result["recipe"]
    y = recipe["target"]
    x_cols = [c for c in df.columns if c != y]

    corr = df.corr()
    summary = pd.DataFrame(
        {
            "Corr with Y": corr[y].drop(y),
            "Skew": df[x_cols].skew(),
            "VIF": compute_vif(df, x_cols),
        }
    ).sort_values("Corr with Y", key=abs, ascending=False)
    top = summary.index[:4].tolist()
    high_vif = [str(c) for c in summary.index[summary["VIF"] > 10]]
    skewed = [str(c) for c in summary.index if abs(summary.loc[c, "Skew"]) > 1 and df[c].nunique() > 2]

    # --- Headline strip ---
    ui.section("Data picture", f"How the {len(x_cols)} predictors relate to {y} and to each other. Nothing here changes your data.")
    ui.metric_strip([
        ("Predictors available", f"{len(x_cols)}", f"Numeric columns in the cleaned table, besides {y}."),
        ("Strongest link with Y", f"{summary.iloc[0]['Corr with Y']:+.2f}",
         f"Correlation of {top[0]} with {y}. Range is -1 to 1; near 0 means little straight-line link."),
        ("High-VIF predictors", f"{len(high_vif)}", "Predictors with VIF above 10: mostly repeating other predictors, which makes coefficients unstable."),
        ("Skewed predictors", f"{len(skewed)}", "Non-binary predictors with absolute skew above 1: a long tail, where a log transform often helps."),
    ])
    st.markdown("**What this shows:** strongest links with Y are "
                + ", ".join(f"**{c}** (r = {summary.loc[c, 'Corr with Y']:+.2f})" for c in top[:3]) + ".")

    # --- Variable summary ---
    st.divider()
    ui.section("Variable summary", "Sorted by strength of correlation with Y. Darker orange = stronger link.")
    st.dataframe(
        summary.style.format({"Corr with Y": "{:.3f}", "Skew": "{:.2f}", "VIF": "{:.1f}"})
        .background_gradient(subset=["Corr with Y"], cmap="Oranges", gmap=summary["Corr with Y"].abs()),
        use_container_width=True,
    )
    how_to_read("summary")

    notes = []
    if high_vif:
        notes.append("**High collinearity (VIF above 10):** " + ", ".join(high_vif)
                     + ". Why it matters: these predictors mostly repeat others, so their coefficients can swing wildly or have the wrong sign.")
    if skewed:
        notes.append("**Strongly skewed predictors:** " + ", ".join(skewed[:8])
                     + ". Why it matters: a few huge values can dominate the fit.")
    if recipe["dropped_high_cardinality"]:
        notes.append("**Possible panel data:** " + ", ".join(map(str, recipe["dropped_high_cardinality"]))
                     + " was dropped as a high-cardinality text column. Why it matters: if it identifies repeated entities (such as one country observed many times), rows are not independent, so p-values look more certain than they are.")
    ui.health_list(notes)

    # --- Charts ---
    st.divider()
    ui.section("Charts", "The shape of Y, and its strongest predictor. More charts are one click away below.")
    c1, c2 = st.columns(2)
    with c1:
        fig = hist_fig(df[y], f"Distribution of {y}")
        st.pyplot(fig)
        plt.close(fig)
    with c2:
        fig = scatter_fig(df, top[0], y)
        st.pyplot(fig)
        plt.close(fig)
    st.caption(f"Left: how {y} is spread out. Right: {top[0]} against {y}, with the best straight-line fit in red.")

    with st.expander("More scatter plots (next strongest predictors)"):
        cols = st.columns(2)
        for i, c in enumerate(top[1:]):
            with cols[i % 2]:
                fig = scatter_fig(df, c, y)
                st.pyplot(fig)
                plt.close(fig)
        st.markdown(HOW_TO_READ["scatter"])
    with st.expander("Distribution of any predictor"):
        picked = st.selectbox("Histogram of predictor", x_cols, index=x_cols.index(top[0]))
        fig = hist_fig(df[picked], f"Distribution of {picked}")
        st.pyplot(fig)
        plt.close(fig)
    with st.expander("Correlation heatmap"):
        fig = heatmap_fig(corr)
        st.pyplot(fig)
        plt.close(fig)
        st.markdown(HOW_TO_READ["heatmap"])
    cat = result["cat_original"]
    if cat.shape[1] > 0:
        with st.expander("Y by category"):
            for c in cat.columns:
                fig = box_fig(cat[c], df[y], c, y)
                st.pyplot(fig)
                plt.close(fig)
                ft = freq_table(cat[c])
                t1, t2 = st.columns(2)
                with t1:
                    st.dataframe(ft.style.format({"Count": "{:,.0f}", "Percent": "{:.1f}%"}), use_container_width=True)
                with t2:
                    fig = bar_fig(ft, c)
                    st.pyplot(fig)
                    plt.close(fig)

    # --- Data quality per column ---
    with st.expander("Filled gaps and outliers per column"):
        q1, q3 = df.quantile(0.25), df.quantile(0.75)
        iqr = q3 - q1
        out = ((df < q1 - 1.5 * iqr) | (df > q3 + 1.5 * iqr)).sum()
        out[df.nunique() <= 2] = 0  # 0/1 columns have no meaningful outliers
        filled = recipe["median_imputed_counts"]
        qual = pd.DataFrame({
            "Filled (were missing)": [filled.get(str(c), 0) for c in df.columns],
            "Outliers (1.5×IQR rule)": out,
            "Outliers %": out / len(df) * 100,
        }, index=df.columns).sort_values("Outliers (1.5×IQR rule)", ascending=False)
        st.dataframe(qual.style.format({"Filled (were missing)": "{:,.0f}", "Outliers (1.5×IQR rule)": "{:,.0f}", "Outliers %": "{:.1f}%"}),
                     use_container_width=True)
        st.caption("A column with many filled values leans on median guesses, so treat its coefficient with more caution. "
                   "Outliers are values more than 1.5 interquartile ranges beyond the middle half of the data. They are not errors, just values that can pull a fit.")

    with st.expander("Show Class Math & Descriptive Stats"):
        st.markdown("**Descriptive statistics (Excel layout)**")
        st.dataframe(descriptive_stats(df).style.format("{:,.4f}", na_rep="#N/A"), use_container_width=True)
        st.markdown("**Five-number summary**")
        st.dataframe(five_number(df).style.format("{:,.4f}"), use_container_width=True)
        st.markdown("**Correlation matrix (Pearson)**")
        st.dataframe(corr.style.format("{:.3f}"), use_container_width=True)

    st.divider()
    show_segment_explorer(result)

    # --- Claude's brief ---
    st.divider()
    ui.section("Claude's brief")
    api_key = get_api_key()
    profile = build_profile(df, recipe, summary, corr, result["cat_original"])
    if not api_key:
        st.caption("Enter your API key in the sidebar to enable the brief.")
    if st.button("Generate Claude brief", disabled=not api_key):
        with st.spinner("Claude is reading the profile..."):
            try:
                st.session_state.brief = ask_claude_brief(profile, api_key)
            except Exception as e:
                st.error(f"Claude call failed: {e}")
    if st.session_state.brief:
        with st.container(border=True):
            st.markdown(st.session_state.brief)
    with st.expander("Exactly what Claude was sent"):
        st.json(profile)

    st.divider()
    notes_box("Analyst notes (patterns, hypotheses, or concerns from the data picture)", "notes_stage2")

    # --- Base Data Model ---
    st.divider()
    ui.section("Base Data Model", f"Choose the predictors to keep. Target (Y): {y}. Keep it plain here; transforms and interactions come in Stage 4.")
    st.multiselect("Predictors to keep", x_cols, default=x_cols, key="base_x")
    st.button(
        "Approve Base Data Model →",
        on_click=approve_base_model,
        disabled=len(st.session_state.get("base_x", x_cols)) == 0,
        type="primary",
    )
    











# ---------- Stage 3: Baseline regression ----------
def residual_diagnostics(model):
    """Numeric residual checks: shape of residuals and a test for unequal spread."""
    resid = model.resid
    return {
        "residual_skew": float(resid.skew()),
        "residual_excess_kurtosis": float(resid.kurt()),
        "breusch_pagan_p": float(het_breuschpagan(resid, model.model.exog)[1]),
        "durbin_watson": float(durbin_watson(resid)),
    }


def qq_fig(resid):
    fig, ax = plt.subplots(figsize=(5, 3.2))
    stats.probplot(resid, dist="norm", plot=ax)
    ax.set_title("Q-Q plot of residuals")
    ax.get_lines()[0].set_markersize(3)
    ax.get_lines()[0].set_markerfacecolor("#6A9AED")
    ax.get_lines()[0].set_markeredgecolor("#6A9AED")
    fig.tight_layout()
    return fig


def resid_fitted_fig(fitted, resid):
    fig, ax = plt.subplots(figsize=(5, 3.2))
    ax.scatter(fitted, resid, s=8, alpha=0.4, color="#4C78A8")
    ax.axhline(0, color="#E45756")
    ax.set_xlabel("Fitted values")
    ax.set_ylabel("Residuals")
    ax.set_title("Residuals vs fitted")
    fig.tight_layout()
    return fig


BASELINE_SYSTEM = (
    "You are a statistics tutor helping an introductory college statistics student. "
    "You receive a compact JSON summary of a baseline multiple linear regression. "
    "Write a plain-English reply with these short sections: "
    "1) Overall fit (R Square, Adjusted R Square, Significance F, standard error in Y's units). "
    f"2) Which predictors look useful and which have p-values above {ALPHA} "
    "(remember collinearity and possible panel data can distort p-values). "
    "3) Residual diagnostics: what the residual skew, excess kurtosis, Breusch-Pagan p-value and Durbin-Watson "
    "suggest about normality, unequal spread and correlated errors (Durbin-Watson only means something if rows are in a meaningful order). "
    "4) Ideas the analyst could try in the next stage (log transforms, squared terms, interactions, "
    "combining or dropping variables), tied to specific numbers. Do not choose for them. "
    "Use any analyst notes or segment findings if present. "
    "Be concise, use only the numbers provided, and do not invent facts."
)


def build_baseline_profile(model, y, x_cols, df, recipe, diag):
    coef = pd.DataFrame({"coef": model.params, "p": model.pvalues}).drop("const")
    vif = compute_vif(df, x_cols)
    return {
        "target": str(y),
        "rows": int(model.nobs),
        "n_predictors": len(x_cols),
        "r_square": num(model.rsquared, 4),
        "adjusted_r_square": num(model.rsquared_adj, 4),
        "standard_error": num(np.sqrt(model.mse_resid), 4),
        "significance_f": num(model.f_pvalue, 6),
        "coefficients_by_p_value": [
            {"name": str(c), "coef": num(r["coef"], 4), "p": num(r["p"], 4)}
            for c, r in coef.sort_values("p").head(30).iterrows()
        ],
        f"not_significant_at_{ALPHA}": [str(c) for c in coef.index[coef["p"] > ALPHA]],
        "high_vif_over_10": [str(c) for c in vif.index[vif > 10]],
        "residual_diagnostics": {k: num(v, 4) for k, v in diag.items()},
        "possible_panel_data": bool(recipe["dropped_high_cardinality"]),
        "splits_explored": len(recipe.get("splits_explored", [])),
        "segment_findings": recipe.get("segment_findings", {}),
        "analyst_notes": {k: v for k, v in recipe.get("analyst_notes", {}).items() if v.strip()},
    }


def reference_caption(result, model_cols):
    """Name the dropped (reference) level of each one-hot encoded text column used in the model."""
    cat, parts = result["cat_original"], []
    for c in cat.columns:
        levels = sorted(cat[c].unique())  # get_dummies(drop_first=True) drops the first sorted level
        if any(f"{c}_{lvl}" in model_cols for lvl in levels[1:]):
            parts.append(f"**{c}** = {levels[0]}")
    if parts:
        st.caption("Reference categories (the level each dummy variable is compared against): " + "; ".join(parts)
                   + ". A dummy coefficient is the difference in Y for that level versus the reference level, holding the other predictors fixed.")


def show_influential_rows(model, raw, top_n=10):
    """Expander listing the rows with the largest Cook's distance, next to their original data."""
    with st.expander("Which rows are most influential? (Cook's distance)"):
        try:
            cooks = pd.Series(model.get_influence().cooks_distance[0], index=model.resid.index)
        except Exception:
            st.caption("Cook's distance could not be computed for this model.")
            return
        cut = 4 / int(model.nobs)
        top = cooks.nlargest(top_n)
        stats_cols = pd.DataFrame({
            "Cook's distance": top,
            "Fitted": model.fittedvalues.loc[top.index],
            "Residual": model.resid.loc[top.index],
        }).round(4)
        st.dataframe(pd.concat([stats_cols, raw.loc[top.index]], axis=1), use_container_width=True)
        st.caption(f"{int((cooks > cut).sum()):,} of {len(cooks):,} rows are above the rule of thumb 4/n = {cut:.4f}; the {len(top)} largest are shown, "
                   "with the original columns from your file (row labels are positions in the uploaded table, starting at 0). "
                   "These are rows to look at, not rows to delete: check for data-entry errors, or whether they are a genuinely different kind of case. "
                   "Refitting without them shows how much they steer the coefficients.")


def coef_ci_fig(model, y, x_cols, df, top_n=15):
    """Standardized effect of each predictor with its confidence bar (strongest at top)."""
    ratio = df[x_cols].std() / df[y].std()  # puts every predictor on the same scale
    ci = model.conf_int(alpha=ALPHA).drop("const")
    beta = model.params.drop("const") * ratio
    lo, hi = ci[0] * ratio, ci[1] * ratio
    sig = model.pvalues.drop("const") <= ALPHA
    order = beta.abs().sort_values(ascending=False).index[:top_n][::-1]
    fig, ax = plt.subplots(figsize=(7, max(2.5, 0.38 * len(order) + 1)))
    for i, c in enumerate(order):
        color = ui.ACCENT if sig[c] else ui.SOFT
        ax.plot([lo[c], hi[c]], [i, i], color=color, linewidth=2)
        ax.plot(beta[c], i, "o", color=color)
    ax.axvline(0, color=ui.HIGHLIGHT, linestyle="--", linewidth=1)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([str(c) for c in order], fontsize=8)
    ax.set_xlabel("Change in Y (in SDs of Y) per 1 SD change in the predictor")
    fig.tight_layout()
    return fig


def show_stage_3():
    result, bm = st.session_state.clean, st.session_state.base_model
    if bm is None:
        st.info("Approve a Base Data Model in Stage 2 first.")
        return
    df, recipe = result["df"], result["recipe"]
    y, x_cols = bm["target"], bm["x_cols"]

    model = run_regression(df, y, x_cols)
    reg, anova, coef = toolpak_tables(model)
    diag = residual_diagnostics(model)
    recipe["baseline"] = {
        "r_square": float(model.rsquared),
        "adjusted_r_square": float(model.rsquared_adj),
        "standard_error": float(np.sqrt(model.mse_resid)),
        "significance_f": float(model.f_pvalue),
        "n_predictors": len(x_cols),
        **diag,
    }
    se = float(np.sqrt(model.mse_resid))
    n_sig = int((model.pvalues.drop("const") <= ALPHA).sum())

    # --- Headline strip ---
    ui.section("Baseline regression", f"Predicting {y} from the {len(x_cols)} predictors in your Base Data Model.")
    ui.metric_strip([
        ("Adjusted R Square", f"{model.rsquared_adj:.3f}", f"Share of the variation in Y explained, after penalizing extra predictors. Plain R Square is {model.rsquared:.4f}."),
        ("Typical miss", f"±{se:.2f}", f"Standard Error, in {y} units. About 95% of predictions land within ±{2 * se:.2f}."),
        ("Useful predictors", f"{n_sig} of {len(x_cols)}", f"Predictors with a p-value at or below {ALPHA}."),
        ("Model significance", f"{model.f_pvalue:.3g}", f"Significance F. Below {ALPHA} means the model as a whole beats guessing the average."),
    ])
    how_to_read("fit")
    st.markdown("**At a glance**")
    st.markdown("\n".join(f"- {t}" for t in takeaways(model, y, x_cols, df)))

    # --- Coefficients ---
    st.divider()
    ui.section("Which predictors matter?", f"Each dot is a predictor's effect; the bar is its {CONF_LEVEL}% confidence range.")
    fig = coef_ci_fig(model, y, x_cols, df)
    st.pyplot(fig)
    plt.close(fig)
    st.caption("Blue = significant, gray = not. A bar that crosses the dashed zero line means the data cannot rule out no effect. Effects are in standard deviations so predictors with different units can be compared. Only the 15 strongest are drawn.")
    reference_caption(result, x_cols)
    show_influential_rows(model, result["raw"])
    with st.expander("All coefficients (smallest p-value first)"):
        ct = coef.drop(index="Intercept")[["Coefficients", "P-value"]].sort_values("P-value")
        ct[f"Significant at {ALPHA}"] = np.where(ct["P-value"] <= ALPHA, "Yes", "No")
        st.dataframe(ct.style.format({"Coefficients": "{:.4f}", "P-value": "{:.4f}"}), use_container_width=True)
        st.markdown(HOW_TO_READ["coef"])
    with st.expander("Excel ToolPak tables"):
        st.markdown("**Regression Statistics**")
        st.dataframe(reg.style.format("{:,.6f}"), use_container_width=True)
        st.markdown("**ANOVA**")
        st.dataframe(anova.style.format("{:,.6g}", na_rep=""), use_container_width=True)
        st.markdown("**Coefficients**")
        st.dataframe(coef.style.format("{:,.6g}"), use_container_width=True)

    # --- Residual diagnostics (one click away) ---
    with st.expander("How are the errors behaving? (residual diagnostics)"):
        ui.metric_strip([
            ("Residual skew", f"{diag['residual_skew']:.3f}"),
            ("Excess kurtosis", f"{diag['residual_excess_kurtosis']:.3f}"),
            ("Breusch-Pagan p", f"{diag['breusch_pagan_p']:.3g}"),
            ("Durbin-Watson", f"{diag['durbin_watson']:.2f}"),
        ])
        st.caption(f"Skew and kurtosis near 0 suggest roughly normal residuals. A Breusch-Pagan p below {ALPHA} suggests the spread of residuals is not constant. Durbin-Watson near 2 means neighboring rows' errors are not correlated; it only means something if your file is sorted meaningfully (for example by entity, then year).")
        c1, c2, c3 = st.columns(3)
        with c1:
            fig = qq_fig(model.resid)
            st.pyplot(fig)
            plt.close(fig)
        with c2:
            fig = resid_fitted_fig(model.fittedvalues, model.resid)
            st.pyplot(fig)
            plt.close(fig)
        with c3:
            fig = hist_fig(model.resid, "Histogram of residuals")
            st.pyplot(fig)
            plt.close(fig)
        if recipe["dropped_high_cardinality"]:
            st.caption("Panel-data caution: repeated entities mean rows are not independent, so p-values here are likely too optimistic.")
        st.markdown(HOW_TO_READ["resid"])

    # --- Data health notes ---
    notes = health_notes(model, x_cols, df, recipe, diag)
    recipe["health_notes"] = notes
    st.divider()
    ui.section("Data health notes",
               f"{len(notes)} thing(s) worth a look. Fixes are covered in the Stage 6 report." if notes
               else "Automatic checks for the assumptions behind regression.")
    ui.health_list(notes)

    # --- Claude's read ---
    st.divider()
    ui.section("Claude's read on the baseline")
    api_key = get_api_key()
    if not api_key:
        st.caption("Enter your API key in the sidebar to enable this.")
    profile = build_baseline_profile(model, y, x_cols, df, recipe, diag)
    profile["health_notes"] = notes
    if st.button("Ask Claude about the baseline", disabled=not api_key):
        with st.spinner("Claude is reading the baseline..."):
            try:
                st.session_state.baseline_brief = ask_claude_brief(profile, api_key, BASELINE_SYSTEM)
            except Exception as e:
                st.error(f"Claude call failed: {e}")
    if st.session_state.baseline_brief:
        with st.container(border=True):
            st.markdown(st.session_state.baseline_brief)
        recipe["baseline_claude_read"] = st.session_state.baseline_brief
    with st.expander("Exactly what Claude was sent"):
        st.json(profile)

    st.divider()
    notes_box("Analyst notes (what stands out in the baseline and diagnostics)", "notes_stage3")
    st.button("Approve baseline →", on_click=approve_stage, args=(2,), type="primary")













# ---------- Stage 4: Feature workshop ----------
OPS = ["log", "square", "interaction", "combine", "drop"]
HOWS = ["sum", "mean", "ratio", "difference"]
PLAN_COLS = ["op", "col_a", "col_b", "how", "reason"]


def reset_features():
    """Clear the feature plan (new file, new cleaning, or a new Base Data Model)."""
    for k in ["plan_df", "suggested", "feature_work"]:
        st.session_state.pop(k, None)
    st.session_state.plan_version = st.session_state.get("plan_version", 0) + 1
    c = st.session_state.get("clean")
    if c:
        c.pop("feature_table", None)
        for k in ("feature_plan", "pool_columns", "engineered_columns"):
            c["recipe"].pop(k, None)
    reset_search()


def build_feature(step, pool):
    """Compute one new column from the menu. Raises ValueError with a friendly reason if invalid."""
    op, cols, how = step["op"], step["cols"], step.get("how")
    need = 2 if op in ("interaction", "combine") else 1
    if len(cols) != need:
        raise ValueError(f"{op} needs {need} column(s), got {len(cols)}")
    missing = [c for c in cols if c not in pool.columns]
    if missing:
        raise ValueError("unknown column(s): " + ", ".join(map(str, missing)))
    a = pool[cols[0]].astype(float)
    if op == "log":
        if a.min() <= -1:
            raise ValueError(f"{cols[0]} has values at or below -1, so ln(1+x) is invalid")
        return f"ln1p_{cols[0]}", np.log1p(a)
    if op == "square":
        return f"{cols[0]}_sq", a ** 2
    b = pool[cols[1]].astype(float)
    if op == "interaction":
        return f"{cols[0]}_x_{cols[1]}", a * b
    if how == "sum":
        return f"sum_{cols[0]}_{cols[1]}", a + b
    if how == "mean":
        return f"mean_{cols[0]}_{cols[1]}", (a + b) / 2
    if how == "difference":
        return f"{cols[0]}_minus_{cols[1]}", a - b
    if how == "ratio":
        if (b == 0).any():
            raise ValueError(f"{cols[1]} contains zeros, so the ratio is undefined")
        return f"{cols[0]}_per_{cols[1]}", a / b
    raise ValueError("combine needs how = sum, mean, ratio or difference")


def apply_plan(df, y, x_cols, plan):
    """Apply steps in order. Returns the working table and a log of (step, ok, message)."""
    pool = df.copy()  # every cleaned column plus anything built so far
    work = df[[y] + list(x_cols)].copy()  # starts as the Base Data Model
    log = []
    for i, step in enumerate(plan, start=1):
        label = f"{i}. {step['op']} " + ", ".join(map(str, step["cols"])) + (f" ({step['how']})" if step.get("how") else "")
        try:
            if step["op"] == "drop":
                gone = [c for c in step["cols"] if c in work.columns and c != y]
                if not gone:
                    raise ValueError("column is not in the working table")
                work = work.drop(columns=gone)
                log.append((label, "✅", f"dropped {gone[0]}"))
                continue
            name, s = build_feature(step, pool)
            if name in work.columns:
                raise ValueError(f"{name} already exists")
            if not np.isfinite(s).all():
                raise ValueError("result has missing or infinite values")
            if s.nunique() <= 1:
                raise ValueError("result is constant")
            pool[name] = s
            work[name] = s
            log.append((label, "✅", f"built {name}"))
        except ValueError as e:
            log.append((label, "❌", str(e)))
    return work, log


def step_from_row(row):
    def val(k):
        v = row.get(k)
        return None if pd.isna(v) or v == "" else v
    op = val("op")
    if op is None:
        return None
    cols = [c for c in (val("col_a"), val("col_b")) if c is not None]
    return {"op": op, "cols": cols, "how": val("how"), "reason": val("reason") or ""}


def parse_plan(text, avail):
    """Parse Claude's JSON. Rows with an invalid op or unknown column are skipped and reported."""
    t = text.strip().replace("```json", "").replace("```", "").strip()
    try:
        data = json.loads(t)
    except json.JSONDecodeError:
        data = json.loads(t[t.index("{"): t.rindex("}") + 1])
    rows, skipped = [], []
    for s in data.get("plan", []):
        cols = s.get("cols") or []
        how = s.get("how") if s.get("how") in HOWS else None
        if s.get("op") not in OPS or not cols or any(c not in avail for c in cols[:2]):
            skipped.append(f"{s.get('op')} {cols}")
            continue
        rows.append({
            "op": s["op"], "col_a": cols[0], "col_b": cols[1] if len(cols) > 1 else None,
            "how": how, "reason": s.get("reason", ""),
        })
    return data.get("explanation", ""), pd.DataFrame(rows, columns=PLAN_COLS, dtype=object), skipped


FEATURE_SYSTEM = (
    "You are a statistics tutor helping an introductory college statistics student. "
    "You receive a compact JSON summary of a baseline multiple linear regression plus the columns available. "
    "Suggest a feature plan using ONLY these operations: "
    "log (ln(1+x) of one column), square (one column squared), interaction (product of two columns), "
    "combine (two columns, how = sum, mean, ratio or difference), drop (remove one column from the model). "
    "Use ONLY column names listed in available_columns. Give at most 8 steps, each tied to specific numbers "
    "in the summary (skew, VIF, p-values, residual diagnostics, segment findings, analyst notes). "
    "Prefer few, well-justified steps. "
    "Respond with JSON only, no markdown fences, in exactly this shape: "
    '{"explanation": "short plain-English overview", "plan": [{"op": "log", "cols": ["column_name"], '
    '"how": null, "reason": "one sentence"}]}. '
    "Use how only for combine. Do not invent facts."
)


def approve_features(plan):
    """Save the plan and the new pool of columns, then unlock Stage 5."""
    result = st.session_state.clean
    work = st.session_state.feature_work
    y = result["recipe"]["target"]
    result["feature_table"] = work
    result["recipe"]["feature_plan"] = plan
    result["recipe"]["pool_columns"] = [str(c) for c in work.columns if c != y]
    result["recipe"]["engineered_columns"] = [str(c) for c in work.columns if c not in result["df"].columns]
    reset_search()
    approve_stage(3)


def show_stage_4():
    result, bm = st.session_state.clean, st.session_state.base_model
    if bm is None or "baseline" not in result["recipe"]:
        st.info("Approve the baseline in Stage 3 first.")
        return
    df, recipe = result["df"], result["recipe"]
    y, x_cols = bm["target"], bm["x_cols"]
    avail = [c for c in df.columns if c != y]

    ui.section("Feature workshop", "Claude proposes steps from a fixed menu. You edit, add, or delete rows. Python checks every step and builds the columns. Your choices decide what goes forward.")
    strip = st.container()  # filled in below, once the plan has been checked

    with st.expander("What Stage 3 flagged (for reference while you choose features)"):
        ui.health_list(recipe.get("health_notes", []))

    # --- Ask Claude ---
    api_key = get_api_key()
    if not api_key:
        st.caption("Enter your API key in the sidebar to ask Claude. You can still build a plan by hand.")
    model = run_regression(df, y, x_cols)
    profile = build_baseline_profile(model, y, x_cols, df, recipe, residual_diagnostics(model))
    profile["available_columns"] = [str(c) for c in avail]
    profile["in_base_model"] = [str(c) for c in x_cols]
    profile["column_skew"] = {str(c): num(df[c].skew(), 2) for c in avail}
    profile["columns_with_values_at_or_below_minus_1"] = [str(c) for c in avail if df[c].min() <= -1]
    if st.button("Ask Claude for a feature plan", disabled=not api_key):
        with st.spinner("Claude is building a plan..."):
            try:
                text = ask_claude_brief(profile, api_key, FEATURE_SYSTEM, teach=False)
                expl, plan_df, skipped = parse_plan(text, avail)
                st.session_state.suggested = {"explanation": expl, "skipped": skipped}
                st.session_state.plan_df = plan_df
                st.session_state.plan_version = st.session_state.get("plan_version", 0) + 1
                recipe["feature_claude_explanation"] = expl
            except Exception as e:
                st.error(f"Claude call or JSON parsing failed: {e}")
    sug = st.session_state.get("suggested")
    if sug:
        with st.container(border=True):
            st.markdown(sug["explanation"])
        if sug["skipped"]:
            st.warning("Skipped invalid suggestions: " + "; ".join(sug["skipped"]))
    with st.expander("Exactly what Claude was sent"):
        st.json(profile)

    # --- The plan ---
    st.divider()
    ui.section("Feature plan", "Edit cells, add rows at the bottom, or select a row and press Delete.")
    plan_df = st.session_state.get("plan_df")
    if plan_df is None:
        plan_df = pd.DataFrame(columns=PLAN_COLS, dtype=object)
    edited = st.data_editor(
        plan_df,
        num_rows="dynamic",
        key=f"plan_editor_{st.session_state.get('plan_version', 0)}",
        use_container_width=True,
        column_config={
            "op": st.column_config.SelectboxColumn("Operation", options=OPS),
            "col_a": st.column_config.SelectboxColumn("Column A", options=[str(c) for c in avail]),
            "col_b": st.column_config.SelectboxColumn("Column B (interaction/combine)", options=[str(c) for c in avail]),
            "how": st.column_config.SelectboxColumn("How (combine only)", options=HOWS),
            "reason": st.column_config.TextColumn("Reason"),
        },
    )
    plan = [s for s in (step_from_row(r) for _, r in edited.iterrows()) if s]
    work, log = apply_plan(df, y, x_cols, plan)

    # --- Check results ---
    ok_n = sum(1 for _, ok, _ in log if ok == "✅")
    bad_n = len(log) - ok_n
    log_df = pd.DataFrame(log, columns=["Step", "OK", "Result"])
    if bad_n:
        st.warning(f"{bad_n} step(s) could not be built and were skipped. Fix or delete them below.")
        st.dataframe(log_df, use_container_width=True, hide_index=True)
    elif log:
        with st.expander(f"All {ok_n} step(s) built successfully"):
            st.dataframe(log_df, use_container_width=True, hide_index=True)

    new_cols = [c for c in work.columns if c not in df.columns]
    pool_n = work.shape[1] - 1
    checks = runnable_check(work)
    if not all(checks.values()):
        st.error("The working table is not runnable: " + ", ".join(k for k, ok in checks.items() if not ok))
    st.caption("A step can make columns that overlap heavily with their parents (for example a sum plus both originals). That is fine, because Stage 5 tests subsets instead of everything at once.")

    with strip:
        ui.metric_strip([
            ("Baseline Adjusted R Square", f"{recipe['baseline']['adjusted_r_square']:.3f}",
             f"The score to beat, from {len(x_cols)} predictors in your Base Data Model."),
            ("Plan steps built", f"{ok_n} of {len(log)}" if log else "0",
             "Steps that passed Python's checks (right number of columns, valid math, result not constant)."),
            ("Engineered columns", f"{len(new_cols)}", "New columns built from your plan."),
            ("Pool for Stage 5", f"{pool_n}",
             "Predictors the search can choose from. Around 10 to 20 works best, since larger pools are searched by random sampling."),
        ])

    st.session_state.feature_work = work
    st.divider()
    notes_box("Analyst notes (why you chose these features, what you rejected)", "notes_stage4")
    st.button("Approve feature plan →", on_click=approve_features, args=(plan,),
              disabled=not all(checks.values()), type="primary")











# ---------- Teaching aids and data health ----------
TEACH_LINE = (
    " Also: define any statistical term in a few words the first time you use it, "
    "and say in one sentence why each issue matters. Where the numbers hint at them, "
    "mention outliers or influential rows, curvature, panel data, and winner's curse risk. "
    "Do not give step-by-step fixes. Finish with a short 'Pro takeaways' section of 3 bullets."
)

HOW_TO_READ = {
    "summary": "**Corr with Y** is how strongly a predictor moves with Y (-1 to 1; near 0 means little straight-line link). **VIF** above 10 means a predictor is mostly repeating other predictors, which makes its coefficient unstable. **Skew** far from 0 means a long tail, and a log transform often helps.",
    "scatter": "Each dot is one row; the red line is the best straight-line fit. A tight cloud around the line means a strong link. A bend in the dots means the link is curved, so a straight line fits poorly. A few far-off dots are outliers.",
    "heatmap": "Red means two columns rise together, blue means one rises as the other falls. Look at the row for Y to see what matters, then look for big red or blue blocks among predictors: those pairs overlap and cause high VIF.",
    "fit": f"**R Square** is the share of Y's ups and downs the model explains. **Adjusted R Square** is the same but penalizes useless predictors, so it is the fairer score. **Standard Error** is the typical miss, in Y's own units. **Significance F** below {ALPHA} means the model as a whole beats guessing the average.",
    "coef": f"Each coefficient is the change in Y for a 1-unit increase in that predictor, holding the others fixed. A **P-value** below {ALPHA} means the link is unlikely to be pure chance. Coefficients are in each predictor's own units, so compare sizes with care.",
    "resid": "**Q-Q plot:** dots hugging the line mean residuals are roughly normal. **Residuals vs fitted:** you want a shapeless band around 0. A funnel means unequal spread, and a U or arch means curvature. **Histogram:** should look like a bell shape centered on 0.",
    "search": "Each bar counts models with that Adjusted R Square. The red dashed line is your baseline; bars to its right beat it. Adjusted R Square rewards fit but penalizes extra predictors, so it is a fairer judge than R Square. If the best models barely beat the baseline, the extra features did not add much. If the top models are nearly tied, prefer the simpler one.",
    "holdout": "The **holdout** rows were hidden from the search. **Holdout R Square** is how well each model predicts those unseen rows (it can go negative if the model is worse than guessing the average). **Shrinkage** is the training Adjusted R Square minus the holdout R Square. A big shrinkage means the model looked better than it really is, which is winner's curse made visible. Compare shrinkage across models and against the baseline. Do not re-pick the winner by holdout score, or the holdout stops being unseen.",
}


def how_to_read(key):
    with st.expander("How to read this"):
        st.markdown(HOW_TO_READ[key])


def takeaways(model, y, x_cols, df):
    """Plain-English headline points a pro would pull at a glance."""
    se = float(np.sqrt(model.mse_resid))
    std = (model.params.drop("const") * df[x_cols].std() / df[y].std()).abs().sort_values(ascending=False)
    n_sig = int((model.pvalues.drop("const") <= ALPHA).sum())
    return [
        f"The model explains **{model.rsquared * 100:.1f}%** of the variation in {y} ({model.rsquared_adj * 100:.1f}% after penalizing extra predictors).",
        f"A typical prediction misses by about **{se:.2f}** {y} units; roughly 95% of predictions land within **±{2 * se:.2f}**.",
        "Biggest influence on Y (after putting predictors on the same scale): " + ", ".join(map(str, std.index[:3])) + ".",
        f"**{n_sig} of {len(x_cols)}** predictors are statistically significant at {ALPHA}.",
    ]


def health_notes(model, x_cols, df, recipe, diag):
    """What looks wrong and why it matters. Fixes are saved for the Stage 6 report."""
    notes = []
    n = int(model.nobs)
    if recipe["dropped_high_cardinality"]:
        notes.append("**Panel data:** the same entities likely repeat across rows, so rows are not independent. Why it matters: p-values and confidence intervals look more certain than they really are.")
    if diag["breusch_pagan_p"] < ALPHA:
        notes.append(f"**Unequal spread (Breusch-Pagan p = {diag['breusch_pagan_p']:.3g}):** prediction errors are not the same size everywhere. Why it matters: coefficients are fine, but standard errors and p-values are unreliable. Check the residuals-vs-fitted chart to see how dramatic it is.")
    if abs(diag["residual_skew"]) > 1 or abs(diag["residual_excess_kurtosis"]) > 3:
        notes.append("**Residuals are not close to normal** (large skew or heavy tails). Why it matters: confidence intervals and p-values assume bell-shaped errors.")
    dw = diag["durbin_watson"]
    if dw < 1.5 or dw > 2.5:
        notes.append(f"**Correlated errors (Durbin-Watson = {dw:.2f}; about 2 is healthy):** neighboring rows have related errors. Why it matters: standard errors and p-values look more certain than they are. This depends on row order, so it is only meaningful if the file is sorted meaningfully (for example by entity, then year). It detects the problem but does not fix it.")
    try:
        p_reset = float(linear_reset(model, power=2, test_type="fitted", use_f=True).pvalue)
        if p_reset < 0.05:
            notes.append(f"**Possible curvature (RESET test p = {p_reset:.3g}):** a straight-line model may be missing a bend. Why it matters: predictions will be biased at the ends and middle. Log or squared terms (Stage 4) are the usual remedy.")
    except Exception:
        pass
    try:
        cooks = model.get_influence().cooks_distance[0]
        share = float((cooks > 4 / n).mean())
        if cooks.max() > 1 or share > 0.05:
            notes.append(f"**Influential rows:** {share * 100:.1f}% of rows have a Cook's distance above 4/n (largest is {cooks.max():.2f}). Why it matters: a few extreme rows may be steering the coefficients.")
    except Exception:
        pass
    vif = compute_vif(df, x_cols)
    high = [str(c) for c in vif.index[vif > 10]]
    if high:
        notes.append("**High collinearity (VIF above 10):** " + ", ".join(high) + ". Why it matters: coefficients for these can swing wildly or have the wrong sign.")
    skewed = [str(c) for c in x_cols if abs(df[c].skew()) > 1 and df[c].nunique() > 2]
    if skewed:
        notes.append("**Strongly skewed predictors:** " + ", ".join(skewed[:8]) + ". Why it matters: a few huge values can dominate the fit.")
    if len(recipe.get("splits_explored", [])) >= 5:
        notes.append(f"**Many splits explored ({len(recipe['splits_explored'])}):** some group differences will appear by chance, so treat them as hypotheses.")
    return notes












# ---------- Stage 5: Pool & Search ----------
FULL_LIMIT = 32767      # run every subset when there are this many or fewer
DEFAULT_SAMPLE = 5000   # random subsets when the pool is larger
TOP_N = 25              # leaderboard size (these are refit with statsmodels)
SEARCH_SEED = 42        # fixed so a sampled search is reproducible
HOLDOUT_FRAC = 0.20     # share of rows set aside, never seen by the search
HOLDOUT_SEED = 42       # fixed so the split is reproducible (Stage 6 re-creates it)
SEARCH_RECIPE_KEYS = ("models_searched", "search_mode", "search_pool", "search_max_size",
                      "search_skipped", "search_benchmarks", "leaderboard", "search_starts", "holdout","final_model","final_report")


def reset_search():
    """Clear Stage 5 results and refresh the pool picker."""
    st.session_state.pop("search", None)
    st.session_state.pop("final_report", None)
    st.session_state.pop("excel_file", None)
    st.session_state.pool_version = st.session_state.get("pool_version", 0) + 1
    c = st.session_state.get("clean")
    if c:
        for k in SEARCH_RECIPE_KEYS:
            c["recipe"].pop(k, None)


def entity_candidates(raw, min_levels=10):
    """Text columns whose values repeat across rows (for example Country): possible entity IDs."""
    return [c for c in raw.columns
            if is_text(raw[c]) and min_levels <= raw[c].nunique() <= len(raw) / 2]


def split_holdout(index, raw, entity_col, frac=HOLDOUT_FRAC, seed=HOLDOUT_SEED):
    """Boolean array, True = holdout row. Random rows, or whole entities if entity_col is given."""
    rng = np.random.default_rng(seed)
    if entity_col:
        ents = raw.loc[index, entity_col].astype(str)
        uniq = ents.unique()
        held = rng.choice(uniq, size=max(1, int(round(len(uniq) * frac))), replace=False)
        return ents.isin(held).to_numpy()
    mask = np.zeros(len(index), dtype=bool)
    mask[rng.choice(len(index), size=max(1, int(round(len(index) * frac))), replace=False)] = True
    return mask


def fit_and_score(train, test, y, names):
    """Fit on train rows, score on test rows (out-of-sample R Square and typical error)."""
    m = run_regression(train, y, names)
    X = sm.add_constant(test[names].astype(float), has_constant="add")
    err = test[y].astype(float) - m.predict(X)
    sst = float(((test[y] - test[y].mean()) ** 2).sum())
    return {"train_adj": float(m.rsquared_adj),
            "holdout_r2": 1 - float((err ** 2).sum()) / sst,
            "holdout_rmse": float(np.sqrt((err ** 2).mean()))}


def attach_holdout(res, work, df, mask, y, base_cols, entity_col):
    """Add holdout columns to the leaderboard and score the baseline. Fit on train, score on holdout."""
    train, test = work[~mask], work[mask]
    for row in res["top"]:
        names = row["Variables"].split(", ")
        if not all(n in train.columns for n in names):
            continue
        h = fit_and_score(train, test, y, names)
        row["Holdout R Square"] = h["holdout_r2"]
        row["Holdout Error (RMSE)"] = h["holdout_rmse"]
        row["Shrinkage"] = row["Adjusted R Square"] - h["holdout_r2"]
        row["Error Ratio"] = h["holdout_rmse"] / row["Standard Error"]
    base = None
    if all(c in df.columns for c in base_cols):
        base = fit_and_score(df.loc[train.index], df.loc[test.index], y, base_cols)
    res["holdout"] = {"mode": "whole entities" if entity_col else "random rows",
                      "entity_column": entity_col, "fraction": HOLDOUT_FRAC, "seed": HOLDOUT_SEED,
                      "n_train": int((~mask).sum()), "n_holdout": int(mask.sum()), "baseline": base}


def count_subsets(k, max_size):
    return sum(math.comb(k, s) for s in range(1, max_size + 1))


def make_subsets(k, max_size, sample_n, must=()):
    """Index tuples to test. All if few enough, otherwise a random sample plus anchor models."""
    if count_subsets(k, max_size) <= FULL_LIMIT:
        return [c for s in range(1, max_size + 1) for c in combinations(range(k), s)], "all"
    rng = np.random.default_rng(SEARCH_SEED)
    seen, tries = set(), 0
    while len(seen) < sample_n and tries < sample_n * 20:
        size = int(rng.integers(1, max_size + 1))
        seen.add(tuple(sorted(rng.choice(k, size=size, replace=False).tolist())))
        tries += 1
    anchors = [tuple(range(k))] + [tuple(i for i in range(k) if i != j) for j in range(k)]
    anchors += [tuple(sorted(m)) for m in must]
    seen.update(a for a in anchors if 0 < len(a) <= max_size)
    return sorted(seen), "sample"


def score_subsets(df, y, cols, subsets, progress=None):
    """Fast Adjusted R Square for each subset (same OLS answer, tiny matrix solves)."""
    n = len(df)
    X = df[cols].to_numpy(dtype=float)
    sd = X.std(axis=0)
    sd[sd == 0] = 1.0  # a column that is constant in these rows becomes all zeros and is skipped as singular
    Z = (X - X.mean(axis=0)) / sd
    yc = df[y].to_numpy(dtype=float)
    yc = yc - yc.mean()
    G, c, sst = Z.T @ Z, Z.T @ yc, float(yc @ yc)
    keep, scores = [], []
    for i, idx in enumerate(subsets):
        idx = list(idx)
        k = len(idx)
        Gs = G[np.ix_(idx, idx)]
        if n - k - 1 <= 0 or np.linalg.cond(Gs) > 1e10:
            continue  # singular or too many predictors: skip
        b = np.linalg.solve(Gs, c[idx])
        r2 = float(b @ c[idx]) / sst
        keep.append(idx)
        scores.append(1 - (1 - r2) * (n - 1) / (n - k - 1))
        if progress is not None and i % 500 == 0:
            progress.progress(min(i / len(subsets), 1.0))
    return keep, np.array(scores), len(subsets) - len(keep)


def run_search(df, y, cols, max_size, sample_n, progress=None, must=()):
    subsets, mode = make_subsets(len(cols), max_size, sample_n, must)
    keep, scores, skipped = score_subsets(df, y, cols, subsets, progress)
    if len(scores) == 0:
        raise ValueError("No valid models could be fitted from this pool.")
    order = np.argsort(-scores)
    top = []
    for rank, j in enumerate(order[:TOP_N], start=1):
        names = [cols[i] for i in keep[j]]
        m = run_regression(df, y, names)  # exact statsmodels numbers for the leaderboard
        top.append({
            "Rank": rank,
            "Adjusted R Square": float(m.rsquared_adj),
            "R Square": float(m.rsquared),
            "Standard Error": float(np.sqrt(m.mse_resid)),
            "Predictors": len(names),
            "Variables": ", ".join(map(str, names)),
        })
    check_ok = abs(top[0]["Adjusted R Square"] - float(scores[order[0]])) < 1e-8
    return {"mode": mode, "models_run": len(keep), "skipped": skipped, "scores": scores,
            "top": top, "pool": list(cols), "max_size": max_size, "check_ok": check_ok}


def make_scorer(df, y, cols):
    """Returns score(indices) -> Adjusted R Square (or None if unfittable), with caching."""
    n = len(df)
    X = df[cols].to_numpy(dtype=float)
    sd = X.std(axis=0)
    sd[sd == 0] = 1.0  # a column that is constant in these rows becomes all zeros and is skipped as singular
    Z = (X - X.mean(axis=0)) / sd
    yc = df[y].to_numpy(dtype=float)
    yc = yc - yc.mean()
    G, c, sst = Z.T @ Z, Z.T @ yc, float(yc @ yc)
    cache = {}

    def score(idx):
        key = tuple(sorted(idx))
        if key in cache:
            return cache[key]
        k, val = len(key), None
        if k > 0 and n - k - 1 > 0:
            Gs = G[np.ix_(key, key)]
            if np.linalg.cond(Gs) <= 1e10:
                ck = c[list(key)]
                r2 = float(np.linalg.solve(Gs, ck) @ ck) / sst
                val = 1 - (1 - r2) * (n - 1) / (n - k - 1)
        cache[key] = val
        return val

    return score, cache


def hill_climb(score, k, start, max_size):
    """Steepest ascent: apply the best single add, remove, or swap until nothing improves."""
    cur = tuple(sorted(start))
    cur_s = score(cur)
    while True:
        out = [i for i in range(k) if i not in cur]
        nbrs = []
        if len(cur) < max_size:
            nbrs += [cur + (i,) for i in out]
        if len(cur) > 1:
            nbrs += [tuple(x for x in cur if x != j) for j in cur]
        nbrs += [tuple(i if x == j else x for x in cur) for j in cur for i in out]
        best, best_s = None, cur_s
        for nb in nbrs:
            s = score(nb)
            if s is not None and s > best_s + 1e-9:
                best, best_s = tuple(sorted(nb)), s
        if best is None:
            return cur, cur_s
        cur, cur_s = best, best_s


def run_hybrid(df, y, cols, max_size, n_random, n_starts, must=(), progress=None):
    """Random scouting, then hill-climb from the baseline, the full pool, and the best scouts."""
    k = len(cols)
    score, cache = make_scorer(df, y, cols)
    rng = np.random.default_rng(SEARCH_SEED)
    scouts = set()
    for _ in range(n_random):
        size = int(rng.integers(1, max_size + 1))
        scouts.add(tuple(sorted(rng.choice(k, size=size, replace=False).tolist())))
    ranked = sorted((score(s), s) for s in scouts if score(s) is not None)[::-1]

    starts = [("Baseline", tuple(sorted(m))) for m in must]
    if k <= max_size:
        starts.append(("Full pool", tuple(range(k))))
    starts += [(f"Random #{i}", s) for i, (_, s) in enumerate(ranked[:n_starts], 1)]

    rows, ends = [], set()
    for i, (label, s0) in enumerate(starts):
        s_start = score(s0)
        if s_start is None:
            continue
        end, s_end = hill_climb(score, k, s0, max_size)
        ends.add(end)
        rows.append({"Start": label, "Start Adj R Square": s_start, "End Adj R Square": s_end,
                     "Predictors": len(end), "Variables": ", ".join(str(cols[j]) for j in end)})
        if progress is not None:
            progress.progress((i + 1) / len(starts))

    valid = {key: v for key, v in cache.items() if v is not None}
    if not valid:
        raise ValueError("No valid models could be fitted from this pool.")
    order = sorted(valid, key=valid.get, reverse=True)
    top = []
    for rank, key in enumerate(order[:TOP_N], start=1):
        names = [cols[i] for i in key]
        m = run_regression(df, y, names)
        top.append({"Rank": rank, "Adjusted R Square": float(m.rsquared_adj),
                    "R Square": float(m.rsquared), "Standard Error": float(np.sqrt(m.mse_resid)),
                    "Predictors": len(names), "Variables": ", ".join(map(str, names))})
    check_ok = abs(top[0]["Adjusted R Square"] - valid[order[0]]) < 1e-8
    return {"mode": "hybrid", "models_run": len(valid), "skipped": len(cache) - len(valid),
            "scores": np.array(list(valid.values())), "top": top, "pool": list(cols),
            "max_size": max_size, "check_ok": check_ok, "starts": rows, "distinct_ends": len(ends)}


def store_search(res, pick, max_size, recipe):
    """Save a search result to session state and the recipe."""
    st.session_state.search = res
    s = res["scores"]
    recipe["models_searched"] = res["models_run"]
    recipe["search_mode"] = res["mode"]
    recipe["search_pool"] = pick
    recipe["search_max_size"] = max_size
    recipe["search_skipped"] = res["skipped"]
    recipe["search_benchmarks"] = {"best": float(s.max()), "average": float(s.mean()),
                                   "median": float(np.median(s)), "worst": float(s.min())}
    recipe["leaderboard"] = res["top"]
    if "starts" in res:
        recipe["search_starts"] = res["starts"]
    recipe["holdout"] = res.get("holdout")


def search_fig(scores, base_adj):
    fig, ax = plt.subplots(figsize=(6, 3.2))
    ax.hist(scores, bins=40, color="#4C78A8", edgecolor="#16213A")
    ax.axvline(base_adj, color="#E45756", linestyle="--", label="Baseline")
    ax.axvline(scores.max(), color="#54A24B", label="Best model")
    ax.set_xlabel("Adjusted R Square")
    ax.set_ylabel("Number of models")
    ax.set_title("Adjusted R Square across all tested models")
    ax.legend(fontsize=7)
    fig.tight_layout()
    return fig



def show_stage_5():
    result = st.session_state.clean
    if result is None or "feature_table" not in result:
        st.info("Approve the feature plan in Stage 4 first.")
        return
    work, recipe = result["feature_table"], result["recipe"]
    y = recipe["target"]
    all_pool = recipe["pool_columns"]

    ui.section("Pool & Search", "The app fits many models from different subsets of your pool and ranks them by Adjusted R Square. Python does the search; you decide what the results mean.")
    strip = st.container()  # filled in below, once the controls are set

    # --- Pool ---
    pick = st.multiselect("Variables in the search pool (roughly 10 to 20 works best)", all_pool,
                          default=all_pool, key=f"pool_pick_{st.session_state.get('pool_version', 0)}")
    k = len(pick)
    if k == 0:
        st.warning("Pick at least one variable.")
        return

    # --- Holdout ---
    df_clean, raw = result["df"], result["raw"]
    base_cols = recipe["base_model_columns"]
    entity_col = None
    with st.container(border=True):
        st.markdown("**Holdout check**")
        st.caption(f"{HOLDOUT_FRAC:.0%} of rows are set aside before searching. The search only sees the other {1 - HOLDOUT_FRAC:.0%}, so the holdout gives an honest test of the winner.")
        ents = entity_candidates(raw)
        by_entity = st.checkbox("Hold out whole entities instead of random rows", key="holdout_entities", disabled=not ents)
        if ents:
            st.caption("Recommended when the same entity repeats across rows (panel data, such as a country observed every year). A random split would put the same entity on both sides and make the holdout look better than it should.")
            if by_entity:
                entity_col = st.selectbox("Column that identifies the entity", ents, key=f"holdout_col_{st.session_state.get('pool_version', 0)}")
        else:
            st.caption("Needs a text column with repeating values (for example Country). None was found, so random rows will be used.")
        mask = split_holdout(work.index, raw, entity_col)
        train, test = work[~mask], work[mask]
        st.caption(f"Holdout mode: {'whole entities' if entity_col else 'random rows'}. Spread of Y (SD): training {train[y].std():.3f}, holdout {test[y].std():.3f}. "
                   "If the holdout spread is larger, holdout R Square gets a boost that is not real skill.")
    base_adj = run_regression(df_clean.loc[train.index], y, base_cols).rsquared_adj

    # --- Search settings ---
    cap = min(k, len(train) - 2)
    max_size = st.slider("Largest model to test (number of predictors)", 1, cap, cap) if cap > 1 else 1
    total = count_subsets(k, max_size)
    sample_n = 0
    if total <= FULL_LIMIT:
        st.write(f"{total:,} possible models, so **every** one will be run.")
    else:
        st.write(f"{total:,} possible models, too many to run all, so a **random sample** will be tested.")
        sample_n = st.number_input("Sample size", 500, 50000, DEFAULT_SAMPLE, step=500)

    with strip:
        ui.metric_strip([
            ("Variables in pool", f"{k}", "Predictors the search can choose from."),
            ("Training rows", f"{len(train):,}", "The only rows the search sees."),
            ("Holdout rows", f"{len(test):,}", "Set aside to test the winner on unseen data."),
            ("Possible models", f"{total:,}",
             "Every subset up to the size limit." if total <= FULL_LIMIT else "Too many to run all, so a random sample is tested."),
        ])

    # --- Run buttons ---
    if st.button("Run search", type="primary"):
        bar = st.progress(0.0)
        try:
            base_in_pool = all(c in pick for c in base_cols)
            must = [[pick.index(c) for c in base_cols]] if base_in_pool else []
            res = run_search(train, y, pick, max_size, int(sample_n), bar, must)
            res["baseline_testable"] = base_in_pool
            attach_holdout(res, work, df_clean, mask, y, base_cols, entity_col)
            store_search(res, pick, max_size, recipe)
        except Exception as e:
            st.error(f"Search failed: {e}")
        bar.empty()

    with st.expander("Riff search (optional): scout with random models, then improve the best starting points one change at a time"):
        r1, r2 = st.columns(2)
        n_rand = r1.number_input("Random models to scout starting points", 200, 20000, 2000, step=200)
        n_starts = r2.number_input("Random starting points to riff from", 0, 50, 10)
        if st.button("Run riff search"):
            bar = st.progress(0.0)
            try:
                base_in_pool = all(c in pick for c in base_cols)
                must = [[pick.index(c) for c in base_cols]] if base_in_pool else []
                res = run_hybrid(train, y, pick, max_size, int(n_rand), int(n_starts), must, bar)
                res["baseline_testable"] = base_in_pool
                attach_holdout(res, work, df_clean, mask, y, base_cols, entity_col)
                store_search(res, pick, max_size, recipe)
            except Exception as e:
                st.error(f"Search failed: {e}")
            bar.empty()

    res = st.session_state.get("search")
    if res is None:
        return
    stale = res["pool"] != pick or res["max_size"] != max_size or (res.get("holdout") or {}).get("entity_column") != entity_col
    if stale:
        st.warning("The pool, model-size limit, or holdout setting changed since the last run. Run the search again.")

    # --- Results ---
    st.divider()
    s = res["scores"]
    how = {"all": "every possible model", "sample": "a random sample of models",
           "hybrid": "random scouting plus riffing from promising starts"}[res["mode"]]
    ui.section("Search results", f"Ran {res['models_run']:,} models ({how}); skipped {res['skipped']:,} that could not be fitted.")
    if not res.get("baseline_testable", True):
        st.info("Some Base Data Model predictors are not in this pool (dropped in Stage 4 or removed above), so no tested model reproduces the baseline exactly. The best model can legitimately score below it.")
    if not res["check_ok"]:
        st.error("The fast scorer disagreed with statsmodels for the top model. Do not trust this search; please report it.")

    ui.metric_strip([
        ("Best model", f"{s.max():.4f}", "Highest Adjusted R Square found.", f"{s.max() - base_adj:+.4f} vs baseline"),
        ("Average", f"{s.mean():.4f}", "Mean Adjusted R Square of the models tested."),
        ("Worst", f"{s.min():.4f}", "Lowest Adjusted R Square of the models tested."),
        ("Baseline", f"{base_adj:.4f}", "Your Base Data Model, fit on the training rows."),
    ])
    fig = search_fig(s, base_adj)
    st.pyplot(fig)
    plt.close(fig)
    how_to_read("search")

    # --- Leaderboard ---
    st.divider()
    ui.section("Leaderboard", f"Top {len(res['top'])} models by Adjusted R Square.")
    lb = pd.DataFrame(res["top"])
    lb = lb[[c for c in lb.columns if c != "Variables"] + ["Variables"]]
    fmt = {c: "{:.4f}" for c in ["Adjusted R Square", "R Square", "Holdout R Square", "Shrinkage"] if c in lb}
    fmt.update({c: "{:.3f}" for c in ["Standard Error", "Holdout Error (RMSE)", "Error Ratio"] if c in lb})
    st.dataframe(lb.style.format(fmt), use_container_width=True, hide_index=True)
    st.caption("Winner's curse: the best of thousands of models looks better than it will on new data, partly by luck. Treat the winner as a strong candidate, not a proven result. Many of the top models differ by tiny amounts, so look at the whole top group, not just rank 1.")

    with st.expander("How often does each variable appear in the top models?"):
        names = [n for r in res["top"] for n in r["Variables"].split(", ")]
        freq = pd.Series(names).value_counts()
        freq_df = pd.DataFrame({"Models containing it": freq,
                                "Share of top models": freq / len(res["top"]) * 100})
        st.dataframe(freq_df.style.format({"Models containing it": "{:,.0f}", "Share of top models": "{:.0f}%"})
                     .background_gradient(subset=["Share of top models"], cmap="Oranges"),
                     use_container_width=True)
        st.caption(f"Counts only the {len(res['top'])} leaderboard models. A variable in nearly all of them is robustly useful. One that appears in only a few may owe its place to luck.")

    # --- Holdout check ---
    h = res.get("holdout")
    top1 = res["top"][0]
    if h and "Holdout R Square" in top1:
        st.divider()
        ui.section("Holdout check: winner vs baseline",
                   f"Holdout mode: {h['mode']}. {h['n_train']:,} training rows, {h['n_holdout']:,} held out. Baseline and models are all fit on the training rows only.")
        items = [
            ("Winner, training Adj R Square", f"{top1['Adjusted R Square']:.4f}", "How well the winner fits the rows it was searched on."),
            ("Winner, holdout R Square", f"{top1['Holdout R Square']:.4f}", "How well the winner predicts rows it never saw."),
            ("Winner shrinkage", f"{top1['Shrinkage']:+.4f}", "Training Adjusted R Square minus holdout R Square. Bigger means the winner looked better than it really is."),
        ]
        if h["baseline"]:
            items.append(("Baseline holdout R Square", f"{h['baseline']['holdout_r2']:.4f}",
                          "The same test for your Base Data Model.",
                          f"shrinkage {h['baseline']['train_adj'] - h['baseline']['holdout_r2']:+.4f}"))
        ui.metric_strip(items)
        how_to_read("holdout")

    # --- Riff starts ---
    if res.get("starts"):
        st.divider()
        ui.section("Where each starting point ended up")
        st.dataframe(pd.DataFrame(res["starts"]).style.format(
            {"Start Adj R Square": "{:.4f}", "End Adj R Square": "{:.4f}"}),
            use_container_width=True, hide_index=True)
        st.caption(f"{len(res['starts'])} starts reached {res['distinct_ends']} distinct final model(s). "
                   "One distinct model means riffing alone finds the peak. Several means there are multiple peaks, so add more random starting points. "
                   "Note: Average and Worst now describe the models explored, not a random draw, because riffing mostly visits good models.")

    st.divider()
    notes_box("Analyst notes (which models look sensible to you, and why)", "notes_stage5")
    st.button("Approve search results →", on_click=approve_stage, args=(4,), disabled=stale, type="primary")
















# ---------- Excel export ----------
ROW_CAP = 50000  # data sheets write at most this many rows
XLSX_MIME = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"


def xl_val(v):
    """Convert numpy/NaN/list values into something openpyxl can store."""
    if isinstance(v, np.integer):
        return int(v)
    if isinstance(v, np.bool_):
        return bool(v)
    if isinstance(v, np.floating):
        v = float(v)
    if isinstance(v, float) and not np.isfinite(v):
        return None
    if isinstance(v, (list, dict, tuple, set)):
        return str(v)
    return v


def write_df(ws, data, row, index=True, max_rows=None):
    """Write a DataFrame with a bold header starting at `row`. Returns the next free row."""
    d = data.reset_index() if index else data.copy()
    if index:
        d.columns = [""] + [str(c) for c in data.columns]
    if max_rows is not None:
        d = d.head(max_rows)
    for j, c in enumerate(d.columns, start=1):
        ws.cell(row=row, column=j, value=(str(c) or None)).font = Font(bold=True)
    for i, rec in enumerate(d.itertuples(index=False, name=None), start=row + 1):
        for j, v in enumerate(rec, start=1):
            ws.cell(row=i, column=j, value=xl_val(v))
    return row + len(d) + 2


def put_title(ws, row, text):
    ws.cell(row=row, column=1, value=text).font = Font(bold=True, size=12)
    return row + 1


def put_lines(ws, row, lines, wrap=False):
    """Write each non-empty line in column A (blank lines just skip a row). Returns the next free row."""
    for line in lines:
        if str(line).strip():
            c = ws.cell(row=row, column=1, value=str(line))
            if wrap:
                c.alignment = Alignment(wrap_text=True, vertical="top")
        row += 1
    return row + 1


def write_data_sheet(wb, name, data, notes):
    """A sheet of raw rows with notes on top and a visible truncation note if rows were cut."""
    ws = wb.create_sheet(name)
    notes = list(notes)
    if len(data) > ROW_CAP:
        notes.insert(0, f"TRUNCATED: only the first {ROW_CAP:,} of {len(data):,} rows are shown on this sheet.")
    r = put_lines(ws, 1, notes)
    write_df(ws, data, r, index=False, max_rows=ROW_CAP)


def fig_png(fig, dpi=100):
    """Turn a matplotlib figure into PNG bytes plus its pixel width and height."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi)
    w, h = fig.get_size_inches() * dpi
    plt.close(fig)
    return buf.getvalue(), int(w), int(h)


def chart_items(result, recipe, win_m, base_adj, scores):
    """List of (title, png_bytes, width_px, height_px). A chart that fails is skipped."""
    df, raw, y = result["df"], result["raw"], recipe["target"]
    x_cols = [c for c in df.columns if c != y]
    items = []

    def add(title, make, dpi=100):
        try:
            with plt.style.context("default"):  # Excel images stay on a white background
                items.append((title,) + fig_png(make(), dpi))
        except Exception:
            plt.close("all")

    if scores is not None:
        add("Search: Adjusted R Square across tested models", lambda: search_fig(scores, base_adj))
    add("Selected model: Q-Q plot of residuals", lambda: qq_fig(win_m.resid))
    add("Selected model: residuals vs fitted", lambda: resid_fitted_fig(win_m.fittedvalues, win_m.resid))
    add("Selected model: histogram of residuals", lambda: hist_fig(win_m.resid, "Histogram of residuals"))
    add(f"Distribution of {y}", lambda: hist_fig(df[y], f"Distribution of {y}"))

    corr = df.corr()
    top = corr[y].drop(y).abs().sort_values(ascending=False).index[:4]
    for c in top:
        add(f"{c} vs {y}", lambda c=c: scatter_fig(df, c, y))
    add("Correlation heatmap", lambda: heatmap_fig(corr), dpi=70)

    cat = result["cat_original"]
    for c in cat.columns:
        add(f"{y} by {c}", lambda c=c: box_fig(cat[c], df[y], c, y))
        add(f"Count by {c}", lambda c=c: bar_fig(freq_table(cat[c]), c))

    for label in recipe.get("splits_explored", [])[:5]:
        median = label.endswith(" (median split)")
        col = label.removesuffix(" (median split)").removesuffix(" (each value)")
        try:
            groups = make_groups(raw, col, median)
            if groups.nunique() < 2:
                continue
            _, cg = segment_tables(df, y, x_cols, groups)
            binary = [c for c in x_cols if df[c].nunique() <= 2]
            strength = cg.drop(columns="Gap").abs().max(axis=1)
            picks = [c for c in cg["Gap"].dropna().index if c not in binary and strength[c] >= 0.2][:2]
        except Exception:
            continue
        for c in picks:
            add(f"Split by {label}: {c} vs {y}", lambda c=c, g=groups: overlay_fig(df, c, y, g))
    return items


def build_workbook(result, recipe, base_m, win_m, mask, win_cols, report_text, search_scores=None):
    """Tier 1 export: six sheets, static values. Returns the .xlsx file as bytes."""
    df, work, raw, y = result["df"], result["feature_table"], result["raw"], recipe["target"]
    x_cols = [c for c in df.columns if c != y]
    fm = recipe["final_model"]
    wb = Workbook()

    # ---- 1. AI Report ----
    ws = wb.active
    ws.title = "AI Report"
    ws.column_dimensions["A"].width = 100
    r = put_title(ws, 1, "AI-Guided Regression Report")
    r = put_lines(ws, r, [
        "All numbers in this workbook are static values, not live Excel formulas.",
        f"Target (Y): {y}",
        f"Selected model: leaderboard rank {fm['rank']}, {len(win_cols)} predictors: " + ", ".join(map(str, win_cols)),
        "Model statistics are fit on the training rows only; holdout rows are used just for out-of-sample checks.",
        f"Models searched: {recipe.get('models_searched', 'n/a'):,} ({recipe.get('search_mode', 'n/a')} search)"
        if isinstance(recipe.get("models_searched"), int) else "Models searched: n/a",
        f"Splits explored in the Segment Explorer: {len(recipe.get('splits_explored', []))}"
        + (" (" + "; ".join(recipe["splits_explored"]) + ")" if recipe.get("splits_explored") else ""),
    ], wrap=True)
    r = put_title(ws, r, "Claude's final report")
    if report_text:
        r = put_lines(ws, r, report_text.replace("**", "").split("\n"), wrap=True)
    else:
        r = put_lines(ws, r, ["No final report was written for this model rank. Click 'Write final report' in Stage 6, then rebuild the workbook."])
    notes = {k: v for k, v in recipe.get("analyst_notes", {}).items() if v.strip()}
    r = put_title(ws, r, "Analyst notes")
    r = put_lines(ws, r, [f"{k.replace('notes_stage', 'Stage ')}: {v}" for k, v in notes.items()] or ["(none written)"], wrap=True)
    r = put_title(ws, r, "Changes from the base model")
    ch = pd.DataFrame(fm["changes_from_base"])
    if ch.empty:
        r = put_lines(ws, r, ["The selected model uses exactly the base model's variables."])
    else:
        r = write_df(ws, ch, r, index=False)
    r = put_title(ws, r, "Limitations and next steps (flagged by the automatic checks)")
    gaps = [g for g in KNOWN_GAPS if not g.startswith("Panel data") or recipe.get("dropped_high_cardinality")]
    r = put_lines(ws, r, [t.replace("**", "") for t in fm["health_notes"]] or ["No major issues flagged by the automatic checks."], wrap=True)
    r = put_title(ws, r, "Known gaps in this app")
    put_lines(ws, r, gaps, wrap=True)

    # ---- 2. ToolPak Replication ----
    ws = wb.create_sheet("ToolPak Replication")
    ws.column_dimensions["A"].width = 24
    r = put_lines(ws, 1, [
        "Excel Data Analysis ToolPak layout (Regression Statistics, ANOVA, Coefficients). Static values.",
        "Both models are fit on the training rows only. To reproduce the selected model in Excel, run the ToolPak on the Train rows of the 'Model Data' sheet.",
    ])
    for label, m in (("Base model", base_m), ("Selected model", win_m)):
        reg, anova, coef = toolpak_tables(m)
        r = put_title(ws, r, f"{label}: Regression Statistics")
        r = write_df(ws, reg, r)
        r = put_title(ws, r, f"{label}: ANOVA")
        r = write_df(ws, anova, r)
        r = put_title(ws, r, f"{label}: Coefficients")
        r = write_df(ws, coef, r)

    # ---- 3. Leaderboard ----
    ws = wb.create_sheet("Leaderboard")
    r = put_lines(ws, 1, [
        f"Top {len(recipe['leaderboard'])} models by Adjusted R Square (fit on training rows).",
        "Holdout columns report out-of-sample performance. Do not re-pick the winner by holdout score.",
    ])
    write_df(ws, pd.DataFrame(recipe["leaderboard"]), r, index=False)

    # ---- 4. Data Picture ----
    ws = wb.create_sheet("Data Picture")
    r = put_lines(ws, 1, ["Statistics below use all rows of the cleaned data. Static values."])
    r = put_title(ws, r, "Descriptive statistics (Excel layout)")
    r = write_df(ws, descriptive_stats(df), r)
    r = put_title(ws, r, "Correlation matrix (Pearson)")
    r = write_df(ws, df.corr(), r)
    for label in recipe.get("splits_explored", []):
        median = label.endswith(" (median split)")
        col = label.removesuffix(" (median split)").removesuffix(" (each value)")
        try:
            groups = make_groups(raw, col, median)
            if groups.nunique() < 2:
                continue
            summ, corr_g = segment_tables(df, y, x_cols, groups)
        except Exception:
            continue
        r = put_title(ws, r, f"Segment comparison: {label} (group sizes and Y)")
        r = write_df(ws, summ, r)
        r = put_title(ws, r, f"Segment comparison: {label} (correlation with Y by group, and the gap)")
        r = write_df(ws, corr_g, r)

    # ---- 5. Model Data (winner's columns, Train rows first) ----
    md = work[[y] + list(win_cols)].copy()
    md["Set"] = np.where(mask, "Holdout", "Train")
    md = pd.concat([md[~mask], md[mask]])
    write_data_sheet(wb, "Model Data", md, [
        "The selected model's columns (Y first, then predictors) plus a Set column.",
        f"The {int((~mask).sum()):,} Train rows come first, then the Holdout rows. Run the ToolPak on the Train rows only: it reproduces the Stage 6 numbers.",
        "Engineered columns are included as built in Stage 4.",
    ])

    # ---- 6. Cleaned Data ----
    write_data_sheet(wb, "Cleaned Data", df, [
        "The table after cleaning (text columns one-hot encoded, missing predictor values filled with medians).",
        "Engineered columns from Stage 4 are not included here; see 'Model Data' for the selected model's columns.",
    ])


    # ---- 7. Charts (images drawn by Python) ----
    ws = wb.create_sheet("Charts")
    put_lines(ws, 1, ["Charts drawn by Python (matplotlib) from the same data. These are pictures, not live Excel charts."])
    base_adj = float(base_m.rsquared_adj)
    r, side, row_h = 4, 0, 0
    for title, png, w, h in chart_items(result, recipe, win_m, base_adj, search_scores):
        wide = w > 600
        if wide and side == 1:  # a wide chart starts on a fresh row
            r, side, row_h = r + row_h, 0, 0
        col = 1 if side == 0 else 10
        ws.cell(row=r, column=col, value=title).font = Font(bold=True)
        img = XLImage(io.BytesIO(png))
        img.anchor = f"{'A' if side == 0 else 'J'}{r + 1}"
        ws.add_image(img)
        need = int(h / 20) + 3  # default row height is about 20 pixels
        if wide:
            r, side, row_h = r + need, 0, 0
        elif side == 0:
            side, row_h = 1, need
        else:
            r, side, row_h = r + max(row_h, need), 0, 0
    wb.move_sheet("Charts", offset=-2)  # sits after Data Picture, before the big data sheets

    buf = io.BytesIO()
    wb.save(buf)
    return buf.getvalue()















# ---------- Stage 6: Results ----------
NEAR_TIE = 0.001  # a model within this much Adjusted R Square of the selected one counts as a near-tie


def model_metrics(train, test, y, names):
    """Fit on training rows; return the model plus training and holdout numbers."""
    m = run_regression(train, y, names)
    h = fit_and_score(train, test, y, names)
    se = float(np.sqrt(m.mse_resid))
    return m, {"Adjusted R Square": float(m.rsquared_adj), "R Square": float(m.rsquared),
               "Standard Error": se, "Predictors": len(names),
               "Holdout R Square": h["holdout_r2"], "Holdout Error (RMSE)": h["holdout_rmse"],
               "Error Ratio": h["holdout_rmse"] / se}


def compare_table(base_m, win_m, base_cols, win_cols):
    """Every variable in either model, marked Kept, Added, or Removed."""
    rows = []
    for c in dict.fromkeys(list(base_cols) + list(win_cols)):
        inb, inw = c in base_cols, c in win_cols
        rows.append({"Variable": str(c), "Status": "Kept" if inb and inw else ("Added" if inw else "Removed"),
                     "Base coef": base_m.params.get(c, np.nan), "Base p": base_m.pvalues.get(c, np.nan),
                     "Best coef": win_m.params.get(c, np.nan), "Best p": win_m.pvalues.get(c, np.nan)})
    t = pd.DataFrame(rows)
    order = {"Kept": 0, "Added": 1, "Removed": 2}
    return t.sort_values("Status", key=lambda s: s.map(order), kind="stable").reset_index(drop=True)


def engineered_impact(train, test, y, win_cols, engineered, win_adj, win_hold):
    """For each engineered column in the model, refit without it and measure the drop."""
    out = []
    for c in win_cols:
        rest = [x for x in win_cols if x != c]
        if c not in engineered or not rest:
            continue
        h = fit_and_score(train, test, y, rest)
        out.append({"Engineered variable": str(c),
                    "Adj R Square without it": h["train_adj"],
                    "Drop in Adj R Square": win_adj - h["train_adj"],
                    "Holdout R Square without it": h["holdout_r2"],
                    "Drop in Holdout R Square": win_hold - h["holdout_r2"]})
    return pd.DataFrame(out)


def step_name(step):
    """The column name a plan step creates (same naming rules as build_feature). None for drop."""
    op, cols, how = step["op"], step["cols"], step.get("how")
    if op == "log":
        return f"ln1p_{cols[0]}"
    if op == "square":
        return f"{cols[0]}_sq"
    if op == "interaction":
        return f"{cols[0]}_x_{cols[1]}"
    if op == "combine":
        a, b = cols[0], cols[1]
        return {"sum": f"sum_{a}_{b}", "mean": f"mean_{a}_{b}",
                "difference": f"{a}_minus_{b}", "ratio": f"{a}_per_{b}"}.get(how)
    return None


def step_recipe(step):
    """Plain-English formula for a plan step."""
    op, c, how = step["op"], step["cols"], step.get("how")
    if op == "log":
        return f"ln(1 + {c[0]})"
    if op == "square":
        return f"{c[0]} squared"
    if op == "interaction":
        return f"{c[0]} × {c[1]}"
    if op == "combine":
        return {"sum": f"{c[0]} + {c[1]}", "mean": f"({c[0]} + {c[1]}) / 2",
                "difference": f"{c[0]} − {c[1]}", "ratio": f"{c[0]} ÷ {c[1]}"}.get(how, "")
    return ""


def changes_table(plan, base_cols, win_cols):
    """Every difference between the base model and the selected model, with how and why."""
    made = {step_name(s): s for s in plan if step_name(s)}
    dropped = {c: s for s in plan if s["op"] == "drop" for c in s["cols"]}
    rows = []
    for c in win_cols:
        if c in base_cols:
            continue
        if c in made:
            s = made[c]
            rows.append({"Variable": str(c), "Change": f"Added: {s['op']}" + (f" ({s['how']})" if s.get("how") else ""),
                         "Formula": step_recipe(s), "Reason": s.get("reason", "")})
        else:
            rows.append({"Variable": str(c), "Change": "Added: original column not in base model", "Formula": "-", "Reason": ""})
    for c in base_cols:
        if c in win_cols:
            continue
        if c in dropped:
            rows.append({"Variable": str(c), "Change": "Removed: dropped in Stage 4 plan", "Formula": "-",
                         "Reason": dropped[c].get("reason", "")})
        else:
            rows.append({"Variable": str(c), "Change": "Removed: left out by the search", "Formula": "-",
                         "Reason": "Search found a higher Adjusted R Square without it"})
    return pd.DataFrame(rows, columns=["Variable", "Change", "Formula", "Reason"])


KNOWN_GAPS = [
    "Panel data is flagged but not corrected (no clustered standard errors).",
    "No robust standard errors for unequal spread.",
    "Linear OLS only; curvature is handled only through log and square terms.",
    "The holdout is one split, so its score is itself noisy.",
    "Median imputation was used for missing predictor values.",
    "Post-selection inference: the selected model was chosen from many tested models, so its p-values and confidence intervals are optimistic; the holdout results are the more honest check."
]

REPORT_SYSTEM = (
    "You are a statistics tutor writing the final report for an introductory college statistics student and professor. "
    "You receive a compact JSON summary of a regression analysis: a base model, a search over many variable subsets, "
    "a selected model, and a holdout check. Write a plain-English report with these short sections: "
    "1) Summary: what was predicted, the base model vs the selected model (Adjusted R Square, standard error, holdout R Square, error ratio). "
    "2) Exact changes from the base model: list every added and removed variable with its formula and reason, so anyone can reproduce it. "
    "3) How much to trust it: how many models were searched (state the number), winner's curse, shrinkage on the holdout, "
    "and whether near-tied runners-up are simpler. "
    "4) What the analyst noticed: weave in the analyst notes, the segments explored (state how many) and any segment findings, "
    "and say group differences found by trying many splits are hypotheses. "
    "5) Limitations and next steps: for each item in health_notes, explain in a sentence why it matters, then give 2 to 3 starter steps "
    "to fix or check it OUTSIDE this app (for example clustered standard errors for panel data, robust standard errors for unequal spread, "
    "inspecting high Cook's distance rows, log or squared terms or splines for curvature, dropping or combining collinear variables). "
    "Also cover known_app_gaps. "
    "6) Pro takeaways: 3 bullets. "
    "Define any statistical term in a few words the first time you use it. "
    "Use only the numbers provided, do not invent facts, and do not choose for the analyst."
)


def build_report_profile(recipe):
    """Compact JSON for the final report. No raw rows."""
    fm, h = recipe["final_model"], recipe.get("holdout") or {}
    prof = {
        "target": recipe["target"],
        "base_model_columns": recipe["base_model_columns"],
        "baseline_from_stage3": recipe.get("baseline"),
        "selected_model": {k: fm[k] for k in (
            "rank", "variables", "metrics_base", "metrics_best", "changes_from_base",
            "engineered_impact", "runners_up", "line_diagnostics", "health_notes")},
        "search": {
            "models_searched": recipe.get("models_searched"), "mode": recipe.get("search_mode"),
            "pool": recipe.get("search_pool"), "max_size": recipe.get("search_max_size"),
            "skipped": recipe.get("search_skipped"), "benchmarks": recipe.get("search_benchmarks"),
        },
        "holdout": {k: h.get(k) for k in ("mode", "entity_column", "fraction", "n_train", "n_holdout", "baseline")},
        "splits_explored": recipe.get("splits_explored", []),
        "segment_findings": recipe.get("segment_findings", {}),
        "analyst_notes": {k: v for k, v in recipe.get("analyst_notes", {}).items() if v.strip()},
        "baseline_claude_read": recipe.get("baseline_claude_read", ""),
        "stage3_health_notes": recipe.get("health_notes", []),
        "known_app_gaps": [g for g in KNOWN_GAPS
                           if not g.startswith("Panel data") or recipe.get("dropped_high_cardinality")],
    }
    return json.loads(json.dumps(prof, default=str))


def actual_pred_fig(actual, pred, title, lim):
    """Actual vs predicted. Dots on the dashed line are perfect predictions."""
    fig, ax = plt.subplots(figsize=(5, 3.6))
    ax.scatter(pred, actual, s=8, alpha=0.4, color=ui.ACCENT)
    ax.plot(lim, lim, color=ui.HIGHLIGHT, linestyle="--", linewidth=1)
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title)
    fig.tight_layout()
    return fig


def show_stage_6():
    result = st.session_state.clean
    if result is None or "feature_table" not in result or not result["recipe"].get("leaderboard"):
        st.info("Run and approve a search in Stage 5 first.")
        return
    work, df, raw, recipe = result["feature_table"], result["df"], result["raw"], result["recipe"]
    y, top, h = recipe["target"], recipe["leaderboard"], recipe.get("holdout")
    if not h:
        st.info("No holdout found. Re-run the search in Stage 5.")
        return
    mask = split_holdout(work.index, raw, h["entity_column"])
    train, test = work[~mask], work[mask]
    base_cols = recipe["base_model_columns"]

    ui.section("Results", f"Everything is fit on the {len(train):,} training rows, so it matches the Stage 5 leaderboard. "
                          f"The {len(test):,} holdout rows ({h['mode']}) are used only to check performance on unseen data.")
    rank = st.selectbox("Model to examine (rank 1 is the search winner)", [r["Rank"] for r in top],
                        key=f"final_rank_{st.session_state.get('pool_version', 0)}")
    sel = next(r for r in top if r["Rank"] == rank)
    win_cols = sel["Variables"].split(", ")

    base_m, bm = model_metrics(df.loc[train.index], df.loc[test.index], y, base_cols)
    win_m, wm = model_metrics(train, test, y, win_cols)

    # ---- Verdict + headline strip ----
    er = wm["Error Ratio"]
    msg = (f"Holdout error is {er:.2f}× the training error. Holdout R Square is {wm['Holdout R Square']:.3f} "
           f"(base model: {bm['Holdout R Square']:.3f}).")
    if er <= 1.15:
        st.success("**Holds up on unseen rows.** " + msg)
    elif er < 1.3:
        st.info("**Some loss on unseen rows.** " + msg)
    else:
        st.warning("**Overstates itself on unseen rows.** " + msg)
    ui.metric_strip([
        ("Adjusted R Square", f"{wm['Adjusted R Square']:.3f}",
         f"Training fit of the selected model. Base model: {bm['Adjusted R Square']:.4f}.",
         f"{wm['Adjusted R Square'] - bm['Adjusted R Square']:+.4f} vs base"),
        ("Holdout R Square", f"{wm['Holdout R Square']:.3f}",
         f"How well the model predicts rows it never saw. Base model: {bm['Holdout R Square']:.4f}.",
         f"{wm['Holdout R Square'] - bm['Holdout R Square']:+.4f} vs base"),
        ("Error ratio", f"{er:.2f}",
         "Holdout error divided by training error. Near 1 means the model holds up; 1.3 or more means it overstated itself. "
         "If the holdout rows are more spread out than the training rows, this ratio rises for reasons that are not the model's fault."),
        ("Predictors", f"{wm['Predictors']}", f"The base model used {bm['Predictors']}."),
    ])
    how_to_read("holdout")

    # ---- Actual vs predicted on the holdout rows ----
    st.divider()
    ui.section("How does it do on unseen rows?", "Each dot is one holdout row. Dots on the dashed line were predicted perfectly.")
    base_test = df.loc[test.index]
    pred_b = base_m.predict(sm.add_constant(base_test[base_cols].astype(float), has_constant="add"))
    pred_w = win_m.predict(sm.add_constant(test[win_cols].astype(float), has_constant="add"))
    lo = float(min(test[y].min(), pred_b.min(), pred_w.min()))
    hi = float(max(test[y].max(), pred_b.max(), pred_w.max()))
    c1, c2 = st.columns(2)
    with c1:
        fig = actual_pred_fig(test[y], pred_b, "Base model", (lo, hi))
        st.pyplot(fig)
        plt.close(fig)
    with c2:
        fig = actual_pred_fig(test[y], pred_w, f"Selected model (rank {int(rank)})", (lo, hi))
        st.pyplot(fig)
        plt.close(fig)
    st.caption("Both charts use the same axes. A tighter cloud around the dashed line means better predictions. "
               "A cloud that bends away from the line at the ends means the model is biased there.")

    # ---- Changes from base ----
    st.divider()
    comp = compare_table(base_m, win_m, base_cols, win_cols)
    plan = recipe.get("feature_plan", [])
    changes = changes_table(plan, base_cols, win_cols)
    ui.section("What changed from the base model?", "The recipe for reproducing the selected model.")
    if changes.empty:
        st.info("The selected model uses exactly the base model's variables.")
    else:
        st.dataframe(changes, use_container_width=True, hide_index=True)
        st.caption("Start from the cleaned data, build each Added column with its formula, drop each Removed one, then fit.")
    unused = [step_name(s) for s in plan if step_name(s) and step_name(s) not in win_cols]
    if unused:
        st.caption("Built in Stage 4 but not used by this model: " + ", ".join(map(str, unused)))
    n_models = recipe.get("models_searched")
    n_txt = f"{n_models:,} models" if isinstance(n_models, int) else "many models"
    st.info(f"**Read the p-values with care.** This model was picked out of {n_txt}, so its p-values and confidence intervals look more certain than they are, "
            "because choosing the best-looking model builds in some luck. The holdout results above are the more honest check.")
    reference_caption(result, win_cols)
    show_influential_rows(win_m, raw)
    with st.expander("Base vs best: full numbers and coefficients"):
        fmt_rows = {}
        for k in bm:
            fmt_rows[k] = [f"{bm[k]:,.0f}", f"{wm[k]:,.0f}"] if k == "Predictors" else [f"{bm[k]:.4f}", f"{wm[k]:.4f}"]
        st.dataframe(pd.DataFrame(fmt_rows, index=["Base", "Best"]).T, use_container_width=True)
        st.dataframe(comp.style.format({"Base coef": "{:.4f}", "Base p": "{:.4f}", "Best coef": "{:.4f}", "Best p": "{:.4f}"}, na_rep="-"),
                     use_container_width=True, hide_index=True)
        st.caption("Kept = in both models. Added = only in the best model (often engineered). Removed = in the base model but not the best. "
                   "Coefficients for engineered columns are in the engineered column's own units.")

    imp = engineered_impact(train, test, y, win_cols, recipe.get("engineered_columns", []),
                            wm["Adjusted R Square"], wm["Holdout R Square"])
    with st.expander("Impact of each engineered variable"):
        if imp.empty:
            st.info("This model uses no engineered variables.")
        else:
            st.dataframe(imp.style.format({c: "{:.4f}" for c in imp.columns if c != "Engineered variable"}),
                         use_container_width=True, hide_index=True)
            st.caption("Each row refits the model without that one variable. A bigger drop means the variable matters more. A drop near 0 means it adds little, so a simpler model may do just as well.")

    wset, rows = set(win_cols), []
    for r in [r for r in top if r["Rank"] != rank][:3]:
        s = set(r["Variables"].split(", "))
        gap = r["Adjusted R Square"] - sel["Adjusted R Square"]
        if gap > 0:
            verdict = "Scores higher than the selected model"
        elif gap >= -NEAR_TIE:
            verdict = "Near-tie and simpler: consider it" if r["Predictors"] < sel["Predictors"] else "Near-tie"
        else:
            verdict = "Clearly lower"
        rows.append({"Rank": r["Rank"], "Adj R Square vs selected": gap, "Predictors": r["Predictors"],
                     "Holdout R Square": r.get("Holdout R Square"), "Error Ratio": r.get("Error Ratio"),
                     "Only in this model": ", ".join(sorted(s - wset)) or "-",
                     "Only in selected": ", ".join(sorted(wset - s)) or "-", "Verdict": verdict})
    runners = pd.DataFrame(rows)
    with st.expander("Why not the other top models?"):
        if not runners.empty:
            st.dataframe(runners.style.format({"Adj R Square vs selected": "{:+.4f}", "Holdout R Square": "{:.4f}", "Error Ratio": "{:.3f}"}, na_rep="-"),
                         use_container_width=True, hide_index=True)
        st.caption(f"Near-tie means within {NEAR_TIE} Adjusted R Square. When models are nearly tied, the simpler one is usually the better choice. Holdout numbers are for reporting; do not re-pick by them.")

    # ---- LINE checks ----
    st.divider()
    diag = residual_diagnostics(win_m)
    notes = health_notes(win_m, win_cols, train, recipe, diag)
    ui.section("Do the model's assumptions hold?",
               f"{len(notes)} thing(s) worth a look in the selected model." if notes
               else "Automatic checks on the selected model.")
    ui.metric_strip([(k.replace("_", " ").capitalize(), f"{v:.3g}") for k, v in diag.items()])
    st.caption("**L**inearity: residuals vs fitted (and RESET). **I**ndependence: Durbin-Watson. **N**ormality: Q-Q plot, histogram, skew, kurtosis. **E**qual variance: Breusch-Pagan and any funnel shape.")
    ui.health_list(notes)
    with st.expander("Residual charts"):
        c1, c2, c3 = st.columns(3)
        with c1:
            fig = qq_fig(win_m.resid)
            st.pyplot(fig)
            plt.close(fig)
        with c2:
            fig = resid_fitted_fig(win_m.fittedvalues, win_m.resid)
            st.pyplot(fig)
            plt.close(fig)
        with c3:
            fig = hist_fig(win_m.resid, "Histogram of residuals")
            st.pyplot(fig)
            plt.close(fig)
        st.markdown(HOW_TO_READ["resid"])

    with st.expander("Excel ToolPak tables (selected model)"):
        reg, anova, coef = toolpak_tables(win_m)
        st.markdown("**Regression Statistics**")
        st.dataframe(reg.style.format("{:,.6f}"), use_container_width=True)
        st.markdown("**ANOVA**")
        st.dataframe(anova.style.format("{:,.6g}", na_rep=""), use_container_width=True)
        st.markdown("**Coefficients**")
        st.dataframe(coef.style.format("{:,.6g}"), use_container_width=True)

    # ---- Save for the report and export ----
    def records(t):
        return t.astype(object).where(t.notna(), None).to_dict("records")
    recipe["final_model"] = {
        "rank": int(rank), "variables": win_cols, "fit_on": "training rows",
        "metrics_base": bm, "metrics_best": wm,
        "kept": comp.loc[comp["Status"] == "Kept", "Variable"].tolist(),
        "added": comp.loc[comp["Status"] == "Added", "Variable"].tolist(),
        "removed": comp.loc[comp["Status"] == "Removed", "Variable"].tolist(),
        "base_vs_best": records(comp), "engineered_impact": records(imp),
        "changes_from_base": records(changes),
        "runners_up": records(runners), "line_diagnostics": {k: float(v) for k, v in diag.items()},
        "health_notes": notes,
    }

    st.divider()
    notes_box("Analyst notes (why you picked this model, what you would check next)", "notes_stage6")

    # ---- Claude's final report ----
    st.divider()
    ui.section("Claude's final report", "Uses the model selected above, your analyst notes, and the health notes. Switch MODEL_BRIEF to claude-sonnet-5-5 for the final run.")
    api_key = get_api_key()
    if not api_key:
        st.caption("Enter your API key in the sidebar to enable this.")
    profile = build_report_profile(recipe)
    if st.button("Write final report", disabled=not api_key):
        with st.spinner("Claude is writing the report..."):
            try:
                text = ask_claude_brief(profile, api_key, REPORT_SYSTEM, teach=False)
                st.session_state.final_report = {"rank": int(rank), "text": text}
                recipe["final_report"] = st.session_state.final_report
            except Exception as e:
                st.error(f"Claude call failed: {e}")
    fr = st.session_state.get("final_report")
    if fr and fr["rank"] == int(rank):
        with st.container(border=True):
            st.markdown(fr["text"])
    elif fr:
        st.caption(f"The saved report is for rank {fr['rank']}. Click the button to write one for rank {int(rank)}.")
    with st.expander("Exactly what Claude was sent"):
        st.json(profile)

    # ---- Excel export ----
    st.divider()
    ui.section("Excel export", "Seven sheets: AI Report, ToolPak Replication, Leaderboard, Data Picture, Charts, Model Data, Cleaned Data. "
                               "Numbers are static values, not live formulas. Write the final report first if you want it included, and rebuild after changing your notes.")
    if st.button("Build Excel workbook"):
        with st.spinner("Building the workbook (large files take a few seconds)..."):
            try:
                report_text = fr["text"] if fr and fr["rank"] == int(rank) else None
                st.session_state.excel_file = {
                    "rank": int(rank),
                    "bytes": build_workbook(result, recipe, base_m, win_m, mask, win_cols, report_text,
                                            (st.session_state.get("search") or {}).get("scores")),
                }
            except Exception as e:
                st.error(f"Excel export failed: {e}")
    xf = st.session_state.get("excel_file")
    if xf and xf["rank"] == int(rank):
        st.download_button("Download Excel workbook", xf["bytes"], file_name="regression_report.xlsx",
                           mime=XLSX_MIME, type="primary")
    elif xf:
        st.caption(f"The built workbook is for rank {xf['rank']}. Click the button to build one for rank {int(rank)}.")













# ---------- Main ----------
ui.apply_styles()
ui.apply_chart_theme()
ui.render_sidebar(STAGES, bool(get_api_key()))
ui.render_header()

# One screen function per stage, in the same order as STAGES.
STAGE_SCREENS = [show_stage_1, show_stage_2, show_stage_3, show_stage_4, show_stage_5, show_stage_6]
STAGE_SCREENS[STAGES.index(st.session_state.stage_nav)]()  # the () at the end runs it
