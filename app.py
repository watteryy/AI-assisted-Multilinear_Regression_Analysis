import os
import warnings
from itertools import combinations
import numpy as np
import pandas as pd
import streamlit as st
import statsmodels.api as sm
import matplotlib.pyplot as plt
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
    "1. Data Audit & Prep",
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


def clean_data(df, y_name, exclude, encode_high, audit=None):
    """Make the table regression-ready. Returns a dict with the clean table, report, and recipe.
    `audit` (optional) is the Data audit record: its auto-fixes and approved fixes lead the report."""
    report = list(audit["auto_fixes"]) + list(audit["applied"]) if audit else []
    recipe = {"target": y_name, "excluded_by_user": list(exclude)}
    if audit:
        recipe["audit"] = audit
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
    n = len(work)
    basis = np.empty((n, len(x_cols) + 1))  # orthonormal basis of the columns kept so far, starting with the intercept
    basis[:, 0] = 1 / np.sqrt(n)
    used = 1
    for c in x_cols:
        v = work[c].to_numpy(dtype=float)
        z = (v - v.mean()) / v.std()
        resid = z
        for _ in range(2):  # twice, for numerical stability
            resid = resid - basis[:, :used] @ (basis[:, :used].T @ resid)
        size = np.linalg.norm(resid)
        if size > 1e-8 * np.linalg.norm(z):  # adds a new direction, so it is not a combination of the kept columns
            basis[:, used] = resid / size
            used += 1
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


# ---------- Data audit (U2) ----------
# Prep makes the table runnable; the audit looks for errors in the data itself.
# Safe fixes happen on load (auto_fix). Everything else is flagged with evidence and only changes if the user approves.
MISSING_TOKENS = {"?", "n/a", "na", "n.a.", "none", "null", "nan", "-", "--", "missing", "unknown", "#n/a"}
SENTINELS = {-99999, -9999, -999, -99, 999, 9999, 99999}   # common "no data" codes; flagged only when set apart from the rest of the column
SUMMARY_WORDS = {"total", "world", "all", "average", "overall", "global"}
LEAK_R = 0.98            # |correlation| with Y above this looks like leakage
NUM_TEXT_SHARE = 0.9     # a text column converts to numbers when this share of its values parse


def parse_number_text(s):
    """Floats from text such as '$1,200' or '45%' (NaN where it does not parse)."""
    return pd.to_numeric(s.astype(str).str.replace(r"[\$,%\s]", "", regex=True), errors="coerce")


def _tokens(s):
    """True where a text cell is a 'missing' placeholder such as ? or N/A."""
    return s.notna() & s.astype(str).str.strip().str.lower().isin(MISSING_TOKENS)


def auto_fix(df):
    """Safe tidy-ups on load: trim spaces in names and text values, convert numbers stored as text. Returns (table, log)."""
    df, log = df.copy(), []
    renamed = {c: c.strip() for c in df.columns
               if isinstance(c, str) and c != c.strip() and c.strip() not in df.columns}  # skip if the trimmed name already exists
    if renamed:
        df = df.rename(columns=renamed)
        log.append(f"Auto-fix: trimmed spaces around {len(renamed)} column name(s).")
    for c in [c for c in df.columns if is_text(df[c])]:
        s = df[c]
        trimmed = s.where(s.isna(), s.astype(str).str.strip())
        if (trimmed != s)[s.notna()].any():
            df[c] = s = trimmed
            log.append(f"Auto-fix: trimmed leading and trailing spaces in the values of {c}.")
        present = s.notna()
        if not present.any():
            continue
        parsed, tok = parse_number_text(s), _tokens(s)
        if parsed.notna().any() and (parsed.notna() | tok)[present].mean() >= NUM_TEXT_SHARE:
            df[c] = parsed
            extra = f"; {int(tok.sum()):,} placeholder(s) became missing" if tok.any() else ""
            log.append(f"Auto-fix: {c} held numbers stored as text (symbols such as $ , % removed), converted to numeric{extra}.")
    return df, log


def audit_data(df, y_name):
    """List of findings, most severe first. Each has evidence, a plain 'why', optional rows, and an optional fix."""
    out = []

    def add(kind, severity, title, col, evidence, why, rows=None, fix=None, fix_label=None, extra=None):
        rows = None if rows is None else list(rows)
        out.append({"id": f"{kind}:{col}", "kind": kind, "severity": severity, "title": title, "column": col,
                    "evidence": evidence, "why": why, "rows": rows, "fix_label": fix_label,
                    "fix": None if fix is None else {"type": fix, "column": col, "rows": rows, **(extra or {})}})

    dup = df.duplicated()
    if dup.any():
        add("dup_rows", "High", f"{int(dup.sum()):,} exact duplicate row(s)", None,
            "These rows are identical to an earlier row in every column.",
            "Repeated rows count the same observation twice, which makes results look more certain than they are.",
            dup[dup].index, "drop_rows", "Drop the duplicates (keep the first copy)")

    for c in df.columns:
        s = df[c]
        if is_text(s):
            tok = _tokens(s)
            if tok.any():
                seen = s[tok].astype(str).str.strip().value_counts().head(3)
                add("missing_code", "Medium", f"{int(tok.sum()):,} placeholder(s) standing in for missing values in {c}", c,
                    "Placeholders found: " + ", ".join(f"'{k}' x{v}" for k, v in seen.items()) + ".",
                    "They are read as real labels, so the column would get a made-up category instead of a gap.",
                    tok[tok].index, "to_missing", "Mark them as missing")
            if s.nunique() <= 50:
                raw = s.dropna().astype(str)
                v = pd.DataFrame({"raw": raw, "key": raw.str.lower()}).drop_duplicates()
                multi = v.groupby("key")["raw"].nunique()
                multi = multi[multi > 1]
                if len(multi):
                    counts, mapping = raw.value_counts(), {}
                    for k in multi.index:
                        group = v.loc[v["key"] == k, "raw"].tolist()
                        best = max(group, key=lambda x: counts[x])
                        mapping.update({x: best for x in group if x != best})
                    shown = "; ".join(" / ".join(repr(x) for x in v.loc[v["key"] == k, "raw"]) for k in list(multi.index)[:3])
                    add("labels", "Low", f"{len(multi)} label(s) in {c} differ only by capitalization", c,
                        f"Spellings seen: {shown}.",
                        "The same category would be counted as separate levels.",
                        s.index[s.isin(list(mapping))], "merge_labels", "Merge each into its most common spelling",
                        {"map": mapping})
            if s.nunique() >= 3:
                low = s.dropna().astype(str).str.strip().str.lower()
                hit = low.isin(SUMMARY_WORDS) | low.str.split().str[0].isin(SUMMARY_WORDS)
                if hit.any() and hit.mean() < 0.2:
                    add("summary_rows", "Medium", f"{int(hit.sum()):,} row(s) in {c} look like totals or averages", c,
                        "Labels such as total, world, all, average or overall appear among the entity labels.",
                        "A summary row duplicates the rows it summarizes, so it can distort every coefficient.",
                        hit[hit].index)
        else:
            vals = s.dropna()
            if vals.empty or s.nunique() <= 2:
                continue
            def set_apart(v):
                """A code sits at the edge of the column, with a gap to the other values of more than 2 SDs of those values."""
                rest = vals[vals != v]
                if rest.empty or v not in (vals.min(), vals.max()):
                    return False
                edge = rest.max() if v > rest.max() else rest.min()
                return abs(v - edge) > 2 * rest.std()
            sent = [v for v in SENTINELS if int((s == v).sum()) >= 2 and set_apart(v)]
            if sent:
                hit = s.isin(sent)
                rest = s[~hit]
                add("missing_code", "Medium", f"{int(hit.sum()):,} value(s) in {c} look like a missing-data code", c,
                    f"The value {sent[0]:g} repeats {int((s == sent[0]).sum()):,} times at the edge of the column; "
                    f"the other values range from {rest.min():g} to {rest.max():g}.",
                    "A code like this is read as a real number, so it can pull the whole fit.",
                    hit[hit].index, "to_missing", "Mark them as missing")
                continue
            neg = s < 0
            if neg.any() and (vals % 1 == 0).all() and (vals >= 0).mean() >= 0.95:
                add("negative", "Medium", f"{int(neg.sum()):,} negative value(s) in {c}, which otherwise looks like a count", c,
                    f"The lowest value is {vals.min():g}.", "A count cannot be negative, so these may be entry errors.", neg[neg].index)
            inside = ((vals >= 0) & (vals <= 100)).mean()
            if 0.99 <= inside < 1 and vals.nunique() > 5 and vals.max() <= 200:
                bad = (s < 0) | (s > 100)
                add("out_of_range", "Medium", f"{int(bad.sum()):,} value(s) in {c} outside 0 to 100, though most sit inside it", c,
                    f"Values run from {vals.min():g} to {vals.max():g}.",
                    "If this column is a percentage or share, values outside 0 to 100 are impossible.", bad[bad].index)

    if y_name in df.columns:
        r = df.select_dtypes("number").drop(columns=[y_name], errors="ignore").corrwith(df[y_name]).abs()
        for c in r.index[r > LEAK_R]:
            add("leakage", "High", f"{c} is almost a copy of {y_name} (|r| = {r[c]:.3f})", c,
                f"The correlation between {c} and {y_name} is {r[c]:.3f}.",
                "A predictor that nearly equals Y often comes from the same source or is computed from it, so the fit looks perfect without being useful.")

    exact = df.duplicated(keep=False)
    times = [c for c in df.columns if c != y_name and not is_text(df[c]) and 3 <= df[c].nunique() <= 200
             and (df[c].dropna() % 1 == 0).all() and df[c].max() - df[c].min() + 1 <= 1.5 * df[c].nunique()]
    for e in entity_candidates(df, 3)[:3]:
        for t in times[:5]:
            pair = df.duplicated([e, t], keep=False)
            rep = pair & ~exact
            if rep.any() and pair.mean() < 0.1:  # a real key is mostly unique; a heavily repeated pair is just a grouping
                add("dup_keys", "Medium", f"{int(rep.sum()):,} rows share the same {e} and {t} but hold different values", f"{e}+{t}",
                    f"Each {e} and {t} pair should appear once if {t} is a time period, but these pairs repeat.",
                    "Repeated keys can mean double entries or a missing grouping column, so rows are not independent.", rep[rep].index)
                break

    out.sort(key=lambda f: {"High": 0, "Medium": 1, "Low": 2}[f["severity"]])
    return out


def apply_audit(df, findings, decisions):
    """Apply only the fixes the user approved. Returns (table, log of what was applied)."""
    df, applied, drop = df.copy(), [], set()
    for f in findings:
        fx = f["fix"]
        if not fx or not decisions.get(f["id"]):
            continue
        t, c, rows = fx["type"], fx["column"], fx["rows"]
        if t == "to_missing":
            if not is_text(df[c]):
                df[c] = df[c].astype(float)
            df.loc[rows, c] = np.nan
            applied.append(f"Audit fix: marked {len(rows):,} value(s) in {c} as missing.")
        elif t == "merge_labels":
            df[c] = df[c].replace(fx["map"])
            applied.append(f"Audit fix: merged {len(fx['map'])} label spelling(s) in {c}.")
        elif t == "drop_rows":
            drop.update(rows)
    if drop:
        df = df.drop(index=list(drop))
        applied.append(f"Audit fix: dropped {len(drop):,} duplicate row(s).")
    return df, applied


@st.cache_data(show_spinner=False)
def prepare_upload(name, data):
    """Read and tidy the uploaded file once per file (Streamlit reruns the script on every click)."""
    buf = io.BytesIO(data)
    return auto_fix(pd.read_csv(buf) if name.lower().endswith(".csv") else pd.read_excel(buf))


audit_cached = st.cache_data(show_spinner=False)(audit_data)  # recomputed only when the table or the target changes


def audit_record(auto_log, findings, decisions):
    """What goes into recipe['audit']: auto-fixes, every finding with its decision. Filled with 'applied' after apply_audit."""
    status = lambda f: "flag only" if not f["fix"] else ("fix applied" if decisions.get(f["id"]) else "left as is")
    return {"auto_fixes": list(auto_log), "applied": [],
            "findings": [{"severity": f["severity"], "title": f["title"], "evidence": f["evidence"], "decision": status(f)}
                         for f in findings]}


def audit_lines(au):
    """One-line summaries for the report and Excel. No cell values, so they are safe to send to Claude."""
    if not au:
        return ["No audit was recorded."]
    lines = list(au["auto_fixes"]) + list(au["applied"])
    lines += [f"[{f['severity']}] {f['title']}: {f['decision']}" for f in au["findings"]]
    return lines or ["The audit found nothing to flag."]


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
    for k in [k for k in st.session_state if str(k).startswith("audit_")]:
        st.session_state.pop(k)  # audit decision radios from the previous file
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
    df, auto_log = prepare_upload(uploaded.name, uploaded.getvalue())

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
        st.dataframe(df.head(10))
    if len(miss):
        with st.expander(f"Which columns have gaps? ({len(miss)})"):
            st.dataframe(
                pd.DataFrame({"Missing": miss, "Percent": miss / len(df) * 100}).style.format({"Missing": "{:,.0f}", "Percent": "{:.1f}%"}),
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

    # --- Data audit: errors in the data itself ---
    st.divider()
    ui.section("Data audit", "Looks for errors in the data itself, not just gaps. Nothing changes unless you pick a fix; every choice is logged in the recipe.")
    findings = audit_cached(df, y_name)
    n_dup = int(df.duplicated().sum())
    cols_hit = {f["column"] for f in findings if f["column"]}
    ui.metric_strip([
        ("Rows", f"{len(df):,}"),
        ("Exact duplicates", f"{n_dup:,}", "Rows identical to an earlier row in every column."),
        ("Findings", f"{len(findings)}", f"Things worth a look, involving {len(cols_hit)} column(s) by name."),
        ("Auto-fixes applied", f"{len(auto_log)}", "Safe tidy-ups done on load: trimmed spaces, numbers stored as text."),
    ])
    if auto_log:
        with st.expander("Automatic tidy-ups already applied"):
            st.markdown("\n".join(f"- {t}" for t in auto_log))
    if not findings:
        st.success("No data errors flagged by the automatic checks.")
    decisions = {}
    for f in findings:
        with st.container(border=True):
            st.markdown(f":orange[●] **{f['title']}**  ·  {f['severity']}")
            st.caption(f["evidence"] + " " + f["why"])
            if f["fix"]:
                choice = st.radio(f"Decision: {f['title']}", ["Leave as is", f["fix_label"]], key=f"audit_{f['id']}",
                                  horizontal=True, label_visibility="collapsed")
                decisions[f["id"]] = choice != "Leave as is"
            else:
                st.caption("Flag only: the app does not change this. Decide whether to exclude or fix it in your source file.")
            if f["rows"]:
                with st.expander(f"See the rows ({len(f['rows']):,}; first 10 shown)"):
                    st.dataframe(df.loc[f["rows"][:10]])

    if st.button("Run Completeness Check", type="primary"):
        fixed, applied = apply_audit(df, findings, decisions)
        audit = audit_record(auto_log, findings, decisions)
        audit["applied"] = applied
        st.session_state.clean = clean_data(fixed, y_name, exclude, encode_high, audit)
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
        st.dataframe(clean_df.head(10))

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
    """Variance inflation factor for each predictor. VIF is the diagonal of the inverse correlation matrix,
    so one matrix inverse replaces one regression per column (same numbers, much faster on big tables)."""
    X = df[x_cols].to_numpy(dtype=float)
    if X.shape[1] == 1:
        return pd.Series([1.0], index=x_cols)
    try:
        with np.errstate(all="ignore"):
            vif = np.diag(np.linalg.inv(np.corrcoef(X, rowvar=False)))
        vif = np.where(np.isfinite(vif) & (vif > 0), vif, np.inf)
    except np.linalg.LinAlgError:  # singular: perfect collinearity
        vif = np.full(X.shape[1], np.inf)
    return pd.Series(vif, index=x_cols)


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
        "ranked_findings": recipe.get("data_picture", []),
        "cleaning_recipe": {k: v for k, v in recipe.items() if k not in ("final_columns", "audit", "data_picture")},  # audit evidence can hold cell values
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
    "ranked_findings are Python's own checks, most severe first: build the brief around them. "
    "Be concise, use only the numbers provided, and do not invent facts about the data."
)


def ask_claude_brief(profile, api_key, system=BRIEF_SYSTEM, teach=True):
    client = anthropic.Anthropic(api_key=api_key, timeout=120.0, max_retries=2)
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


def split_is_median(raw, col):
    """Many-valued numeric columns are always split at the median; 3+ valued ones use the user's checkbox."""
    if is_text(raw[col]):
        return False
    n = raw[col].nunique()
    if n > MAX_LEVELS:
        return True
    return n > 2 and bool(st.session_state.get(f"seg_median_{col}", False))


def run_split():
    """Button callback: remember the current split and count it once in the recipe.
    Reads the widgets from session state at click time. Arguments bound at render time can be stale when the
    user changes the column and clicks before the slow Stage 2 rerun finishes, which loaded the old column."""
    raw = st.session_state.clean["raw"]
    col = st.session_state.get("seg_col")
    if col not in raw.columns:
        return
    median_split = split_is_median(raw, col)
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
    st.button("Compare groups", on_click=run_split, type="primary")

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
    )
    if small:
        st.warning(f"Small group(s) (under {SMALL_GROUP} rows), so correlations here are unreliable: " + ", ".join(map(str, small)))
    if recipe["dropped_high_cardinality"]:
        st.caption("Rows repeat the same entities (panel data), so group differences can look stronger than they are.")

    # --- Correlation gaps ---
    st.markdown("**Correlation of each predictor with Y, by group** (largest gaps first)")
    st.dataframe(
        corr.style.format("{:.3f}", na_rep="n/a").background_gradient(subset=["Gap"], cmap="Oranges"),
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
            st.dataframe(coef.style.format("{:.4f}", na_rep="n/a"))
            st.dataframe(
                gstats.style.format({"n": "{:,.0f}", "Predictors used": "{:.0f}", "R Square": "{:.3f}", "Adjusted R Square": "{:.3f}"}, na_rep="n/a"),
            )
            st.markdown("**P-values by group**")
            st.dataframe(pval.style.format("{:.4f}", na_rep="n/a"))
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






# ---------- Ranked data picture (U3-A) ----------
# Python runs a library of checks on the cleaned table. Each one that trips gets a severity from 0 to 1,
# a small evidence table, a plain "why", an action, and a chart that is only drawn when the user asks for it.
TOP_FINDINGS = 5   # findings shown up front; the rest sit in an expander


def _clip01(v):
    return float(min(1.0, max(0.0, v)))


def skew_fig(series, name):
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.2))
    axes[0].hist(series, bins=30, color=ui.ACCENT, edgecolor=ui.CARD)
    axes[0].set_title(str(name), fontsize=9)
    axes[1].hist(np.log1p(series), bins=30, color=ui.HIGHLIGHT, edgecolor=ui.CARD)
    axes[1].set_title(f"ln(1+{name})", fontsize=9)
    fig.tight_layout()
    return fig


def curve_fig(df, x, y):
    """X vs Y with the straight-line fit and the fit that also uses x squared."""
    mu, sd = df[x].mean(), df[x].std()
    z = (df[x] - mu) / sd
    zs = np.linspace(z.min(), z.max(), 100)
    fig, ax = plt.subplots(figsize=(5, 3.2))
    ax.scatter(df[x], df[y], s=8, alpha=0.3, color=ui.SOFT)
    for deg, color, label in ((1, ui.HIGHLIGHT, "Straight line"), (2, "#54A24B", "With x squared")):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # a nearly discrete predictor can trigger a harmless conditioning warning
            fit = np.polyfit(z, df[y], deg)
        ax.plot(mu + sd * zs, np.polyval(fit, zs), color=color, linewidth=2, label=label)
    ax.set_xlabel(str(x))
    ax.set_ylabel(str(y))
    ax.legend(fontsize=7)
    fig.tight_layout()
    return fig


def cooks_fig(cooks):
    fig, ax = plt.subplots(figsize=(6, 3.2))
    ax.vlines(range(len(cooks)), 0, cooks, color=ui.ACCENT, linewidth=0.6)
    ax.axhline(4 / len(cooks), color=ui.HIGHLIGHT, linestyle="--", label="Rule of thumb 4/n")
    ax.set_xlabel("Row position")
    ax.set_ylabel("Cook's distance")
    ax.legend(fontsize=7)
    fig.tight_layout()
    return fig


def barh_fig(series, title, xlabel, top_n=10):
    s = series.sort_values(ascending=False).head(top_n)[::-1]
    fig, ax = plt.subplots(figsize=(6, max(2.2, 0.35 * len(s) + 1)))
    ax.barh([str(i) for i in s.index], s.values, color=ui.ACCENT)
    ax.set_title(title, fontsize=9)
    ax.set_xlabel(xlabel)
    fig.tight_layout()
    return fig


def picture_checks(df, y, recipe, raw):
    """Checks on the cleaned table, most severe first. Dataset-agnostic: nothing here looks at column names."""
    x_cols = [c for c in df.columns if c != y]
    X, yv = df[x_cols], df[y]
    nonbin = [c for c in x_cols if df[c].nunique() > 2]
    corr_y = X.corrwith(yv).fillna(0)
    n, out = len(df), []

    def add(key, title, severity, headline, evidence, fmt, why, action, chart=None):
        if severity > 0:
            out.append({"key": key, "title": title, "severity": _clip01(severity), "headline": headline,
                        "evidence": evidence, "fmt": fmt, "why": why, "action": action, "chart": chart})

    # 1. Skew, ranked by impact: how much ln(1+x) would improve the link with Y (skew alone does not matter much)
    skewed = [c for c in nonbin if abs(df[c].skew()) > 1 and df[c].min() > -1]
    if skewed:
        rows = []
        for c in skewed:
            r_log = abs(np.nan_to_num(np.corrcoef(np.log1p(df[c]), yv)[0, 1]))
            rows.append({"Predictor": c, "Skew": df[c].skew(), "|r| with Y": abs(corr_y[c]),
                         "|r| after ln(1+x)": r_log, "Gain": r_log - abs(corr_y[c])})
        ev = pd.DataFrame(rows).set_index("Predictor").sort_values("Gain", ascending=False)
        g, top = float(ev["Gain"].iloc[0]), ev.index[0]
        add("skew", "Skewed predictors", _clip01(g / 0.3) if g > 0 else 0.1,
            f"{len(skewed)} predictor(s) are strongly skewed; a log helps most for **{top}** (|r| with Y {ev.loc[top, '|r| with Y']:.2f} to {ev.loc[top, '|r| after ln(1+x)']:.2f}).",
            ev.head(8), {c: "{:.3f}" for c in ev.columns},
            "A long tail lets a few huge values steer the fit and can bend the link with Y. Ranked by how much a log would improve the link, not by skew alone.",
            "Try ln(1+x) for the top rows in Stage 4. Regression assumes roughly normal residuals, not normal predictors.",
            lambda top=top: skew_fig(df[top], top))

    # 2. Collinearity: VIF and near-duplicate pairs
    if len(x_cols) > 1:
        vif = compute_vif(df, x_cols)
        high = vif[vif > 10].sort_values(ascending=False)
        if len(high):
            cm = X.corr()
            absc = cm.abs().where(~np.eye(len(x_cols), dtype=bool), 0)
            rows = [{"Predictor": c, "VIF": high[c], "Closest partner": absc[c].idxmax(), "r with partner": cm.loc[c, absc[c].idxmax()]}
                    for c in high.index[:10]]
            a, b = np.unravel_index(np.argmax(absc.to_numpy()), absc.shape)
            vmax = float(high.iloc[0])
            add("collinearity", "Predictors that repeat each other", 1.0 if not np.isfinite(vmax) else math.log10(vmax) / 3,
                f"{len(high)} predictor(s) have VIF above 10 (largest {vmax:,.0f}).",
                pd.DataFrame(rows).set_index("Predictor"), {"VIF": "{:,.1f}", "r with partner": "{:+.3f}"},
                "These predictors mostly repeat others, so their coefficients can swing wildly or flip sign.",
                "Drop one of each pair in the Base Data Model, or combine them in Stage 4.",
                lambda a=a, b=b: scatter_fig(df, x_cols[a], x_cols[b]))

    # 3. Outliers and influence
    try:
        cooks = run_regression(df, y, x_cols).get_influence().cooks_distance[0]
        share, mx = float((cooks > 4 / n).mean()), float(cooks.max())
    except Exception:
        cooks, share, mx = None, 0.0, 0.0
    if cooks is not None and (share > 0.03 or mx > 0.5):
        q1, q3 = X[nonbin].quantile(0.25), X[nonbin].quantile(0.75)
        iqr = q3 - q1
        cnt = ((X[nonbin] < q1 - 1.5 * iqr) | (X[nonbin] > q3 + 1.5 * iqr)).sum()
        ev = pd.DataFrame({"Outliers (1.5×IQR)": cnt, "Share of rows": cnt / n}).sort_values("Outliers (1.5×IQR)", ascending=False).head(8)
        add("outliers", "Influential rows and outliers", max(share / 0.10, mx),
            f"{share * 100:.1f}% of rows have a Cook's distance above 4/n; the largest is {mx:.2f}.",
            ev, {"Outliers (1.5×IQR)": "{:,.0f}", "Share of rows": "{:.1%}"},
            "A few extreme rows can steer the coefficients. The table counts values beyond 1.5 interquartile ranges, by predictor.",
            "Stage 3 lists the most influential rows. Check them for entry errors, then see how much the fit changes without them.",
            lambda cooks=cooks: cooks_fig(cooks))

    # 4. Curvature: does adding x squared explain more of Y than a straight line?
    rows, ss = [], float(((yv - yv.mean()) ** 2).sum())
    for c in nonbin:
        if df[c].nunique() <= 5:
            continue
        z = (df[c] - df[c].mean()) / df[c].std()
        r2_lin = float(corr_y[c] ** 2)
        r2_quad = 1 - float(((yv - np.polyval(np.polyfit(z, yv, 2), z)) ** 2).sum()) / ss
        if r2_quad - r2_lin > 0.01:
            rows.append({"Predictor": c, "R² straight": r2_lin, "R² with square": r2_quad, "Gain": r2_quad - r2_lin})
    if rows:
        ev = pd.DataFrame(rows).set_index("Predictor").sort_values("Gain", ascending=False)
        top = ev.index[0]
        add("curvature", "Curved relationships", ev["Gain"].iloc[0] / 0.20,
            f"{len(ev)} predictor(s) fit noticeably better with a squared term; the biggest gain is **{top}** (R² {ev.loc[top, 'R² straight']:.2f} to {ev.loc[top, 'R² with square']:.2f}).",
            ev.head(8), {c: "{:.3f}" for c in ev.columns},
            "A straight line misses a bend, so predictions are biased at the ends and the middle. R² here is one predictor alone against Y.",
            "Add the square of the top predictors in Stage 4.",
            lambda top=top: curve_fig(df, top, y))

    # 5. Missingness that was filled with medians
    filled = {c: v for c, v in recipe.get("median_imputed_counts", {}).items() if c in df.columns}
    if filled:
        ev = pd.DataFrame({"Filled values": pd.Series(filled)}).sort_values("Filled values", ascending=False)
        ev["Share of rows"] = ev["Filled values"] / n
        mxs = float(ev["Share of rows"].iloc[0])
        if mxs > 0.05:
            add("missing", "Values filled with medians", mxs / 0.30,
                f"{int((ev['Share of rows'] > 0.05).sum())} predictor(s) had more than 5% of values filled; the most is **{ev.index[0]}** at {mxs:.0%}.",
                ev.head(8), {"Filled values": "{:,.0f}", "Share of rows": "{:.1%}"},
                "A filled value is a guess, not data. Heavily filled columns pull coefficients toward the middle and make p-values look better than they are.",
                "Treat results for these predictors with extra caution, or exclude them in Stage 1.",
                lambda ev=ev: barh_fig(ev["Share of rows"] * 100, "Share of rows filled", "% of rows"))

    # 6. Low variance
    rows = []
    for c in x_cols:
        share_top = float(df[c].value_counts(normalize=True).iloc[0])
        if df[c].nunique() == 2 and min(share_top, 1 - share_top) < 0.02:
            rows.append({"Predictor": c, "Most common value's share": share_top, "Why": "almost all rows are the same 0/1 value"})
        elif df[c].nunique() > 2 and share_top > 0.9:
            rows.append({"Predictor": c, "Most common value's share": share_top, "Why": "one value covers over 90% of rows"})
    if rows:
        ev = pd.DataFrame(rows).set_index("Predictor").sort_values("Most common value's share", ascending=False)
        add("lowvar", "Predictors that barely vary", 0.3 + 0.1 * min(len(ev), 5),
            f"{len(ev)} predictor(s) are nearly constant.", ev.head(8), {"Most common value's share": "{:.1%}"},
            "With almost no variation there is little to learn from, so the coefficient is unstable and its effect cannot be told apart from noise.",
            "Consider excluding them in the Base Data Model.",
            lambda ev=ev: barh_fig(ev["Most common value's share"] * 100, "Share of rows with the most common value", "% of rows"))

    # 7. Group differences: 0/1 columns that split Y widely
    rows = []
    for c in [c for c in x_cols if df[c].nunique() == 2]:
        hi, lo = df[c] == df[c].max(), df[c] == df[c].min()
        if min(hi.sum(), lo.sum()) >= SMALL_GROUP:
            rows.append({"Column": c, "Rows with 1": int(hi.sum()), "Rows with 0": int(lo.sum()),
                         "Mean Y (1)": yv[hi].mean(), "Mean Y (0)": yv[lo].mean(),
                         "Gap (SDs of Y)": (yv[hi].mean() - yv[lo].mean()) / yv.std()})
    if rows:
        ev = pd.DataFrame(rows).set_index("Column")
        ev = ev.reindex(ev["Gap (SDs of Y)"].abs().sort_values(ascending=False).index)
        top = ev.index[0]
        gap = float(abs(ev["Gap (SDs of Y)"].iloc[0]))
        if gap > 0.3:
            add("groups", "Groups with different average Y", (gap - 0.3) / 1.5,
                f"**{top}** splits the rows into groups whose average Y differs by {gap:.2f} standard deviations.",
                ev.head(8), {"Rows with 1": "{:,.0f}", "Rows with 0": "{:,.0f}", "Mean Y (1)": "{:.3f}", "Mean Y (0)": "{:.3f}", "Gap (SDs of Y)": "{:+.2f}"},
                "A column that separates Y this much may relate to Y differently inside each group, which one pooled line would blur.",
                "Use the Segment Explorer below to compare the groups, or add an interaction in Stage 4.",
                lambda top=top: box_fig(df[top].map({df[top].min(): "0", df[top].max(): "1"}), yv, top, y))

    # 8. Leakage
    leak = corr_y[corr_y.abs() > LEAK_R]
    if len(leak):
        top = leak.abs().idxmax()
        add("leakage", "Predictors that nearly equal Y", 1.0,
            f"**{top}** has a correlation of {corr_y[top]:+.3f} with Y.",
            pd.DataFrame({"r with Y": leak}), {"r with Y": "{:+.4f}"},
            "A predictor this close to Y usually comes from the same source or is computed from it, so the fit looks perfect without being useful.",
            "Exclude it in Stage 1 unless it is known before Y would be.",
            lambda top=top: scatter_fig(df, top, y))

    # 9. Panel or repeat structure
    ents = [c for c in dict.fromkeys(list(recipe.get("dropped_high_cardinality", [])) + entity_candidates(raw)) if c in raw.columns]
    if ents:
        ev = pd.DataFrame({"Distinct values": [raw[c].nunique() for c in ents],
                           "Rows per value": [len(raw) / raw[c].nunique() for c in ents]}, index=ents)
        add("panel", "Rows that may repeat the same entity", 0.6,
            f"{len(ents)} text column(s) repeat across rows, such as **{ents[0]}** ({ev['Rows per value'].iloc[0]:.1f} rows per value).",
            ev, {"Distinct values": "{:,.0f}", "Rows per value": "{:.1f}"},
            "If these identify repeated entities (one country seen every year), rows are not independent, so p-values look more certain than they are.",
            "Use whole-entity holdout in Stage 5. Clustered standard errors are planned in U6.")

    return sorted(out, key=lambda c: -c["severity"])


def show_picture_findings(checks):
    """Top findings as cards (headline, evidence, why, action, chart on request); the rest in an expander."""
    ui.section("What stands out", "Python's checks, ranked by severity for this dataset. The top five are shown; each has its evidence and a chart one click away.")
    if not checks:
        st.success("No major issues flagged by the automatic checks.")
        return

    def card(chk):
        with st.container(border=True):
            ui.finding_header(chk["title"], chk["severity"])
            st.markdown(chk["headline"])
            st.caption(chk["why"])
            st.dataframe(chk["evidence"].style.format(chk["fmt"], na_rep="n/a"))
            st.caption("**What you can do:** " + chk["action"])
            if chk["chart"] and st.toggle("Show chart", key=f"pic_chart_{chk['key']}"):
                fig = chk["chart"]()
                st.pyplot(fig)
                plt.close(fig)

    for chk in checks[:TOP_FINDINGS]:
        card(chk)
    if len(checks) > TOP_FINDINGS:
        with st.expander(f"{len(checks) - TOP_FINDINGS} more finding(s), lower severity"):
            for chk in checks[TOP_FINDINGS:]:
                card(chk)


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
    )
    how_to_read("summary")

    checks = picture_checks(df, y, recipe, result["raw"])
    recipe["data_picture"] = [{"check": c["key"], "title": c["title"], "severity": round(c["severity"], 2),
                               "headline": c["headline"].replace("**", "")} for c in checks]
    st.divider()
    show_picture_findings(checks)

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
                    st.dataframe(ft.style.format({"Count": "{:,.0f}", "Percent": "{:.1f}%"}))
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
        st.dataframe(qual.style.format({"Filled (were missing)": "{:,.0f}", "Outliers (1.5×IQR rule)": "{:,.0f}", "Outliers %": "{:.1f}%"}))
        st.caption("A column with many filled values leans on median guesses, so treat its coefficient with more caution. "
                   "Outliers are values more than 1.5 interquartile ranges beyond the middle half of the data. They are not errors, just values that can pull a fit.")

    with st.expander("Show Class Math & Descriptive Stats"):
        st.markdown("**Descriptive statistics (Excel layout)**")
        st.dataframe(descriptive_stats(df).style.format("{:,.4f}", na_rep="#N/A"))
        st.markdown("**Five-number summary**")
        st.dataframe(five_number(df).style.format("{:,.4f}"))
        st.markdown("**Correlation matrix (Pearson)**")
        st.dataframe(corr.style.format("{:.3f}"))

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
        st.dataframe(pd.concat([stats_cols, raw.loc[top.index]], axis=1))
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
        st.dataframe(ct.style.format({"Coefficients": "{:.4f}", "P-value": "{:.4f}"}))
        st.markdown(HOW_TO_READ["coef"])
    with st.expander("Excel ToolPak tables"):
        st.markdown("**Regression Statistics**")
        st.dataframe(reg.style.format("{:,.6f}"))
        st.markdown("**ANOVA**")
        st.dataframe(anova.style.format("{:,.6g}", na_rep=""))
        st.markdown("**Coefficients**")
        st.dataframe(coef.style.format("{:,.6g}"))

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













# ---------- Candidate generator and screening (U3-B) ----------
# Python proposes every transformation a person would consider at first glance (from the same fixed menu),
# then tests each one alone against the Base Data Model on the training rows. The analyst ticks what goes into the plan.
GEN_DEFAULTS = {
    "log_skew": 1.0,             # log: non-binary predictors with |skew| above this (and all values above -1)
    "square_top_k": 5,           # square: this many strongest predictors, plus up to this many with curvature evidence
    "combine_r": 0.7,            # combine: pairs of predictors with |r| at least this
    "combine_max_pairs": 8,      # ...at most this many pairs (strongest first)
    "interaction_top_k": 3,      # interaction: this many strongest predictors...
    "interaction_max_binary": 4, # ...times this many 0/1 columns
    "dup_r": 0.98,               # drop a candidate with |r| above this with anything already in the pool
    "pool_cap": 60,              # keep at most this many candidates (best single-move BIC first)
}
GEN_LABELS = {
    "log_skew": "Log: minimum |skew|", "square_top_k": "Square: top predictors",
    "combine_r": "Combine: minimum |r| between the pair", "combine_max_pairs": "Combine: maximum pairs",
    "interaction_top_k": "Interaction: top predictors", "interaction_max_binary": "Interaction: maximum 0/1 columns",
    "dup_r": "Near-duplicate filter: |r| above", "pool_cap": "Pool cap (candidates kept)",
}
SCREEN_PAIR_TOP = 30   # pairs are tested among this many best single additions


def family_of(step):
    """x, ln1p_x and x_sq share the family of x; interactions and combinations are their own family."""
    return str(step["cols"][0]) if step["op"] in ("log", "square") else step_name(step)


def generate_candidates(df, y, x_cols, avail, settings, taken):
    """Candidate steps (source "auto") that build, are not constant, and are not near-duplicates. `taken` = names already in use."""
    yv, n = df[y], len(df)
    nonbin = [c for c in x_cols if df[c].nunique() > 2]
    corr_y = df[avail].corrwith(yv).fillna(0)
    steps = []

    def propose(op, cols, how, reason):
        steps.append({"op": op, "cols": list(cols), "how": how, "reason": reason, "source": "auto"})

    for c in nonbin:
        sk = df[c].skew()
        if abs(sk) > settings["log_skew"] and df[c].min() > -1:
            propose("log", [c], None, f"auto: skew {sk:.1f}")
    curv, ss = {}, float(((yv - yv.mean()) ** 2).sum())
    for c in nonbin:
        if df[c].nunique() > 5:
            z = (df[c] - df[c].mean()) / df[c].std()
            gain = 1 - float(((yv - np.polyval(np.polyfit(z, yv, 2), z)) ** 2).sum()) / ss - float(corr_y[c] ** 2)
            if gain > 0.01:
                curv[c] = gain
    k = int(settings["square_top_k"])
    by_r = list(corr_y[nonbin].abs().sort_values(ascending=False).index[:k])
    for c in dict.fromkeys(by_r + sorted(curv, key=curv.get, reverse=True)[:k]):
        propose("square", [c], None, f"auto: |r| with Y {abs(corr_y[c]):.2f}" + (f", curvature gain {curv[c]:.3f}" if c in curv else ""))
    if len(nonbin) > 1:
        cm = df[nonbin].corr()
        pairs = sorted(((abs(cm.loc[a, b]), a, b) for a, b in combinations(nonbin, 2)
                        if abs(cm.loc[a, b]) >= settings["combine_r"]), reverse=True)[:int(settings["combine_max_pairs"])]
        for _, a, b in pairs:
            why = f"auto: r = {cm.loc[a, b]:+.2f} between the pair"
            propose("combine", [a, b], "mean", why)
            propose("combine", [a, b], "difference", why)
            if (df[b] != 0).all():
                propose("combine", [a, b], "ratio", why)
    bins = [c for c in avail if df[c].nunique() == 2 and min(int((df[c] == df[c].max()).sum()), int((df[c] == df[c].min()).sum())) >= SMALL_GROUP]
    bins = sorted(bins, key=lambda c: -abs(corr_y[c]))[:int(settings["interaction_max_binary"])]
    for a in by_r[:int(settings["interaction_top_k"])]:
        for b in bins:
            if b != a:
                propose("interaction", [a, b], None, f"auto: does the slope of {a} differ between {b} groups?")

    def zs(s):
        return ((s - s.mean()) / s.std()).to_numpy(dtype=float)

    pool = [(str(c), zs(df[c])) for c in x_cols]  # (family, standardized values) already in the pool
    kept, built, seen = [], {}, set(map(str, taken))
    for s in steps:
        try:
            name, col = build_feature(s, df)
        except ValueError:
            continue
        if name in seen or not np.isfinite(col).all() or col.nunique() <= 1:
            continue
        fam, z = family_of(s), zs(col)
        # a square is allowed to track its own parent closely (that pair is how curvature is fit)
        if any(abs(float(z @ o) / (n - 1)) > settings["dup_r"] for f, o in pool if not (s["op"] == "square" and f == fam)):
            continue
        s.update(name=name, family=fam)
        pool.append((fam, z))
        seen.add(name)
        kept.append(s)
        built[name] = col
    return kept, built


def fast_stats(df, y, cols):
    """fit(indices) -> Adjusted R Square, BIC and each term's t Stat, from small matrix solves (same OLS answer)."""
    n = len(df)
    X = df[cols].to_numpy(dtype=float)
    sd = X.std(axis=0)
    sd[sd == 0] = 1.0
    Z = (X - X.mean(axis=0)) / sd
    yc = df[y].to_numpy(dtype=float)
    yc = yc - yc.mean()
    G, c, sst = Z.T @ Z, Z.T @ yc, float(yc @ yc)

    def fit(idx):
        idx = list(idx)
        k = len(idx)
        if k == 0:
            return {"adj": 0.0, "bic": bic_value(n, sst, 0), "t": np.array([])}
        Gs = G[np.ix_(idx, idx)]
        if n - k - 1 <= 0 or np.linalg.cond(Gs) > 1e10:
            return None
        inv = np.linalg.inv(Gs)
        beta = inv @ c[idx]
        sse = sst - float(beta @ c[idx])
        if sse <= 1e-12 * sst:
            return None
        t = beta / np.sqrt(sse / (n - k - 1) * np.diag(inv))
        return {"adj": 1 - (sse / sst) * (n - 1) / (n - k - 1), "bic": bic_value(n, sse, k), "t": t}

    return fit


def screen_moves(train, y, x_cols, cands):
    """One change at a time from the Base Data Model: add each candidate, remove each base predictor, swap a base
    predictor for a member of its family. Then pairs of the best additions. cands = {name: step}."""
    names = list(cands)
    cols = list(x_cols) + names
    fit = fast_stats(train, y, cols)
    base = list(range(len(x_cols)))
    b0 = fit(base)
    if b0 is None:
        raise ValueError("The Base Data Model could not be fitted on the training rows.")
    rows = []

    def add_row(move, var, r, t, step=None):
        if r is not None:
            rows.append({"Move": move, "Variable": var, "Δ BIC": r["bic"] - b0["bic"], "Δ Adj R Square": r["adj"] - b0["adj"],
                         "t Stat": t, "Formula": step_recipe(step) if step else "-", "Source": step["source"] if step else "base",
                         "_idx": None})

    for j, name in enumerate(names, start=len(x_cols)):
        r = fit(base + [j])
        add_row("Add", name, r, float(r["t"][-1]) if r else None, cands[name])
        if rows and rows[-1]["Variable"] == name:
            rows[-1]["_idx"] = j
    for i, name in enumerate(x_cols):
        add_row("Remove", str(name), fit([b for b in base if b != i]), float(b0["t"][i]))
    for j, name in enumerate(names, start=len(x_cols)):
        fam = cands[name].get("family")
        if fam in map(str, x_cols) and cands[name]["op"] == "log":  # a log can replace its parent; a square sits beside it
            i = [str(c) for c in x_cols].index(fam)
            r = fit([b for b in base if b != i] + [j])
            add_row("Swap", f"{fam} → {name}", r, float(r["t"][-1]) if r else None, cands[name])
    moves = pd.DataFrame(rows)
    pairs = pd.DataFrame()
    if not moves.empty:
        adds = moves[moves["Move"] == "Add"].nsmallest(SCREEN_PAIR_TOP, "Δ BIC")
        single = dict(zip(adds["_idx"], adds["Δ BIC"]))
        prow = []
        for a, b in combinations(list(single), 2):
            r = fit(base + [int(a), int(b)])
            if r is not None:
                d = r["bic"] - b0["bic"]
                prow.append({"Add both": f"{cols[int(a)]} + {cols[int(b)]}", "Δ BIC": d, "Δ Adj R Square": r["adj"] - b0["adj"],
                             "Together vs separately": d - single[a] - single[b]})
        pairs = pd.DataFrame(prow)
        if not pairs.empty:
            pairs = pairs.sort_values("Δ BIC").head(15).reset_index(drop=True)
        moves = moves.drop(columns="_idx").sort_values("Δ BIC").reset_index(drop=True)
    return moves, pairs, {"base_bic": b0["bic"], "base_adj": b0["adj"]}


def run_screening(df, y, x_cols, avail, plan, work, raw, settings, recipe):
    """Generate candidates, screen them with the plan's own columns on the default training rows, apply the pool cap."""
    plan_steps = {step_name(s): s for s in plan if step_name(s)}
    plan_cols = [c for c in work.columns if c not in df.columns and c != y]
    gen, built = generate_candidates(df, y, x_cols, avail, settings, list(work.columns) + list(df.columns))
    frame = df[[y] + list(x_cols)].copy()
    cands = {}
    for c in plan_cols:
        frame[c] = work[c]
        st_ = dict(plan_steps.get(c, {"op": "?", "cols": [c], "how": None, "reason": ""}))
        st_.setdefault("source", "user")
        st_["family"] = family_of(st_) if st_["op"] != "?" else c
        cands[c] = st_
    for s in gen:
        frame[s["name"]] = built[s["name"]]
        cands[s["name"]] = s
    mask = split_holdout(frame.index, raw, None)  # the same default split Stage 5 starts with (random rows)
    moves, pairs, base = screen_moves(frame[~mask], y, x_cols, cands)
    adds = moves[(moves["Move"] == "Add") & (moves["Source"] == "auto")].set_index("Variable")
    cap = int(settings["pool_cap"])
    keep = set(adds.nsmallest(cap, "Δ BIC").index)
    gen = [s for s in gen if s["name"] in keep]
    dropped = [v for v in adds.index if v not in keep]
    if dropped:
        moves = moves[~moves["Variable"].isin(dropped)].reset_index(drop=True)
    for s in gen:
        s["screen_bic"] = float(adds.loc[s["name"], "Δ BIC"])
        s["screen_adj"] = float(adds.loc[s["name"], "Δ Adj R Square"])
        s["screen_t"] = adds.loc[s["name"], "t Stat"]
    recipe["candidates_generated"] = [{k: s.get(k) for k in ("name", "op", "cols", "how", "family", "source", "screen_bic", "screen_adj")}
                                      for s in gen]
    best = moves.head(5)
    recipe["screening_summary"] = {
        "train_rows": int((~mask).sum()), "split": "default random-row holdout split (seed HOLDOUT_SEED)",
        "candidates_generated": len(gen), "cut_by_pool_cap": len(dropped), "moves_tested": len(moves), "pairs_tested": len(pairs),
        "best_moves": [f"{r['Move']} {r['Variable']}: Δ BIC {r['Δ BIC']:+.1f}, Δ Adj R Square {r['Δ Adj R Square']:+.4f}" for _, r in best.iterrows()],
        "settings": dict(settings),
    }
    return {"gen": gen, "moves": moves, "pairs": pairs, "base": base, "plan_names": sorted(plan_steps),
            "train_rows": int((~mask).sum()), "cut": len(dropped), "version": st.session_state.get("screen_version", 0) + 1}


def show_candidates(df, y, x_cols, avail, plan, edited, work, raw, recipe):
    """Generator settings, the run button, the candidate table with Add ticks, and the screening tables."""
    ui.section("Automatic candidates and screening",
               "Python proposes every transformation a person would consider at first glance, then tests each one alone against your Base Data Model "
               "on the training rows. Nothing joins the plan until you tick it.")
    with st.expander("Generator settings"):
        settings = {}
        cols = st.columns(4)
        for i, (key, default) in enumerate(GEN_DEFAULTS.items()):
            with cols[i % 4]:
                if isinstance(default, float) and default < 10:
                    settings[key] = st.number_input(GEN_LABELS[key], 0.0, 10.0 if key == "log_skew" else 1.0, default, 0.05, key=f"gen_{key}")
                else:
                    settings[key] = int(st.number_input(GEN_LABELS[key], 0, 500, int(default), 1, key=f"gen_{key}"))
    if st.button("Generate candidates and screen them"):
        with st.spinner("Building and screening candidates..."):
            try:
                st.session_state.screening = run_screening(df, y, x_cols, avail, plan, work, raw, settings, recipe)
                st.session_state.screen_version = st.session_state.screening["version"]
            except Exception as e:
                st.error(f"Screening failed: {e}")
    sc = st.session_state.get("screening")
    if not sc:
        return
    if sc["plan_names"] != sorted(step_name(s) for s in plan if step_name(s)):
        st.warning("The plan changed since the last screening. Generate again to screen the current plan.")
    moves = sc["moves"]
    best = moves.iloc[0] if not moves.empty else None
    ui.metric_strip([
        ("Candidates generated", f"{len(sc['gen'])}", f"Built, valid, and not near-duplicates. {sc['cut']} more were cut by the pool cap." if sc["cut"] else "Built, valid, and not near-duplicates."),
        ("Moves tested", f"{len(moves):,}", "Single changes from the Base Data Model: add, remove, or swap one predictor."),
        ("Best single move", f"{best['Δ BIC']:+.1f} BIC" if best is not None else "n/a",
         f"{best['Move']} {best['Variable']}. Lower BIC is better; a drop of more than about 10 is strong evidence." if best is not None else ""),
        ("Training rows used", f"{sc['train_rows']:,}", "The default random-row training split, so the holdout stays unseen."),
    ])

    # --- Candidate table with the deciding numbers beside the tick box ---
    if sc["gen"]:
        st.markdown("**Generated candidates** (best single move first). Tick to add to the plan.")
        cand = pd.DataFrame([{"Add": False, "Candidate": s["name"], "Formula": step_recipe(s), "Family": s["family"],
                              "Δ BIC": s["screen_bic"], "Δ Adj R Square": s["screen_adj"], "t Stat": s["screen_t"], "Why proposed": s["reason"]}
                             for s in sc["gen"]]).sort_values("Δ BIC").reset_index(drop=True)
        picked = st.data_editor(
            cand, key=f"cand_editor_{sc['version']}", hide_index=True,
            disabled=[c for c in cand.columns if c != "Add"],
            column_config={"Δ BIC": st.column_config.NumberColumn(format="%+.1f"),
                           "Δ Adj R Square": st.column_config.NumberColumn(format="%+.4f"),
                           "t Stat": st.column_config.NumberColumn(format="%.2f")},
        )
        ticked = set(picked.loc[picked["Add"], "Candidate"])
        if st.button(f"Add {len(ticked)} ticked candidate(s) to the plan", disabled=not ticked, type="primary"):
            rows = [{"op": s["op"], "col_a": s["cols"][0], "col_b": s["cols"][1] if len(s["cols"]) > 1 else None,
                     "how": s["how"], "reason": f"{s['reason']}; screening Δ BIC {s['screen_bic']:+.1f}", "source": "auto"}
                    for s in sc["gen"] if s["name"] in ticked]
            st.session_state.plan_df = pd.concat([edited, pd.DataFrame(rows, columns=PLAN_COLS, dtype=object)], ignore_index=True)
            st.session_state.plan_version = st.session_state.get("plan_version", 0) + 1
            st.rerun()
    else:
        st.info("No new candidates: everything the generator would propose is already in the plan, invalid, or a near-duplicate.")

    with st.expander("Screening: every single move, including your plan's columns and removing base predictors"):
        st.dataframe(moves.style.format({"Δ BIC": "{:+.1f}", "Δ Adj R Square": "{:+.4f}", "t Stat": "{:.2f}"}, na_rep="-")
                     .background_gradient(subset=["Δ BIC"], cmap="Oranges", gmap=-moves["Δ BIC"].clip(upper=0)),
                     hide_index=True)
        st.markdown("**How to read this:** each row changes one thing in your Base Data Model and refits on the training rows. "
                    "**Δ BIC** below 0 means the change helps after charging for complexity (below about −10 is strong evidence); "
                    "**Δ Adj R Square** is the same idea on the class scale. **t Stat** is the new term's t (for Remove, the removed term's t in the base model); "
                    "beyond about ±2 means the term is unlikely to be pure chance. A move that helps alone may add little once others are in, "
                    "which is why Stage 5 tests combinations.")
        if not sc["pairs"].empty:
            st.markdown("**Best pairs of additions**")
            st.dataframe(sc["pairs"].style.format({"Δ BIC": "{:+.1f}", "Δ Adj R Square": "{:+.4f}", "Together vs separately": "{:+.1f}"}),
                         hide_index=True)
            st.caption("**Together vs separately** below 0 means the two help more as a pair than their single gains add up to; above 0 means they overlap.")


# ---------- Stage 4: Feature workshop ----------
OPS = ["log", "square", "interaction", "combine", "drop"]
HOWS = ["sum", "mean", "ratio", "difference"]
PLAN_COLS = ["op", "col_a", "col_b", "how", "reason", "source"]


def reset_features():
    """Clear the feature plan (new file, new cleaning, or a new Base Data Model)."""
    for k in ["plan_df", "suggested", "feature_work", "screening"]:
        st.session_state.pop(k, None)
    st.session_state.plan_version = st.session_state.get("plan_version", 0) + 1
    c = st.session_state.get("clean")
    if c:
        c.pop("feature_table", None)
        for k in ("feature_plan", "pool_columns", "engineered_columns", "candidates_generated", "screening_summary"):
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
    return {"op": op, "cols": cols, "how": val("how"), "reason": val("reason") or "", "source": val("source") or "user"}


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
            "how": how, "reason": s.get("reason", ""), "source": "claude",
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
        column_config={
            "op": st.column_config.SelectboxColumn("Operation", options=OPS),
            "col_a": st.column_config.SelectboxColumn("Column A", options=[str(c) for c in avail]),
            "col_b": st.column_config.SelectboxColumn("Column B (interaction/combine)", options=[str(c) for c in avail]),
            "how": st.column_config.SelectboxColumn("How (combine only)", options=HOWS),
            "reason": st.column_config.TextColumn("Reason"),
            "source": st.column_config.TextColumn("Source", disabled=True, help="claude, user, or auto (the candidate generator)"),
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
        st.dataframe(log_df, hide_index=True)
    elif log:
        with st.expander(f"All {ok_n} step(s) built successfully"):
            st.dataframe(log_df, hide_index=True)

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
    show_candidates(df, y, x_cols, avail, plan, edited, work, result["raw"], recipe)
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
                      "search_skipped", "search_benchmarks", "leaderboard", "search_starts", "holdout",
                      "final_model", "final_report", "search_score", "cv", "luck_benchmark",
                      "search_locked", "search_excluded", "search_weights", "search_rules",
                      "batch_learning", "bootstrap_stability")


def reset_search():
    """Clear Stage 5 results and refresh the pool picker."""
    st.session_state.pop("search", None)
    st.session_state.pop("final_report", None)
    st.session_state.pop("excel_file", None)
    st.session_state.pop("var_states", None)
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


SEARCH_SCORE = "bic"   # "bic" (lower is better) or "adj_r2"; the search ranks models by this
CV_FOLDS = 5           # cross-validation folds for the leaderboard models
CV_SEED = 42


def bic_value(n, sse, k):
    """BIC = n*ln(SSE/n) + (k+1)*ln(n). One formula for the fast scorer and statsmodels, so they match exactly."""
    return n * math.log(sse / n) + (k + 1) * math.log(n)


def top_row(rank, m, names):
    """One leaderboard row from a fitted statsmodels model."""
    return {"Rank": rank,
            "Adjusted R Square": float(m.rsquared_adj),
            "R Square": float(m.rsquared),
            "Standard Error": float(np.sqrt(m.mse_resid)),
            "Predictors": len(names),
            "BIC": bic_value(int(m.nobs), float(m.ssr), int(m.df_model)),
            "Variables": ", ".join(map(str, names))}


def add_bic_gap(top):
    """BIC Gap = this model's BIC minus the lowest BIC on the leaderboard (above about 10 is strong evidence of worse)."""
    best = min(r["BIC"] for r in top)
    for r in top:
        r["BIC Gap"] = r["BIC"] - best


def make_folds(index, raw, entity_col, k=CV_FOLDS, seed=CV_SEED):
    """Fold number for every training row. Whole entities share a fold when entity_col is given."""
    rng = np.random.default_rng(seed)
    if entity_col:
        ents = raw.loc[index, entity_col].astype(str).to_numpy()
        uniq = rng.permutation(np.unique(ents))
        k = max(2, min(k, len(uniq)))
        fold_of = {e: i % k for i, e in enumerate(uniq)}
        return np.array([fold_of[e] for e in ents]), k
    k = max(2, min(k, len(index)))
    return rng.permutation(len(index)) % k, k


def attach_cv(res, train, y, raw, entity_col):
    """Cross-validate every leaderboard model on the TRAINING rows only. Adds CV RMSE and its standard error."""
    folds, k = make_folds(train.index, raw, entity_col)
    for row in res["top"]:
        names = row["Variables"].split(", ")
        if not all(n in train.columns for n in names):
            continue
        rmses = []
        for f in range(k):
            tr, te = train[folds != f], train[folds == f]
            if len(te) == 0 or len(tr) <= len(names) + 1:
                continue
            rmses.append(fit_and_score(tr, te, y, names)["holdout_rmse"])
        if len(rmses) >= 2:
            row["CV RMSE"] = float(np.mean(rmses))
            row["CV RMSE SE"] = float(np.std(rmses, ddof=1) / np.sqrt(len(rmses)))
    res["cv_info"] = {"folds": k, "grouped_by": entity_col or "random rows",
                      "mean_guess_rmse": float(train[y].std(ddof=0))}


def attach_holdout(res, work, df, mask, y, base_cols, entity_col):
    """Add holdout columns to the leaderboard and score the baseline. Fit on train, score on holdout. Also adds CV."""
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
    attach_cv(res, train, y, st.session_state.clean["raw"], entity_col)


# ---------- Human checkpoint and steering (U4) ----------
VAR_STATES = ["Lock", "Favor", "Neutral", "Exclude"]
FAVOR_MULT = 2.5   # Favor = drawn this many times as often as Neutral in random draws


def column_info(pool, base_cols, plan):
    """name -> family, role (parent / version / square), parents (for the hierarchy rule) and source, for every pool column."""
    steps = {step_name(s): s for s in plan if step_name(s)}
    info = {}
    for c in pool:
        s = steps.get(c)
        if s is None:
            info[c] = {"family": str(c), "role": "parent", "parents": [], "source": "base" if c in base_cols else "data"}
            continue
        fam = family_of(s)
        role = "square" if s["op"] == "square" else ("parent" if fam == c else "version")
        parents = [str(p) for p in s["cols"]] if s["op"] in ("square", "interaction") else []
        info[c] = {"family": fam, "role": role, "parents": parents, "source": s.get("source") or "user"}
    return info


def make_rule(cols, info, one_per_family, hierarchy):
    """ok(indices, hier=True) -> True when a model obeys the rules, or None when no rule is on.
    Family rule: one version per family (x, ln1p_x, x_sq), except x_sq beside its parent x.
    Hierarchy rule: a square or interaction needs its parent columns (those still in the pool)."""
    if not (one_per_family or hierarchy):
        return None
    fam = [info[c]["family"] for c in cols]
    role = [info[c]["role"] for c in cols]
    where = {c: i for i, c in enumerate(cols)}
    parents = [[where[p] for p in info[c]["parents"] if p in where] for c in cols]

    def ok(idx, hier=True):
        s = set(idx)
        if one_per_family:
            seen = {}
            for i in s:
                seen.setdefault(fam[i], []).append(role[i])
            for roles in seen.values():
                if len(roles) > 1 and sorted(roles) != ["parent", "square"]:
                    return False
        if hierarchy and hier:
            return all(p in s for i in s for p in parents[i])
        return True

    return ok


def build_steer(cols, states, info, mult, one_per_family, hierarchy):
    """Everything the search needs to follow the checkpoint: locked indices, draw weights, the rule, and a plain record."""
    where = {c: i for i, c in enumerate(cols)}
    return {"locked": tuple(i for i, c in enumerate(cols) if states.get(c) == "Lock"),
            "parents": [[where[p] for p in info[c]["parents"] if p in where] for c in cols],
            "weights": np.array([float(mult) if states.get(c) == "Favor" else 1.0 for c in cols]),
            "ok": make_rule(cols, info, one_per_family, hierarchy),
            "cfg": {"locked": [c for c in cols if states.get(c) == "Lock"],
                    "favored": [c for c in cols if states.get(c) == "Favor"],
                    "excluded": sorted(c for c, v in states.items() if v == "Exclude"),
                    "favor_multiplier": float(mult), "one_version_per_family": bool(one_per_family),
                    "hierarchy": bool(hierarchy)}}


def repair(idx, steer, max_size):
    """Turn a start into one that obeys the steering: locked columns first, then the rest in the given order,
    each kept only if the family rule still holds. Then add missing parents (hierarchy) or drop the child."""
    locked = list(steer["locked"]) if steer else []
    ok = steer["ok"] if steer else None
    keep = []
    for i in locked + [i for i in idx if i not in locked]:
        if i not in keep and len(keep) < max_size and (ok is None or ok(keep + [i], hier=False)):
            keep.append(i)
    if ok is not None and not ok(keep):
        for i in list(keep):
            if i not in keep:
                continue
            need = [p for p in steer["parents"][i] if p not in keep]
            if not need:
                continue
            if len(keep) + len(need) <= max_size and ok(keep + need, hier=False):
                keep = keep + need
            elif i not in locked:
                keep.remove(i)
    if not keep or (ok is not None and not ok(keep)):
        return None
    return tuple(sorted(keep))


def count_subsets(k, max_size, n_locked=0):
    """Number of models up to max_size when n_locked columns are in every model (before the family/hierarchy rules)."""
    free, room = k - n_locked, max_size - n_locked
    return sum(math.comb(free, s) for s in range(0 if n_locked else 1, min(room, free) + 1))


def draw_subsets(k, max_size, n, steer, rng):
    """n distinct random models. Favored columns are drawn more often; locked columns are always in; rule-breakers are redrawn."""
    locked = tuple(steer["locked"]) if steer else ()
    ok = steer["ok"] if steer else None
    free = [i for i in range(k) if i not in locked]
    room = min(max_size - len(locked), len(free))
    lo = 0 if locked else 1
    p = None
    if steer is not None and free:
        w = steer["weights"][free]
        p = w / w.sum()
    seen, tries = set(), 0
    while len(seen) < n and tries < n * 20 and room >= lo:
        tries += 1
        size = int(rng.integers(lo, room + 1))
        pickd = rng.choice(free, size=size, replace=False, p=p).tolist() if size else []
        t = tuple(sorted(set(pickd) | set(locked)))
        if t and (ok is None or ok(t)):
            seen.add(t)
    return seen


def make_subsets(k, max_size, sample_n, must=(), steer=None):
    """Index tuples to test. All if few enough, otherwise a (weighted) random sample plus anchor models. Obeys the steering."""
    locked = tuple(steer["locked"]) if steer else ()
    ok = steer["ok"] if steer else None
    if len(locked) > max_size:
        raise ValueError("More variables are locked than the model-size limit allows.")
    free = [i for i in range(k) if i not in locked]
    room = min(max_size - len(locked), len(free))

    def good(t):
        return 0 < len(t) <= max_size and (ok is None or ok(t))

    if count_subsets(k, max_size, len(locked)) <= FULL_LIMIT:
        subs = (tuple(sorted(locked + c)) for s in range(0 if locked else 1, room + 1) for c in combinations(free, s))
        return [t for t in subs if good(t)], "all"
    seen = draw_subsets(k, max_size, sample_n, steer, np.random.default_rng(SEARCH_SEED))
    anchors = [tuple(range(k))] + [tuple(i for i in range(k) if i != j) for j in range(k)] + [tuple(m) for m in must]
    for a in anchors:
        r = repair(a, steer, max_size) if steer else tuple(sorted(a))
        if r and good(r):
            seen.add(r)
    return sorted(seen), "sample"


def score_subsets(df, y, cols, subsets, progress=None):
    """Fast Adjusted R Square and BIC for each subset (same OLS answer, tiny matrix solves)."""
    n = len(df)
    X = df[cols].to_numpy(dtype=float)
    sd = X.std(axis=0)
    sd[sd == 0] = 1.0  # a column that is constant in these rows becomes all zeros and is skipped as singular
    Z = (X - X.mean(axis=0)) / sd
    yc = df[y].to_numpy(dtype=float)
    yc = yc - yc.mean()
    G, c, sst = Z.T @ Z, Z.T @ yc, float(yc @ yc)
    keep, adj_list, bic_list = [], [], []
    for i, idx in enumerate(subsets):
        idx = list(idx)
        k = len(idx)
        Gs = G[np.ix_(idx, idx)]
        if n - k - 1 <= 0 or np.linalg.cond(Gs) > 1e10:
            continue  # singular or too many predictors: skip
        expl = float(np.linalg.solve(Gs, c[idx]) @ c[idx])
        sse = sst - expl
        if sse <= 1e-12 * sst:
            continue  # a perfect fit makes BIC undefined: skip
        keep.append(idx)
        adj_list.append(1 - (1 - expl / sst) * (n - 1) / (n - k - 1))
        bic_list.append(bic_value(n, sse, k))
        if progress is not None and i % 500 == 0:
            progress.progress(min(i / len(subsets), 1.0))
    return keep, np.array(adj_list), len(subsets) - len(keep), np.array(bic_list)


def run_search(df, y, cols, max_size, sample_n, progress=None, must=(), steer=None):
    subsets, mode = make_subsets(len(cols), max_size, sample_n, must, steer)
    keep, adj, skipped, bic = score_subsets(df, y, cols, subsets, progress)
    if len(adj) == 0:
        raise ValueError("No valid models could be fitted from this pool.")
    rank = -bic if SEARCH_SCORE == "bic" else adj  # higher rank value = better model
    order = np.argsort(-rank)
    top = []
    for r, j in enumerate(order[:TOP_N], start=1):
        names = [cols[i] for i in keep[j]]
        top.append(top_row(r, run_regression(df, y, names), names))  # exact statsmodels numbers
    add_bic_gap(top)
    fast = float(rank[order[0]])
    exact = -top[0]["BIC"] if SEARCH_SCORE == "bic" else top[0]["Adjusted R Square"]
    check_ok = abs(fast - exact) < 1e-6 * max(1.0, abs(fast))
    return {"mode": mode, "models_run": len(keep), "skipped": skipped, "scores": adj,
            "top": top, "pool": list(cols), "max_size": max_size, "check_ok": check_ok,
            "score_name": SEARCH_SCORE}


def make_scorer(df, y, cols):
    """Returns (score, cache, adj_cache). score(indices) -> ranking value (higher is better) or None if unfittable."""
    n = len(df)
    X = df[cols].to_numpy(dtype=float)
    sd = X.std(axis=0)
    sd[sd == 0] = 1.0  # a column that is constant in these rows becomes all zeros and is skipped as singular
    Z = (X - X.mean(axis=0)) / sd
    yc = df[y].to_numpy(dtype=float)
    yc = yc - yc.mean()
    G, c, sst = Z.T @ Z, Z.T @ yc, float(yc @ yc)
    cache, adj_cache = {}, {}

    def score(idx):
        key = tuple(sorted(idx))
        if key in cache:
            return cache[key]
        k, val = len(key), None
        if k > 0 and n - k - 1 > 0:
            Gs = G[np.ix_(key, key)]
            if np.linalg.cond(Gs) <= 1e10:
                ck = c[list(key)]
                expl = float(np.linalg.solve(Gs, ck) @ ck)
                sse = sst - expl
                if sse > 1e-12 * sst:
                    adj = 1 - (1 - expl / sst) * (n - 1) / (n - k - 1)
                    adj_cache[key] = adj
                    val = -bic_value(n, sse, k) if SEARCH_SCORE == "bic" else adj
        cache[key] = val
        return val

    return score, cache, adj_cache


def hill_climb(score, k, start, max_size, steer=None):
    """Steepest ascent: apply the best single add, remove, or swap until nothing improves.
    Every legal move is tested each step, so weights do not change the climb; locks and rules do."""
    locked = set(steer["locked"]) if steer else set()
    ok = steer["ok"] if steer else None
    cur = tuple(sorted(start))
    cur_s = score(cur)
    while True:
        out = [i for i in range(k) if i not in cur]
        movable = [j for j in cur if j not in locked]
        nbrs = []
        if len(cur) < max_size:
            nbrs += [cur + (i,) for i in out]
        if len(cur) > 1:
            nbrs += [tuple(x for x in cur if x != j) for j in movable]
        nbrs += [tuple(i if x == j else x for x in cur) for j in movable for i in out]
        best, best_s = None, cur_s
        for nb in nbrs:
            if ok is not None and not ok(nb):
                continue
            s = score(nb)
            if s is not None and s > best_s + 1e-9:
                best, best_s = tuple(sorted(nb)), s
        if best is None:
            return cur, cur_s
        cur, cur_s = best, best_s


def run_hybrid(df, y, cols, max_size, n_random, n_starts, must=(), progress=None, steer=None, plan_start=None):
    """(Weighted) random scouting, then hill-climb from the baseline, your plan, the full pool, and the best scouts."""
    k = len(cols)
    if steer and len(steer["locked"]) > max_size:
        raise ValueError("More variables are locked than the model-size limit allows.")
    score, cache, adj_cache = make_scorer(df, y, cols)
    scouts = draw_subsets(k, max_size, n_random, steer, np.random.default_rng(SEARCH_SEED))
    ranked = sorted((score(s), s) for s in scouts if score(s) is not None)[::-1]

    fix = (lambda t: repair(t, steer, max_size)) if steer else (lambda t: tuple(sorted(t)))
    starts = [("Baseline", fix(m)) for m in must]
    if plan_start:
        starts.append(("Your plan", fix(plan_start)))
    if k <= max_size:
        starts.append(("Full pool", fix(tuple(range(k)))))
    starts += [(f"Random #{i}", s) for i, (_, s) in enumerate(ranked[:n_starts], 1)]
    used, uniq = set(), []
    for label, s0 in starts:
        if s0 and s0 not in used:
            used.add(s0)
            uniq.append((label, s0))

    rows, ends = [], set()
    for i, (label, s0) in enumerate(uniq):
        if score(s0) is None:
            continue
        end, _ = hill_climb(score, k, s0, max_size, steer)
        ends.add(end)
        rows.append({"Start": label, "Start Adj R Square": adj_cache[tuple(sorted(s0))],
                     "End Adj R Square": adj_cache[end],
                     "Predictors": len(end), "Variables": ", ".join(str(cols[j]) for j in end)})
        if progress is not None:
            progress.progress((i + 1) / len(uniq))

    ok = steer["ok"] if steer else None
    valid = {key: v for key, v in cache.items() if v is not None and (ok is None or ok(key))}
    if not valid:
        raise ValueError("No valid models could be fitted from this pool.")
    order = sorted(valid, key=valid.get, reverse=True)
    top = []
    for r, key in enumerate(order[:TOP_N], start=1):
        names = [cols[i] for i in key]
        top.append(top_row(r, run_regression(df, y, names), names))
    add_bic_gap(top)
    fast = valid[order[0]]
    exact = -top[0]["BIC"] if SEARCH_SCORE == "bic" else top[0]["Adjusted R Square"]
    check_ok = abs(fast - exact) < 1e-6 * max(1.0, abs(fast))
    return {"mode": "hybrid", "models_run": len(valid), "skipped": len(cache) - len(valid),
            "scores": np.array([adj_cache[key] for key in valid]), "top": top, "pool": list(cols),
            "max_size": max_size, "check_ok": check_ok, "starts": rows, "distinct_ends": len(ends),
            "score_name": SEARCH_SCORE}


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
    recipe["search_score"] = res.get("score_name", SEARCH_SCORE)
    recipe["cv"] = res.get("cv_info")
    if res.get("learning"):
        recipe["batch_learning"] = learning_summary(res["learning"])
    else:
        recipe.pop("batch_learning", None)
    cfg = (res.get("steer") or {}).get("cfg")
    if cfg:
        recipe["search_locked"] = cfg["locked"]
        recipe["search_excluded"] = cfg["excluded"]
        recipe["search_weights"] = {"favored": cfg["favored"], "favor_multiplier": cfg["favor_multiplier"], "everything_else": 1.0}
        recipe["search_rules"] = {"one_version_per_family": cfg["one_version_per_family"], "hierarchy": cfg["hierarchy"]}


# ---------- Luck benchmark (U1-B) ----------
LUCK_REPEATS = 5   # shuffled-Y repeats
LUCK_SEED = 1000   # permutation seeds are LUCK_SEED + repeat number


def model_bic(df, y, cols):
    """(BIC, Adjusted R Square) of one OLS fit, using the same BIC formula as the search."""
    m = run_regression(df, y, cols)
    return bic_value(int(m.nobs), float(m.ssr), int(m.df_model)), float(m.rsquared_adj)


def luck_benchmark(res, train, base_df, y, base_cols, repeats, progress=None):
    """Re-run the identical search on training rows with Y shuffled. Gain = BIC of a no-predictor model
    minus the best BIC found, so real and shuffled gains measure the same thing: how much the search
    can 'explain' beyond guessing the average. The shuffled searches use the same locks, weights and rules,
    and a learning search learns on the shuffled data too (plus a learning-off run for comparison)."""
    learning = res["settings"]["kind"] == "learn"

    def null_gains(r, d):
        yv = d[y].to_numpy(dtype=float)
        sst = float(((yv - yv.mean()) ** 2).sum())
        return bic_value(len(d), sst, 0) - min(t["BIC"] for t in r["top"]), float(r["scores"].max())

    real_bic, real_adj = null_gains(res, train)
    base_bic, _ = model_bic(base_df, y, base_cols)
    vs_base = base_bic - min(t["BIC"] for t in res["top"])
    sh_bic, sh_adj, sh_off = [], [], []
    for i in range(repeats):
        perm = np.random.default_rng(LUCK_SEED + i).permutation(train[y].to_numpy())
        tr = train.copy()
        tr[y] = perm
        g_bic, g_adj = null_gains(rerun_search(res, tr, y, base_cols), tr)
        sh_bic.append(float(g_bic))
        sh_adj.append(g_adj)
        if learning and res["settings"]["learn"]:
            sh_off.append(float(null_gains(rerun_search(res, tr, y, base_cols, learn=False), tr)[0]))
        if progress is not None:
            progress.progress((i + 1) / repeats)
    out = {"real_gain": float(real_bic), "shuffled_gains": sh_bic,
           "real_gain_adj": float(real_adj), "shuffled_gains_adj": sh_adj,
           "real_gain_vs_baseline": float(vs_base), "repeats": repeats}
    if sh_off:
        out["shuffled_gains_learning_off"] = sh_off
    return out


# ---------- Batch learning and bootstrap stability (U5) ----------
LEARN_BATCH = 500        # models per batch
LEARN_ROUNDS = 5         # batches
LEARN_ETA = 0.3          # how far each round moves a weight toward the new evidence
LEARN_TOP = 0.10         # the top share of a batch that counts as "the best models"
LEARN_CLAMP = (0.25, 4.0)
LEARN_CLIMBS = 5        # after the last batch, hill-climb from the baseline and this many best models found
BOOT_RUNS = 20           # resampled re-runs in the stability check
BOOT_SEED = 2000


LEARN_SCALE = 5.0 if SEARCH_SCORE == "bic" else 0.001  # a gain this size (BIC points, or Adj R Square) doubles the target weight


def marginal_gain(score, models, i, max_size, steer):
    """Average score gain from having variable i in rather than out of each model (add it if absent, drop it if
    present), skipping changes that break a lock, the size limit or the rules. None when no change was legal."""
    ok, locked, gains = steer["ok"], set(steer["locked"]), []
    for s in models:
        with_i = tuple(sorted(set(s) | {i}))
        without = tuple(x for x in s if x != i)
        if len(with_i) > max_size or not without or (locked - set(without)):
            continue
        if ok is not None and not (ok(with_i) and ok(without)):
            continue
        a, b = score(with_i), score(without)
        if a is not None and b is not None:
            gains.append(a - b)
    return float(np.mean(gains)) if gains else None


def plain_steer(k):
    """No locks, equal weights, no rules: what the search does when nobody steers it."""
    return {"locked": (), "weights": np.ones(k), "ok": None, "parents": [[] for _ in range(k)], "cfg": None}


def run_learning(df, y, cols, max_size, batch, rounds, learn=True, must=(), progress=None, steer=None):
    """Batches of (weighted) random models. After each batch, every learnable variable's weight moves toward
    2 ** (g / LEARN_SCALE), where g = its average score gain from being in rather than out of the batch's best
    LEARN_TOP models (marginal_gain): new = (1 - LEARN_ETA) * old + LEARN_ETA * target, clamped to LEARN_CLAMP.
    (The plan's "share among the top models" signal was tried first: it rewarded every variable, because the best
    models are simply bigger, and it could not raise a variable that was already drawn often.) Locked and user-favored
    variables keep the user's setting. With learn=False the weights never change (same batches, no learning)."""
    k = len(cols)
    steer = steer or plain_steer(k)
    if len(steer["locked"]) > max_size:
        raise ValueError("More variables are locked than the model-size limit allows.")
    favored = set((steer.get("cfg") or {}).get("favored", []))
    learnable = [i for i in range(k) if i not in steer["locked"] and cols[i] not in favored]
    shown = [i for i in range(k) if i not in steer["locked"]]
    score, cache, adj_cache = make_scorer(df, y, cols)
    base_starts = []
    for m in must:  # the baseline is always scored, so the leaderboard can be compared with it
        r = repair(m, steer, max_size) if steer["ok"] or steer["locked"] else tuple(sorted(m))
        if r and len(r) <= max_size:
            score(r)
            base_starts.append(r)
    w = steer["weights"].astype(float).copy()
    rng = np.random.default_rng(SEARCH_SEED)
    history = [{"Round": 0, **{str(cols[i]): float(w[i]) for i in shown}}]
    best_by_round = []
    for rd in range(1, rounds + 1):
        sets = draw_subsets(k, max_size, batch, dict(steer, weights=w), rng)
        scored = sorted(((score(s), s) for s in sets if score(s) is not None), reverse=True)
        if scored:
            best_by_round.append(-scored[0][0] if SEARCH_SCORE == "bic" else scored[0][0])
        if learn and len(scored) >= 10:
            top = [s for _, s in scored[:max(10, int(len(scored) * LEARN_TOP))]]
            for i in learnable:
                g = marginal_gain(score, top, i, max_size, steer)
                if g is None:
                    continue  # no legal way to toggle it in these models: keep its weight
                target = 2.0 ** float(np.clip(g / LEARN_SCALE, -2, 2))  # +LEARN_SCALE -> 2x, -LEARN_SCALE -> 0.5x
                w[i] = float(np.clip((1 - LEARN_ETA) * w[i] + LEARN_ETA * target, *LEARN_CLAMP))
        history.append({"Round": rd, **{str(cols[i]): float(w[i]) for i in shown}})
        if progress is not None:
            progress.progress(rd / rounds)

    # Finish like the riff search: climb (best single add / remove / swap) from the baseline and the best models
    # found, so the learning search cannot end just short of a peak the plain or riff search would reach.
    ok = steer["ok"]
    found = sorted((v, key) for key, v in cache.items() if v is not None and (ok is None or ok(key)))[::-1]
    starts = [("Baseline", b) for b in base_starts] + [(f"Learned #{j}", key) for j, (_, key) in enumerate(found[:LEARN_CLIMBS], 1)]
    rows, ends, used = [], set(), set()
    for label, s0 in starts:
        if s0 in used or score(s0) is None:
            continue
        used.add(s0)
        end, _ = hill_climb(score, k, s0, max_size, steer)
        ends.add(end)
        rows.append({"Start": label, "Start Adj R Square": adj_cache[s0], "End Adj R Square": adj_cache[end],
                     "Predictors": len(end), "Variables": ", ".join(str(cols[j]) for j in end)})

    valid = {key: v for key, v in cache.items() if v is not None and (ok is None or ok(key))}
    if not valid:
        raise ValueError("No valid models could be fitted from this pool.")
    order = sorted(valid, key=valid.get, reverse=True)
    top_rows = []
    for r, key in enumerate(order[:TOP_N], start=1):
        names = [cols[i] for i in key]
        top_rows.append(top_row(r, run_regression(df, y, names), names))
    add_bic_gap(top_rows)
    fast = valid[order[0]]
    exact = -top_rows[0]["BIC"] if SEARCH_SCORE == "bic" else top_rows[0]["Adjusted R Square"]
    return {"mode": "learning", "models_run": len(valid), "skipped": len(cache) - len(valid),
            "scores": np.array([adj_cache[key] for key in valid]), "top": top_rows, "pool": list(cols),
            "max_size": max_size, "check_ok": abs(fast - exact) < 1e-6 * max(1.0, abs(fast)),
            "score_name": SEARCH_SCORE, "starts": rows, "distinct_ends": len(ends),
            "learning": {"history": history, "rounds": rounds, "batch": batch, "learn": learn,
                         "best_by_round": best_by_round, "fixed": sorted(favored)}}


def rerun_search(res, data, y, base_cols, learn=None):
    """Repeat the search that produced `res` (same kind, settings, pool, size limit and steering) on other rows."""
    cfg, pick, max_size, steer = res["settings"], res["pool"], res["max_size"], res.get("steer")
    must = [[pick.index(c) for c in base_cols]] if all(c in pick for c in base_cols) else []
    if cfg["kind"] == "riff":
        return run_hybrid(data, y, pick, max_size, cfg["n_rand"], cfg["n_starts"], must, None, steer, cfg.get("plan_start"))
    if cfg["kind"] == "learn":
        return run_learning(data, y, pick, max_size, cfg["batch"], cfg["rounds"],
                            cfg["learn"] if learn is None else learn, must, None, steer)
    return run_search(data, y, pick, max_size, cfg["sample_n"], None, must, steer)


def learning_summary(ln):
    """Compact record of what the learning search learned (for the recipe and the report)."""
    final = ln["history"][-1]
    wts = {k: v for k, v in final.items() if k != "Round" and k not in ln["fixed"]}
    s = pd.Series(wts, dtype=float).sort_values(ascending=False)
    return {"rounds": ln["rounds"], "batch": ln["batch"], "learning_on": ln["learn"], "eta": LEARN_ETA,
            "top_share": LEARN_TOP, "signal": "average BIC change with the variable in vs out of the batch best models", "scale": LEARN_SCALE, "clamp": list(LEARN_CLAMP), "kept_at_user_favor": ln["fixed"],
            "raised_most": {k: round(v, 2) for k, v in s[s > 1].head(8).items()},
            "lowered_most": {k: round(v, 2) for k, v in s[s < 1].tail(8).items()},
            "best_score_by_round": [round(b, 1) for b in ln["best_by_round"]]}


def bootstrap_stability(res, train, y, raw, entity_col, base_cols, runs, progress=None):
    """Re-run the same search on `runs` resampled copies of the training rows (whole entities resampled when an
    entity column is set). For each variable: share of runs whose winner contains it, and share whose top 5 do."""
    ents = raw.loc[train.index, entity_col].astype(str).to_numpy() if entity_col else None
    groups = {e: np.flatnonzero(ents == e) for e in np.unique(ents)} if entity_col else None
    wins, top5, winners, done = {}, {}, set(), 0
    for b in range(runs):
        rng = np.random.default_rng(BOOT_SEED + b)
        if entity_col:
            keys = list(groups)
            idx = np.concatenate([groups[keys[j]] for j in rng.integers(0, len(keys), len(keys))])
        else:
            idx = rng.integers(0, len(train), len(train))
        sample = train.iloc[idx].reset_index(drop=True)
        try:
            r = rerun_search(res, sample, y, base_cols)
        except ValueError:
            continue
        done += 1
        win = r["top"][0]["Variables"].split(", ")
        winners.add(tuple(sorted(win)))
        for v in win:
            wins[v] = wins.get(v, 0) + 1
        for v in {n for row in r["top"][:5] for n in row["Variables"].split(", ")}:
            top5[v] = top5.get(v, 0) + 1
        if progress is not None:
            progress.progress((b + 1) / runs)
    if not done:
        raise ValueError("No resampled run could be fitted.")
    freq = {c: {"wins": wins.get(c, 0) / done * 100, "top5": top5.get(c, 0) / done * 100} for c in res["pool"]}
    return {"runs": done, "grouped_by": entity_col or "random rows", "distinct_winners": len(winners), "freq": freq}


def learning_fig(history, fixed, top_n=8):
    """Weight per round for the variables whose weight moved most."""
    h = pd.DataFrame(history).set_index("Round")
    h = h[[c for c in h.columns if c not in fixed]]
    moved = (h.iloc[-1] - 1).abs().sort_values(ascending=False).index[:top_n]
    fig, ax = plt.subplots(figsize=(6, 3.2))
    for c in moved:
        ax.plot(h.index, h[c], marker="o", label=str(c))
    ax.axhline(1, color="#888888", linestyle="--", linewidth=1)
    ax.set_xlabel("Round")
    ax.set_ylabel("Draw weight (1 = neutral)")
    ax.set_title("What the learning search learned")
    ax.legend(fontsize=6, ncol=2)
    fig.tight_layout()
    return fig


def luck_verdict(real, shuffled):
    """(streamlit alert level, plain-English verdict) comparing the real BIC gain with the shuffled gains."""
    worst_luck = max(shuffled)
    if real <= 0:
        return "warning", "**The search found nothing that beats guessing the average.**"
    if real > 10 and real > 1.5 * worst_luck:
        return "success", "**The fit is well beyond luck.** The real gain is much larger than anything the search found on shuffled data."
    if real > worst_luck:
        return "info", "**Beats luck, but not by a wide margin.** Lean on the holdout and CV numbers."
    return "warning", "**Indistinguishable from luck.** The search found gains this large on shuffled data, so this fit could be noise."


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


def single_moves(train, y, base_in, pool):
    """For each pool column, the one change from the Base Data Model on the training rows: add it
    (or remove it, if it is a base predictor). Returns {name: Move, Δ BIC, Δ Adj R Square, t Stat}."""
    cols = list(base_in) + [c for c in pool if c not in base_in]
    fit = fast_stats(train, y, cols)
    base = list(range(len(base_in)))
    b0 = fit(base)
    out = {}
    for j, c in enumerate(cols):
        is_base = j < len(base_in)
        r = fit([b for b in base if b != j]) if is_base else fit(base + [j])
        t = (b0["t"][j] if b0 else None) if is_base else (r["t"][-1] if r else None)
        good = r is not None and b0 is not None
        out[c] = {"Move": "Remove" if is_base else "Add",
                  "Δ BIC": r["bic"] - b0["bic"] if good else np.nan,
                  "Δ Adj R Square": r["adj"] - b0["adj"] if good else np.nan,
                  "t Stat": float(t) if t is not None else np.nan}
    return out


def show_checkpoint(work, train, y, all_pool, base_cols, recipe, entity_col):
    """Lock / Favor / Neutral / Exclude per variable, with its one-change evidence beside the control.
    Returns (pick, steer, info): the columns the search may use, the steering, and the family info."""
    info = column_info(all_pool, base_cols, recipe.get("feature_plan", []))
    states = st.session_state.setdefault("var_states", {})
    for c in all_pool:
        states.setdefault(c, "Favor" if info[c]["source"] in ("user", "claude") else "Neutral")

    ui.section("Your checkpoint: steer the search",
               "Set each variable's state with the evidence in view. **Lock** = in every model. **Favor** = drawn more often in random draws. "
               "**Neutral** = normal. **Exclude** = never tried. Nothing is forced in unless you lock it: BIC still judges every model. "
               "Skip this and the defaults apply (your own and Claude's plan columns start as Favor).")
    res = st.session_state.get("search")
    key = f"steer_{st.session_state.get('pool_version', 0)}_{entity_col}_{st.session_state.get('search_runs', 0)}"
    snap = st.session_state.get("steer_table")
    if snap is None or snap[0] != key:  # rebuild only when the evidence changes, so edits are never fed back in
        base_in = [c for c in base_cols if c in all_pool]
        ev = single_moves(train, y, base_in, all_pool)
        share = {}
        if res:
            names = [n for r in res["top"] for n in r["Variables"].split(", ")]
            share = (pd.Series(names).value_counts() / len(res["top"]) * 100).to_dict()
        boot = (res or {}).get("bootstrap")
        rows = [{"State": states[c], "Variable": c, "Source": info[c]["source"], "Family": info[c]["family"], **ev[c],
                 **({"In top models": float(share.get(c, 0.0))} if res else {}),
                 **({"Resample wins": boot["freq"].get(c, {}).get("wins", 0.0)} if boot else {})} for c in all_pool]
        snap = (key, pd.DataFrame(rows))
        st.session_state.steer_table = snap
    table = snap[1]
    edited = st.data_editor(
        table, key=key, hide_index=True,
        disabled=[c for c in table.columns if c != "State"],
        column_config={
            "State": st.column_config.SelectboxColumn("State", options=VAR_STATES, required=True,
                                                      help="Lock = in every model, Favor = drawn more often, Exclude = never tried"),
            "Source": st.column_config.TextColumn(help="base = Base Data Model; user, claude or auto = built in the Stage 4 plan"),
            "Family": st.column_config.TextColumn(help="x, ln1p_x and x_sq share a family"),
            "Move": st.column_config.TextColumn("One change", help="Add it to the Base Data Model, or remove it if it is already in"),
            "Δ BIC": st.column_config.NumberColumn(format="%+.1f", help="Below 0 = that one change helps (below about −10 is strong evidence)"),
            "Δ Adj R Square": st.column_config.NumberColumn(format="%+.4f"),
            "t Stat": st.column_config.NumberColumn(format="%.2f", help="The term's t Stat in that model; beyond about ±2 is unlikely to be chance"),
            "In top models": st.column_config.NumberColumn(format="%.0f%%", help="Share of the last run's leaderboard models that contain it"),
            "Resample wins": st.column_config.NumberColumn(format="%.0f%%", help="Share of the stability check's resampled runs whose winner contains it"),
        },
    )
    for c, v in zip(edited["Variable"], edited["State"]):
        states[c] = v or "Neutral"
    st.caption("The evidence columns are one change at a time from your Base Data Model on these training rows. "
               + ("**In top models** is from the last run: a variable in nearly all of them is robustly useful; re-weight and run again to test that." if res else
                  "After a run, an **In top models** column shows how often each variable made the leaderboard."))

    c1, c2, c3 = st.columns(3)
    mult = c1.slider("Favor multiplier", 1.0, 10.0, FAVOR_MULT, 0.5, key="favor_multiplier",
                     help="How many times as often a Favor variable is drawn in random models. It never changes a model's score.")
    one_fam = c2.checkbox("One version per family", value=True, key="rule_family",
                          help="A model uses x or ln1p_x, not both. x_sq may sit beside x (that pair is how curvature is fit).")
    hier = c3.checkbox("Hierarchy rule", value=False, key="rule_hierarchy",
                       help="A square or interaction is only tried when its parent columns are also in the model.")
    pick = [c for c in all_pool if states[c] != "Exclude"]
    steer = build_steer(pick, states, info, mult, one_fam, hier)
    n = {s: sum(states[c] == s for c in all_pool) for s in VAR_STATES}
    st.caption(f"{n['Lock']} locked, {n['Favor']} favored, {n['Neutral']} neutral, {n['Exclude']} excluded. "
               "Weights only change how often a variable is drawn in sampled and riff searches; a full search tries every allowed model anyway.")
    return pick, steer, info


def show_stage_5():
    result = st.session_state.clean
    if result is None or "feature_table" not in result:
        st.info("Approve the feature plan in Stage 4 first.")
        return
    work, recipe = result["feature_table"], result["recipe"]
    y = recipe["target"]
    all_pool = recipe["pool_columns"]

    ui.section("Pool & Search", "The app fits many models from different subsets of your pool and ranks them by BIC (a fit score that charges for extra predictors). Adjusted R Square stays on screen so numbers match Excel. Python does the search; you steer it and decide what the results mean.")
    strip = st.container()  # filled in below, once the controls are set

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

    # --- Checkpoint (replaces the old pool picker: Exclude removes a variable from the pool) ---
    pick, steer, info = show_checkpoint(work, train, y, all_pool, base_cols, recipe, entity_col)
    k, n_locked = len(pick), len(steer["locked"])
    if k == 0:
        st.warning("Every variable is excluded. Set at least one to Neutral, Favor or Lock.")
        return

    # --- Search settings ---
    cap = min(k, len(train) - 2)
    if n_locked > cap:
        st.error(f"{n_locked} variables are locked, more than the largest model allowed ({cap}). Unlock some.")
        return
    lo = max(1, n_locked)
    max_size = st.slider("Largest model to test (number of predictors)", lo, cap, cap) if cap > lo else cap
    total = count_subsets(k, max_size, n_locked)
    rules_on = steer["ok"] is not None
    sample_n = 0
    if total <= FULL_LIMIT:
        st.write(f"{total:,} possible models, so **every** one will be run" + (" (minus those the family or hierarchy rule forbids)." if rules_on else "."))
    else:
        st.write(f"{total:,} possible models, too many to run all, so a **random sample** will be tested" + (", favored variables drawn more often." if steer["cfg"]["favored"] else "."))
        sample_n = st.number_input("Sample size", 500, 50000, DEFAULT_SAMPLE, step=500)

    with strip:
        ui.metric_strip([
            ("Variables in pool", f"{k}", f"Predictors the search can choose from ({n_locked} locked into every model)." if n_locked else "Predictors the search can choose from."),
            ("Training rows", f"{len(train):,}", "The only rows the search sees."),
            ("Holdout rows", f"{len(test):,}", "Set aside to test the winner on unseen data."),
            ("Possible models", f"{total:,}",
             "Every subset up to the size limit, with locked variables always in." if total <= FULL_LIMIT else "Too many to run all, so a random sample is tested."),
        ])

    # --- Run buttons ---
    base_in_pool = all(c in pick for c in base_cols)
    must = [[pick.index(c) for c in base_cols]] if base_in_pool else []
    baseline_testable = base_in_pool and repair(must[0], steer, max_size) == tuple(sorted(must[0]))
    plan_cols = [c for c in pick if info[c]["source"] not in ("base", "data")]
    plan_start = ([pick.index(c) for c in plan_cols] + [pick.index(c) for c in base_cols if c in pick]) if plan_cols else None

    def finish(res, settings):
        res["baseline_testable"] = baseline_testable
        res["settings"] = settings
        res["steer"] = steer
        attach_holdout(res, work, df_clean, mask, y, base_cols, entity_col)
        store_search(res, pick, max_size, recipe)
        recipe.pop("luck_benchmark", None)  # a new search invalidates any earlier luck run
        recipe.pop("bootstrap_stability", None)
        st.session_state.search_runs = st.session_state.get("search_runs", 0) + 1

    ran = False
    if st.button("Run search", type="primary"):
        bar = st.progress(0.0)
        try:
            finish(run_search(train, y, pick, max_size, int(sample_n), bar, must, steer),
                   {"kind": "search", "sample_n": int(sample_n)})
            ran = True
        except Exception as e:
            st.error(f"Search failed: {e}")
        bar.empty()

    with st.expander("Riff search (optional): scout with random models, then improve the best starting points one change at a time"):
        r1, r2 = st.columns(2)
        n_rand = r1.number_input("Random models to scout starting points", 200, 20000, 2000, step=200)
        n_starts = r2.number_input("Random starting points to riff from", 0, 50, 10)
        st.caption("Climbs start from the baseline, " + ("**Your plan** (your Stage 4 columns plus the base), " if plan_start else "")
                   + "the full pool when it fits, and the best random scouts. Locked variables stay in every climb.")
        if st.button("Run riff search"):
            bar = st.progress(0.0)
            try:
                finish(run_hybrid(train, y, pick, max_size, int(n_rand), int(n_starts), must, bar, steer, plan_start),
                       {"kind": "riff", "n_rand": int(n_rand), "n_starts": int(n_starts), "plan_start": plan_start})
                ran = True
            except Exception as e:
                st.error(f"Search failed: {e}")
            bar.empty()
    with st.expander("Learning search (optional): run batches of models, and between batches draw the variables that keep helping more often"):
        l1, l2, l3 = st.columns(3)
        batch = l1.number_input("Models per batch", 100, 5000, LEARN_BATCH, step=100)
        rounds = l2.number_input("Rounds", 1, 20, LEARN_ROUNDS)
        learn_on = l3.checkbox("Learning on", value=True, key="learn_on",
                               help="Off = the same batches with your weights held fixed, for comparison.")
        st.caption(f"After each batch, Python takes the best {LEARN_TOP:.0%} of that batch's models and checks, for each variable, how much the score changes with it in versus out. "
                   f"Variables that help get drawn more often in the next batch, and ones that hurt get drawn less (step {LEARN_ETA}, weights kept between {LEARN_CLAMP[0]} and {LEARN_CLAMP[1]}). "
                   "Your Lock, Favor and Exclude settings always win over learned weights. "
                   f"After the last batch it climbs one change at a time from the baseline and the {LEARN_CLIMBS} best models found, like the riff search. "
                   "Weights only change what gets tried; BIC still judges every model. Learning can chase early luck, so check the luck benchmark afterwards: it learns on shuffled data too.")
        if st.button("Run learning search"):
            bar = st.progress(0.0)
            try:
                finish(run_learning(train, y, pick, max_size, int(batch), int(rounds), learn_on, must, bar, steer),
                       {"kind": "learn", "batch": int(batch), "rounds": int(rounds), "learn": bool(learn_on)})
                ran = True
            except Exception as e:
                st.error(f"Search failed: {e}")
            bar.empty()
    if ran:
        st.rerun()  # redraw the checkpoint table with the new "In top models" column

    res = st.session_state.get("search")
    if res is None:
        return
    stale = (res["pool"] != pick or res["max_size"] != max_size
             or (res.get("holdout") or {}).get("entity_column") != entity_col
             or (res.get("steer") or {}).get("cfg") != steer["cfg"])
    if stale:
        st.warning("The pool, model-size limit, holdout setting, or checkpoint (locks, favors, rules) changed since the last run. Run the search again.")

    # --- Results ---
    st.divider()
    s = res["scores"]
    top1 = res["top"][0]
    how = {"all": "every possible model", "sample": "a random sample of models",
           "hybrid": "random scouting plus riffing from promising starts",
           "learning": "batches of random models, reweighted between batches"}[res["mode"]]
    ui.section("Search results", f"Ran {res['models_run']:,} models ({how}); skipped {res['skipped']:,} that could not be fitted.")
    if not res.get("baseline_testable", True):
        st.info("Some Base Data Model predictors are not in this pool (dropped in Stage 4 or removed above), so no tested model reproduces the baseline exactly. The best model can legitimately score below it.")
    if not res["check_ok"]:
        st.error("The fast scorer disagreed with statsmodels for the top model. Do not trust this search; please report it.")

    by_bic = res.get("score_name") == "bic"
    ui.metric_strip([
        ("Leaderboard #1", f"{top1['Adjusted R Square']:.4f}",
         f"Adjusted R Square of the model ranked first by {'BIC' if by_bic else 'Adjusted R Square'}. The highest Adjusted R Square among all models tested was {s.max():.4f}.",
         f"{top1['Adjusted R Square'] - base_adj:+.4f} vs baseline"),
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
    ui.section("Leaderboard", f"Top {len(res['top'])} models, ranked by {'BIC (lower is better)' if by_bic else 'Adjusted R Square'}.")
    lb = pd.DataFrame(res["top"])
    lb = lb[[c for c in lb.columns if c != "Variables"] + ["Variables"]]
    fmt = {c: "{:.4f}" for c in ["Adjusted R Square", "R Square", "Holdout R Square", "Shrinkage"] if c in lb}
    fmt.update({c: "{:.3f}" for c in ["Standard Error", "Holdout Error (RMSE)", "Error Ratio", "CV RMSE", "CV RMSE SE"] if c in lb})
    fmt.update({c: "{:.1f}" for c in ["BIC", "BIC Gap"] if c in lb})
    st.dataframe(lb.style.format(fmt), hide_index=True)
    cvi = res.get("cv_info")
    st.caption("**BIC** is a fit score that charges extra for every predictor, so it favors simpler models than Adjusted R Square. "
               "**BIC Gap** is how far a model sits above the best BIC; a gap above about 10 is conventionally strong evidence it is worse. "
               + (f"**CV RMSE** is the typical miss when the model is refit {cvi['folds']} times, each time predicting a slice of the training rows it did not see "
                  f"(slices grouped by {cvi['grouped_by']}); **CV RMSE SE** is how much that estimate wobbles between slices. "
                  "Models whose CV RMSE differ by less than the SE are effectively tied. " if cvi else ""))
    st.caption("Winner's curse: the best of thousands of models looks better than it will on new data, partly by luck. Treat the winner as a strong candidate, not a proven result. Many of the top models differ by tiny amounts, so look at the whole top group, not just rank 1.")

    with st.expander("How often does each variable appear in the top models?"):
        names = [n for r in res["top"] for n in r["Variables"].split(", ")]
        freq = pd.Series(names).value_counts()
        freq_df = pd.DataFrame({"Models containing it": freq,
                                "Share of top models": freq / len(res["top"]) * 100})
        st.dataframe(freq_df.style.format({"Models containing it": "{:,.0f}", "Share of top models": "{:.0f}%"})
                     .background_gradient(subset=["Share of top models"], cmap="Oranges"))
        st.caption(f"Counts only the {len(res['top'])} leaderboard models. A variable in nearly all of them is robustly useful. One that appears in only a few may owe its place to luck.")

    # --- Holdout check ---
    h = res.get("holdout")
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

    # --- How much to trust this ---
    st.divider()
    ui.section("How much to trust this",
               "Three honest yardsticks: error on rows the model never saw, error from guessing the average, and how much of the improvement shuffled (meaningless) data can fake.")
    err_items = []
    if "CV RMSE" in top1:
        err_items.append(("Winner CV RMSE", f"{top1['CV RMSE']:.3f}",
                          f"Typical miss in cross-validation on training rows (wobble between folds: {top1.get('CV RMSE SE', float('nan')):.3f})."))
    if "Holdout Error (RMSE)" in top1:
        err_items.append(("Winner holdout RMSE", f"{top1['Holdout Error (RMSE)']:.3f}", "Typical miss on the held-out rows."))
    if cvi:
        err_items.append(("Predict-the-mean RMSE", f"{cvi['mean_guess_rmse']:.3f}",
                          "Typical miss if you always guessed the average Y. The model should be well below this."))
    if err_items:
        ui.metric_strip(err_items)
        st.caption("If CV RMSE and holdout RMSE are close to each other and well below the predict-the-mean RMSE, the model genuinely predicts. "
                   "If they sit near the predict-the-mean value, the fit is mostly an illusion.")

    st.markdown("**Luck benchmark**")
    st.caption("Shuffles Y (breaking any real link to the predictors), then repeats this exact search and measures how much BIC improvement it still finds over the baseline. "
               "That is the improvement you can get from luck alone. Each repeat is a full search, so large pools take a while: lower the repeats to go faster.")
    repeats = st.number_input("Shuffled-Y repeats", 1, 20, LUCK_REPEATS)
    if st.button("Run luck benchmark", disabled=stale):
        bar = st.progress(0.0)
        try:
            res["luck"] = luck_benchmark(res, train, df_clean.loc[train.index], y, base_cols, int(repeats), bar)
            recipe["luck_benchmark"] = res["luck"]
        except Exception as e:
            st.error(f"Luck benchmark failed: {e}")
        bar.empty()
    lk = res.get("luck")
    if lk:
        level, text = luck_verdict(lk["real_gain"], lk["shuffled_gains"])
        getattr(st, level)(text)
        ui.metric_strip([
            ("Real BIC gain", f"{lk['real_gain']:.1f}", "BIC of a no-predictor model minus the best BIC found. Bigger means the model explains more than guessing the average."),
            ("Shuffled gain (average)", f"{np.mean(lk['shuffled_gains']):.1f}", f"Same measure, averaged over {lk['repeats']} searches on shuffled Y."),
            ("Shuffled gain (largest)", f"{max(lk['shuffled_gains']):.1f}", "The biggest fit luck alone produced."),
            ("Best Adj R Square, real", f"{lk['real_gain_adj']:.4f}", "Highest Adjusted R Square tested on the real data.",
             f"shuffled avg {np.mean(lk['shuffled_gains_adj']):.4f}"),
        ])
        if lk.get("shuffled_gains_learning_off"):
            st.caption(f"Learning search: on shuffled data, learning on found an average gain of {np.mean(lk['shuffled_gains']):.1f}; "
                       f"the same batches with learning off found {np.mean(lk['shuffled_gains_learning_off']):.1f}. "
                       "If learning on is clearly higher on shuffled data, learning is partly chasing luck.")
        st.caption(f"Versus your own baseline, the best model gains {lk['real_gain_vs_baseline']:.1f} BIC and "
                   f"{s.max() - base_adj:+.4f} Adjusted R Square. A small number here means the search mostly confirmed your Base Data Model "
                   "(BIC rewards dropping predictors that add nothing), not that it found something new.")
    ln = res.get("learning")
    if ln:
        st.divider()
        ui.section("What the learning search learned",
                   f"{ln['rounds']} rounds of up to {ln['batch']:,} models; learning {'on' if ln['learn'] else 'off (weights held fixed)'}.")
        bests = ln["best_by_round"]
        if bests:
            unit = "BIC" if res.get("score_name") == "bic" else "Adj R Square"
            ui.metric_strip([
                ("Best in round 1", f"{bests[0]:,.1f}" if unit == "BIC" else f"{bests[0]:.4f}", f"Best {unit} in the first batch."),
                ("Best in last round", f"{bests[-1]:,.1f}" if unit == "BIC" else f"{bests[-1]:.4f}", f"Best {unit} in the final batch. Lower BIC is better."),
                ("Models tried", f"{res['models_run']:,}", "Distinct models across all batches."),
                ("Weights kept by you", f"{len(ln['fixed'])}", "Favor variables keep your multiplier; locked ones are always in."),
            ])
        if ln["learn"] and len(ln["history"]) > 1:
            fig = learning_fig(ln["history"], ln["fixed"])
            st.pyplot(fig)
            plt.close(fig)
        with st.expander("Weight history, every variable and round"):
            hist = pd.DataFrame(ln["history"]).set_index("Round").T
            hist = hist.sort_values(hist.columns[-1], ascending=False)
            st.dataframe(hist.style.format("{:.2f}").background_gradient(cmap="Oranges", axis=None))
            st.caption("1.00 = neutral. Above 1 = drawn more often in the next batch, below 1 = less often. "
                       "A variable that rises every round keeps appearing in the best models; one that sinks rarely helps.")

    # --- Stability check ---
    st.divider()
    ui.section("Stability check (bootstrap)",
               "Re-runs this exact search on resampled copies of the training rows" + (f" (whole {entity_col} groups resampled)" if entity_col else "")
               + " and counts how often each variable is in the winner. A variable that wins in most runs is robust; one that wins rarely may owe its place to luck. "
               "Each run is a full search, so this takes a while.")
    runs = st.number_input("Resampled runs", 5, 50, BOOT_RUNS)
    if st.button("Run stability check", disabled=stale):
        bar = st.progress(0.0)
        done = False
        try:
            res["bootstrap"] = bootstrap_stability(res, train, y, raw, entity_col, base_cols, int(runs), bar)
            bs_ = res["bootstrap"]
            recipe["bootstrap_stability"] = {"runs": bs_["runs"], "grouped_by": bs_["grouped_by"], "distinct_winners": bs_["distinct_winners"],
                                             "winner_share_pct": {c: round(v["wins"]) for c, v in sorted(bs_["freq"].items(), key=lambda kv: -kv[1]["wins"]) if v["wins"] > 0}}
            st.session_state.search_runs = st.session_state.get("search_runs", 0) + 1
            done = True
        except Exception as e:
            st.error(f"Stability check failed: {e}")
        bar.empty()
        if done:
            st.rerun()  # show the "Resample wins" column in the checkpoint table
    bs = res.get("bootstrap")
    if bs:
        fr = pd.DataFrame(bs["freq"]).T.rename(columns={"wins": "In the winner", "top5": "In a top-5 model"})
        fr = fr.sort_values(["In the winner", "In a top-5 model"], ascending=False)
        ui.metric_strip([
            ("Runs completed", f"{bs['runs']}", f"Resampled by {bs['grouped_by']}."),
            ("Robust (80%+ of winners)", f"{int((fr['In the winner'] >= 80).sum())}", "Variables in the winner of at least 80% of runs."),
            ("Fragile (under 20%)", f"{int(((fr['In the winner'] > 0) & (fr['In the winner'] < 20)).sum())}", "Variables that won sometimes, but rarely."),
            ("Distinct winning models", f"{bs['distinct_winners']}", "How many different models came out on top. Fewer means a more stable answer."),
        ])
        with st.expander("Selection frequency for every variable"):
            st.dataframe(fr.style.format("{:.0f}%").background_gradient(cmap="Oranges", vmin=0, vmax=100))
            st.caption("Bootstrap runs see slightly different data each time, the way a new sample would. These shares are also in the checkpoint table, "
                       "so you can lock robust variables or exclude fragile ones and run again.")

    if res.get("starts"):
        st.divider()
        ui.section("Where each starting point ended up")
        st.dataframe(pd.DataFrame(res["starts"]).style.format(
            {"Start Adj R Square": "{:.4f}", "End Adj R Square": "{:.4f}"}), hide_index=True)
        st.caption(f"{len(res['starts'])} starts reached {res['distinct_ends']} distinct final model(s). "
                   "Each climb is guided by BIC; the Adjusted R Square columns are shown for reference, so a climb can end at a lower Adjusted R Square than it started. "
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
    for rec in d.itertuples(index=False, name=None):
        ws.append([xl_val(v) for v in rec])  # appends below the header just written (about twice as fast as cell by cell)
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
    r = put_title(ws, r, "Data audit")
    r = put_lines(ws, r, audit_lines(recipe.get("audit")), wrap=True)
    r = put_title(ws, r, "Changes from the base model")
    ch = pd.DataFrame(fm["changes_from_base"])
    if ch.empty:
        r = put_lines(ws, r, ["The selected model uses exactly the base model's variables."])
    else:
        r = write_df(ws, ch, r, index=False)
    r = put_title(ws, r, "Limitations and next steps (flagged by the automatic checks)")
    gaps = known_gaps(recipe)
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
    cl = fm.get("cluster_robust")
    if cl:
        r = put_title(ws, r, f"Selected model: clustered standard errors (by {cl['cluster_column']}, {cl['clusters']} groups)")
        r = put_lines(ws, r, ["Same coefficients; standard errors and p-values allow for repeated rows from the same entity. Excel's ToolPak cannot produce these."])
        r = write_df(ws, pd.DataFrame(cl["table"]).T, r)

    # ---- 3. Leaderboard ----
    ws = wb.create_sheet("Leaderboard")
    by = recipe.get("search_score")
    notes_lb = [
        f"Top {len(recipe['leaderboard'])} models, ranked by {'BIC (lower is better)' if by == 'bic' else 'Adjusted R Square'} (fit on training rows).",
        "BIC Gap = distance from the best BIC. CV RMSE and CV RMSE SE come from cross-validation on the training rows only.",
        "Holdout columns report out-of-sample performance. Do not re-pick the winner by holdout score.",
    ]
    lk = recipe.get("luck_benchmark")
    if lk:
        notes_lb.append(f"Luck benchmark ({lk['repeats']} shuffled-Y repeats): real BIC gain {lk['real_gain']:.1f}; "
                        "shuffled BIC gains " + ", ".join(f"{g:.1f}" for g in lk["shuffled_gains"]) + ".")
    ose = fm.get("one_se_rule")
    if ose:
        notes_lb.append(f"One-standard-error rule: rank {ose['rank']} ({ose['predictors']} predictors) is the simplest model whose CV RMSE "
                        f"({ose['cv_rmse']:.3f}) is within one SE of the best ({ose['best_cv_rmse']:.3f} + {ose['best_cv_se']:.3f}).")
    bl = recipe.get("batch_learning")
    if bl:
        notes_lb.append(f"Learning search: {bl['rounds']} rounds of {bl['batch']} models, learning {'on' if bl['learning_on'] else 'off'}. "
                        "Raised most: " + (", ".join(f"{k} {v}" for k, v in bl["raised_most"].items()) or "none") + ".")
    bs = recipe.get("bootstrap_stability")
    if bs:
        notes_lb.append(f"Stability check: {bs['runs']} resampled runs (by {bs['grouped_by']}), {bs['distinct_winners']} distinct winners. Share of winners containing: "
                        + ", ".join(f"{k} {v}%" for k, v in list(bs["winner_share_pct"].items())[:12]) + ".")
    r = put_lines(ws, 1, notes_lb)
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


def cluster_table(m, index, raw, cluster_col):
    """Refit the same OLS with cluster-robust standard errors (groups = cluster_col), t-based with G - 1 df.
    Coefficients are identical; only the standard errors, p-values and confidence intervals change."""
    groups = pd.factorize(raw.loc[index, cluster_col].astype(str))[0]
    mc = m.model.fit(cov_type="cluster", cov_kwds={"groups": groups}, use_t=True)
    ci = mc.conf_int(alpha=ALPHA)
    t = pd.DataFrame({"Coefficient": m.params, "Plain SE": m.bse, "Plain p": m.pvalues,
                      "Clustered SE": mc.bse, "Clustered p": mc.pvalues,
                      f"Clustered lower {CONF_LEVEL}%": ci[0], f"Clustered upper {CONF_LEVEL}%": ci[1]})
    t["SE ratio"] = t["Clustered SE"] / t["Plain SE"]
    t.index = ["Intercept" if i == "const" else str(i) for i in t.index]
    return t, int(groups.max() + 1)


def cluster_summary(t, n_groups, col):
    """Compact numbers for the recipe, the report and the Excel sheet."""
    v = t.drop(index="Intercept", errors="ignore")
    return {"cluster_column": col, "clusters": n_groups,
            "median_se_ratio": float(v["SE ratio"].median()) if len(v) else None,
            "significant_plain": int((v["Plain p"] < ALPHA).sum()), "significant_clustered": int((v["Clustered p"] < ALPHA).sum()),
            "lost_significance": [str(i) for i in v.index[(v["Plain p"] < ALPHA) & (v["Clustered p"] >= ALPHA)]],
            "table": {str(i): {k: float(r[k]) for k in ("Coefficient", "Plain SE", "Plain p", "Clustered SE", "Clustered p")}
                      for i, r in t.iterrows()}}


def one_se_pick(top):
    """One-standard-error rule: among leaderboard models with CV, the simplest whose CV RMSE is within one SE of the
    lowest CV RMSE (ties broken by lower CV RMSE). None when CV is missing."""
    cv = [r for r in top if r.get("CV RMSE") is not None and r.get("CV RMSE SE") is not None]
    if not cv:
        return None
    best = min(cv, key=lambda r: r["CV RMSE"])
    limit = best["CV RMSE"] + best["CV RMSE SE"]
    pick = min((r for r in cv if r["CV RMSE"] <= limit), key=lambda r: (r["Predictors"], r["CV RMSE"]))
    return {"rank": int(pick["Rank"]), "predictors": int(pick["Predictors"]), "cv_rmse": float(pick["CV RMSE"]),
            "best_rank": int(best["Rank"]), "best_cv_rmse": float(best["CV RMSE"]), "best_cv_se": float(best["CV RMSE SE"]),
            "limit": float(limit), "within": [int(r["Rank"]) for r in cv if r["CV RMSE"] <= limit]}


def known_gaps(recipe):
    """KNOWN_GAPS adjusted to what this run actually did."""
    cl = (recipe.get("final_model") or {}).get("cluster_robust")
    out = []
    for g in KNOWN_GAPS:
        if g.startswith("Panel data"):
            if cl:
                g = (f"Clustered standard errors (by {cl['cluster_column']}, {cl['clusters']} groups) are shown for the selected model only; "
                     "the search, BIC, cross-validation and the ToolPak tables still use ordinary standard errors.")
            elif not recipe.get("dropped_high_cardinality"):
                continue
        out.append(g)
    return out


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
    "Panel data is flagged but not corrected (no clustered standard errors were computed for this run).",
    "No robust standard errors for unequal spread.",
    "Linear OLS only; curvature is handled only through log and square terms.",
    "The holdout is one split, so its score is itself noisy.",
    "Median imputation was used for missing predictor values. The medians use all rows, including the holdout rows, which is a small leak.",
    "Post-selection inference: the selected model was chosen from many tested models, so its p-values and confidence intervals are optimistic; the holdout results are the more honest check."
]

REPORT_SYSTEM = (
    "You are a statistics tutor writing the final report for an introductory college statistics student and professor. "
    "You receive a compact JSON summary of a regression analysis: a base model, a search over many variable subsets, "
    "a selected model, and a holdout check. Write a plain-English report with these short sections: "
    "1) Summary: what was predicted, the base model vs the selected model (Adjusted R Square, standard error, holdout R Square, error ratio). "
    "2) Exact changes from the base model: list every added and removed variable with its formula and reason, so anyone can reproduce it. "
    "3) How much to trust it: how many models were searched (state the number); that models were ranked by BIC "
    "(a fit score that penalizes extra predictors; a BIC gap above about 10 is strong evidence a model is worse); "
    "the selected model's cross-validation RMSE and its standard error compared with the holdout RMSE and the predict-the-mean RMSE; "
    "the luck benchmark (the improvement found when Y is shuffled, versus the real improvement; if luck_benchmark is null, say it was not run); "
    "winner's curse; shrinkage on the holdout; and whether near-tied runners-up are simpler. "
    "If one_standard_error_rule is present, say which leaderboard model it suggests (the simplest whose CV RMSE is within one standard error of the best) "
    "and whether the selected model is that one. If selected_model_cluster_robust is present, explain that clustered standard errors allow for "
    "repeated rows from the same entity, give the median SE ratio, and name any variables that lose significance with them. "
    "Also state any steering in search.steering: locked variables were forced into every model, excluded ones were never tried, "
    "favored ones were only drawn more often (they still had to earn their place on BIC). "
    "If batch_learning is present, explain that the search reweighted variables between batches, that learning can chase luck, "
    "and compare with the luck benchmark (its shuffled runs learned too; shuffled_gains_learning_off shows the same without learning). "
    "If bootstrap_stability is present, call variables in most resampled winners robust and those in few fragile. "
    "4) What the analyst noticed: weave in the analyst notes, the segments explored (state how many) and any segment findings, "
    "and say group differences found by trying many splits are hypotheses. "
    "5) Limitations and next steps: for each item in health_notes, explain in a sentence why it matters, then give 2 to 3 starter steps "
    "to fix or check it OUTSIDE this app (for example clustered standard errors for panel data, robust standard errors for unequal spread, "
    "inspecting high Cook's distance rows, log or squared terms or splines for curvature, dropping or combining collinear variables). "
    "Also cover known_app_gaps, and any data_audit finding whose decision was 'left as is' or 'flag only'. "
    "6) Pro takeaways: 3 bullets. "
    "Define any statistical term in a few words the first time you use it. "
    "Use only the numbers provided, do not invent facts, and do not choose for the analyst."
)


def build_report_profile(recipe):
    """Compact JSON for the final report. No raw rows."""
    fm, h = recipe["final_model"], recipe.get("holdout") or {}
    row = next((r for r in recipe["leaderboard"] if r["Rank"] == fm["rank"]), {})
    prof = {
        "target": recipe["target"],
        "base_model_columns": recipe["base_model_columns"],
        "baseline_from_stage3": recipe.get("baseline"),
        "selected_model": {k: fm[k] for k in (
            "rank", "variables", "metrics_base", "metrics_best", "changes_from_base",
            "engineered_impact", "runners_up", "line_diagnostics", "health_notes")},
        "selected_model_cluster_robust": {k: v for k, v in (fm.get("cluster_robust") or {}).items() if k != "table"} or None,
        "one_standard_error_rule": fm.get("one_se_rule"),
        "selected_model_search_scores": {k: row.get(k) for k in ("BIC", "BIC Gap", "CV RMSE", "CV RMSE SE")},
        "search": {
            "models_searched": recipe.get("models_searched"), "mode": recipe.get("search_mode"),
            "ranked_by": recipe.get("search_score"), "cross_validation": recipe.get("cv"),
            "pool": recipe.get("search_pool"), "max_size": recipe.get("search_max_size"),
            "skipped": recipe.get("search_skipped"), "benchmarks": recipe.get("search_benchmarks"),
            "steering": {"locked": recipe.get("search_locked"), "excluded": recipe.get("search_excluded"),
                         "draw_weights": recipe.get("search_weights"), "rules": recipe.get("search_rules")},
        },
        "luck_benchmark": recipe.get("luck_benchmark"),
        "batch_learning": recipe.get("batch_learning"),
        "bootstrap_stability": recipe.get("bootstrap_stability"),
        "holdout": {k: h.get(k) for k in ("mode", "entity_column", "fraction", "n_train", "n_holdout", "baseline")},
        "splits_explored": recipe.get("splits_explored", []),
        "segment_findings": recipe.get("segment_findings", {}),
        "analyst_notes": {k: v for k, v in recipe.get("analyst_notes", {}).items() if v.strip()},
        "baseline_claude_read": recipe.get("baseline_claude_read", ""),
        "stage3_health_notes": recipe.get("health_notes", []),
        "data_audit": audit_lines(recipe.get("audit")),
        "data_picture_findings": [f"{f['title']} (severity {f['severity']})" for f in recipe.get("data_picture", [])],
        "candidate_screening": recipe.get("screening_summary"),
        "known_app_gaps": known_gaps(recipe),
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
        st.dataframe(changes, hide_index=True)
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
        st.dataframe(pd.DataFrame(fmt_rows, index=["Base", "Best"]).T)
        st.dataframe(comp.style.format({"Base coef": "{:.4f}", "Base p": "{:.4f}", "Best coef": "{:.4f}", "Best p": "{:.4f}"}, na_rep="-"), hide_index=True)
        st.caption("Kept = in both models. Added = only in the best model (often engineered). Removed = in the base model but not the best. "
                   "Coefficients for engineered columns are in the engineered column's own units.")

    # ---- Clustered standard errors (U6) ----
    ents = entity_candidates(raw)
    cl_info, cl_tab = None, None
    if ents:
        st.divider()
        ui.section("Standard errors that allow for repeated entities",
                   "When the same entity (for example a country) appears in many rows, those rows are not independent, so ordinary standard errors are too small. "
                   "Clustered standard errors fix that. The coefficients do not change; only how sure we can be about them.")
        default = h["entity_column"] if h["entity_column"] in ents else ents[0]
        opts = ents + ["Do not cluster"]
        cl_col = st.selectbox("Cluster by", opts, index=opts.index(default), key=f"cluster_col_{st.session_state.get('pool_version', 0)}")
        if cl_col != "Do not cluster":
            cl_tab, n_g = cluster_table(win_m, train.index, raw, cl_col)
            cl_info = cluster_summary(cl_tab, n_g, cl_col)
            lost = cl_info["lost_significance"]
            ui.metric_strip([
                ("Clusters", f"{n_g:,}", f"Distinct {cl_col} values in the training rows."),
                ("Median SE ratio", f"{cl_info['median_se_ratio']:.2f}×", "Clustered SE divided by plain SE, across predictors. Above 1 means the plain numbers were overconfident."),
                ("Significant, plain", f"{cl_info['significant_plain']}", f"Predictors with p below {ALPHA} using ordinary standard errors."),
                ("Significant, clustered", f"{cl_info['significant_clustered']}", f"Predictors with p below {ALPHA} once repeated {cl_col} rows are allowed for."),
            ])
            if lost:
                st.warning(f"**{len(lost)} predictor(s) lose significance once repeated {cl_col} rows are allowed for:** " + ", ".join(lost)
                           + ". Their effect may be real, but this data cannot pin it down as firmly as the plain p-values suggest.")
            elif n_g < 30:
                st.info(f"Only {n_g} clusters: clustered standard errors are themselves rough with fewer than about 30 groups.")
            else:
                st.success("Every predictor that was significant stays significant with clustered standard errors.")
            with st.expander("Plain vs clustered, every coefficient"):
                st.dataframe(cl_tab.style.format({c: "{:.4g}" for c in cl_tab.columns})
                             .background_gradient(subset=["SE ratio"], cmap="Oranges"))
                st.caption(f"Clustered intervals use a t distribution with {n_g - 1} degrees of freedom (clusters minus 1). "
                           "The Excel ToolPak tables and the search keep using plain standard errors, so they still match Excel.")

    imp = engineered_impact(train, test, y, win_cols, recipe.get("engineered_columns", []),
                            wm["Adjusted R Square"], wm["Holdout R Square"])
    with st.expander("Impact of each engineered variable"):
        if imp.empty:
            st.info("This model uses no engineered variables.")
        else:
            st.dataframe(imp.style.format({c: "{:.4f}" for c in imp.columns if c != "Engineered variable"}), hide_index=True)
            st.caption("Each row refits the model without that one variable. A bigger drop means the variable matters more. A drop near 0 means it adds little, so a simpler model may do just as well.")

    ose = one_se_pick(top)
    wset, rows = set(win_cols), []
    others = [r for r in top if r["Rank"] != rank][:3]
    if ose and ose["rank"] != rank and all(r["Rank"] != ose["rank"] for r in others):
        others.append(next(r for r in top if r["Rank"] == ose["rank"]))
    for r in others:
        s = set(r["Variables"].split(", "))
        gap = r["Adjusted R Square"] - sel["Adjusted R Square"]
        near = abs(gap) <= NEAR_TIE
        if ose:
            if r["Rank"] == ose["rank"]:
                verdict = "1-SE rule pick: simplest model that predicts as well as the best"
            elif r["Rank"] in ose["within"]:
                verdict = "Tied with the best on CV" + (", and simpler" if r["Predictors"] < sel["Predictors"] else "")
            else:
                verdict = "Worse on CV (more than one SE behind the best)"
        elif gap > 0:
            verdict = "Scores higher than the selected model"
        elif gap >= -NEAR_TIE:
            verdict = "Near-tie and simpler: consider it" if r["Predictors"] < sel["Predictors"] else "Near-tie"
        else:
            verdict = "Clearly lower"
        rows.append({"Rank": r["Rank"], "Predictors": r["Predictors"], "CV RMSE": r.get("CV RMSE"), "BIC Gap": r.get("BIC Gap"),
                     "Adj R Square vs selected": gap, "Adj R Square near-tie": "yes" if near else "no",
                     "Holdout R Square": r.get("Holdout R Square"), "Error Ratio": r.get("Error Ratio"),
                     "Only in this model": ", ".join(sorted(s - wset)) or "-",
                     "Only in selected": ", ".join(sorted(wset - s)) or "-", "Verdict": verdict})
    runners = pd.DataFrame(rows)
    if ose:
        why = (f"rank {ose['best_rank']} has the lowest CV RMSE ({ose['best_cv_rmse']:.3f}, give or take {ose['best_cv_se']:.3f}), "
               f"so any model with CV RMSE up to {ose['limit']:.3f} predicts about as well.")
        if ose["rank"] == rank:
            st.success(f"**The selected model is the one-standard-error pick:** the simplest model that predicts as well as the best. In detail, {why}")
        else:
            st.info(f"**The one-standard-error rule suggests rank {ose['rank']}** ({ose['predictors']} predictors; the selected model has {sel['Predictors']}). "
                    f"In detail, {why} Simpler models that predict as well usually hold up better on new data. You can pick it in the box at the top.")
    with st.expander("Why not the other top models?"):
        if not runners.empty:
            st.dataframe(runners.style.format({"Adj R Square vs selected": "{:+.4f}", "Holdout R Square": "{:.4f}", "Error Ratio": "{:.3f}",
                                               "CV RMSE": "{:.3f}", "BIC Gap": "{:.1f}"}, na_rep="-"), hide_index=True)
        st.caption("The main verdict uses the one-standard-error rule on cross-validation: models whose CV RMSE is within one standard error of the best are treated as tied, "
                   f"and the simplest tied model is suggested. The near-tie column is the older check (within {NEAR_TIE} Adjusted R Square). "
                   "Holdout numbers are for reporting; do not re-pick by them.")

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
        st.dataframe(reg.style.format("{:,.6f}"))
        st.markdown("**ANOVA**")
        st.dataframe(anova.style.format("{:,.6g}", na_rep=""))
        st.markdown("**Coefficients**")
        st.dataframe(coef.style.format("{:,.6g}"))

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
        "health_notes": notes, "one_se_rule": ose, "cluster_robust": cl_info,
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
