import matplotlib.pyplot as plt
from cycler import cycler
import streamlit as st

# ---------- Palette (matches .streamlit/config.toml) ----------
CARD, LINE = "#16213A", "#2A3C61"
TEXT, SOFT = "#E8EEF9", "#B4C2DC"
ACCENT, HIGHLIGHT = "#4C8DFF", "#FF9F43"  # accent = "do this next"; highlight = "look here"

# ---------- Styles ----------
# Small touches the theme file can't do. Everything here is cosmetic.
CSS = """
<style>
/* Page width and title */
.block-container { max-width: 1280px; padding-top: 2.5rem; }
.stApp h1, [data-testid="stHeading"] h1 {
    font-family: Georgia, 'Times New Roman', serif !important;
    font-weight: 600 !important; font-size: 2.2rem !important;
}
h2, h3 { margin-top: 0.4rem; }

/* Number cards: every st.metric in every stage */
[data-testid="stMetric"] {
    background: #16213A; border: 1px solid #2A3C61; border-radius: 12px; padding: 14px 18px;
}
[data-testid="stMetricLabel"] { color: #B4C2DC; }
[data-testid="stMetricValue"] { font-size: 1.9rem; font-weight: 600; }

/* Expanders, alerts, bordered boxes */
[data-testid="stExpander"] { border: 1px solid #2A3C61; border-radius: 12px; background: #131D33; }
[data-testid="stAlert"] { border-radius: 12px; }
[data-testid="stVerticalBlockBorderWrapper"] { border-radius: 12px; }

/* Readable captions (no dim gray) and quiet dividers */
[data-testid="stCaptionContainer"] { color: #B4C2DC; }
hr { border-color: #2A3C61; }

/* Sidebar step buttons: left aligned */
[data-testid="stSidebar"] .stButton > button { justify-content: flex-start; text-align: left; }
</style>
"""


def apply_styles():
    """Inject the CSS above into the page (call once per run)."""
    st.markdown(CSS, unsafe_allow_html=True)


def apply_chart_theme():
    """Dark matplotlib defaults so every chart matches the app. Excel images reset to light (see chart_items)."""
    plt.rcParams.update({
        "figure.facecolor": CARD, "axes.facecolor": CARD, "savefig.facecolor": CARD,
        "axes.edgecolor": LINE, "axes.labelcolor": TEXT, "text.color": TEXT,
        "xtick.color": SOFT, "ytick.color": SOFT, "grid.color": LINE,
        "axes.spines.top": False, "axes.spines.right": False,
        "legend.facecolor": CARD, "legend.edgecolor": LINE,
        "axes.prop_cycle": cycler(color=[ACCENT, HIGHLIGHT, "#54A24B", "#E45756", "#B279A2", "#9D755D"]),
        # Boxplots default to black lines, which vanish on navy
        "boxplot.boxprops.color": TEXT, "boxplot.whiskerprops.color": TEXT,
        "boxplot.capprops.color": TEXT, "boxplot.medianprops.color": HIGHLIGHT,
        "boxplot.flierprops.markeredgecolor": SOFT,
    })


# ---------- Reusable pieces for every stage ----------
def metric_strip(items):
    """Headline strip: a row of number cards.
    items = [(label, value), (label, value, hover_help), or (label, value, hover_help, delta)]."""
    cols = st.columns(len(items))
    for col, item in zip(cols, items):
        col.metric(item[0], item[1],
                   delta=item[3] if len(item) > 3 else None,
                   help=item[2] if len(item) > 2 else None)

def section(title, caption=None):
    """Section heading with an optional one-line plain-English caption."""
    st.subheader(title)
    if caption:
        st.caption(caption)


def first_look_hint():
    """Shown before a file is uploaded: three short boxes saying what happens."""
    steps = [
        ("1. Upload", "A .csv or .xlsx table with one row per observation."),
        ("2. Choose Y", "The numeric column you want to predict."),
        ("3. Check", "Fills gaps and encodes text so every column is numeric."),
    ]
    for col, (title, text) in zip(st.columns(3), steps):
        with col.container(border=True):
            st.markdown(f"**{title}**")
            st.caption(text)


def health_list(notes):
    """Data health notes as calm cards: headline on top, 'why it matters' underneath."""
    if not notes:
        st.success("No major issues flagged by the automatic checks.")
        return
    for t in notes:
        head, sep, rest = t.partition(":**")  # notes look like "**Title:** explanation"
        with st.container(border=True):
            if sep:
                st.markdown(f":orange[●] **{head.lstrip('*')}**")
                st.caption(rest.strip())
            else:
                st.markdown(":orange[●] " + t.replace("**", ""))


# ---------- Sidebar ----------
def go_to(stage):
    """Button callback: jump to a stage the user has already unlocked."""
    st.session_state.stage_nav = stage


def render_sidebar(stages, has_key):
    """Progress bar, one button per stage (locked ones dimmed), then the API key box."""
    unlocked = st.session_state.unlocked
    current = stages.index(st.session_state.stage_nav)
    with st.sidebar:
        st.markdown("### Your progress")
        st.progress((current + 1) / len(stages))  # fills with the step you are on
        st.caption(f"Step {current + 1} of {len(stages)}")
        for i, stage in enumerate(stages):
            name = stage.split(". ", 1)[1]  # "1. Upload & Complete" -> "Upload & Complete"
            mark = "✓" if i < unlocked else str(i + 1)
            st.button(
                f"{mark}   {name}",
                key=f"nav_{i}",
                on_click=go_to,
                args=(stage,),
                disabled=i > unlocked,
                type="primary" if i == current else "secondary",
                use_container_width=True,
            )
        st.divider()
        st.text_input("Anthropic API key", type="password", key="api_key_box",
                      help="Only needed for Claude's written explanations. Everything else works without it.")
        st.caption("✓ Key found" if has_key else "No key set: Claude features are off")


# ---------- Page header ----------
def render_header():
    st.title("AI-Guided Multiple Regression Tool")
    st.caption("Upload a table, choose what to predict, and explore which variables relate to it.")
