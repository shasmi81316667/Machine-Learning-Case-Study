"""ML Studio: steel-plate fault classification + customer segmentation, in a dark neon Streamlit UI."""
import math
import os
import numpy as np, pandas as pd
import plotly.express as px, plotly.graph_objects as go
import streamlit as st
import ml

st.set_page_config(page_title="ML Studio", page_icon="🧠", layout="wide", initial_sidebar_state="expanded")

PALETTE = ["#8b5cf6", "#22d3ee", "#f472b6", "#fbbf24", "#34d399", "#fb7185", "#60a5fa", "#a3e635", "#f97316", "#c084fc"]
INK, MUTED, GRID = "#e6e9f5", "#8b93b8", "rgba(139,147,184,.15)"

# ───────────────────────────── Look & feel ─────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
html, body, [class*="css"], .stApp { font-family: 'Inter', sans-serif; }
.stApp { background: radial-gradient(1200px 600px at 10% -10%, rgba(139,92,246,.22), transparent 60%),
                     radial-gradient(900px 500px at 100% 0%, rgba(34,211,238,.14), transparent 55%), #0b1020; }
header[data-testid="stHeader"] { background: transparent; }
.block-container { padding-top: 1.6rem; max-width: 1400px; }
[data-testid="stSidebar"] { background: linear-gradient(180deg,#121936 0%,#0d1230 100%); border-right: 1px solid rgba(139,92,246,.25); }
h1,h2,h3 { letter-spacing: -.02em; }

.hero { position: relative; overflow: hidden; padding: 30px 34px; border-radius: 22px; margin-bottom: 18px;
        background: linear-gradient(120deg, rgba(139,92,246,.35), rgba(34,211,238,.18) 60%, rgba(244,114,182,.2));
        border: 1px solid rgba(255,255,255,.12); box-shadow: 0 20px 60px rgba(139,92,246,.18); }
.hero:after { content:""; position:absolute; right:-60px; top:-60px; width:260px; height:260px; border-radius:50%;
              background: radial-gradient(circle, rgba(255,255,255,.18), transparent 65%); }
.hero h1 { margin:0; font-size: 2.3rem; font-weight: 800; background: linear-gradient(90deg,#fff,#c4b5fd 60%,#67e8f9);
           -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
.hero p { margin: 8px 0 14px; color: #c9cff0; font-size: 1.02rem; max-width: 820px; }
.chip { display:inline-block; padding: 4px 12px; margin: 0 6px 6px 0; border-radius: 999px; font-size:.78rem; font-weight:600;
        background: rgba(255,255,255,.09); border: 1px solid rgba(255,255,255,.16); color:#e9ecff; }

.kpi { padding: 16px 18px; border-radius: 16px; background: rgba(255,255,255,.04); border: 1px solid rgba(255,255,255,.09);
       backdrop-filter: blur(6px); height: 100%; }
.kpi .l { color: #8b93b8; font-size:.74rem; text-transform: uppercase; letter-spacing:.09em; font-weight:600; }
.kpi .v { font-size: 1.35rem; font-weight: 800; margin-top: 2px; color:#fff; }
.kpi .s { color:#a5adcf; font-size:.8rem; margin-top: 2px; }
.kpi.a { border-top: 3px solid #8b5cf6; } .kpi.b { border-top: 3px solid #22d3ee; }
.kpi.c { border-top: 3px solid #f472b6; } .kpi.d { border-top: 3px solid #fbbf24; }

.step { display:flex; align-items:center; gap:12px; margin: 10px 0 6px; }
.step .n { width:30px; height:30px; border-radius:50%; display:flex; align-items:center; justify-content:center; font-weight:800;
           background: linear-gradient(135deg,#8b5cf6,#22d3ee); color:#fff; flex: none; }
.step .t { font-weight: 700; font-size: 1.12rem; } .step .d { color:#8b93b8; font-size:.86rem; }

.result { padding: 22px; border-radius: 20px; text-align:center; border: 1px solid rgba(255,255,255,.14);
          background: linear-gradient(160deg, rgba(139,92,246,.30), rgba(34,211,238,.12)); box-shadow: 0 16px 50px rgba(139,92,246,.22); }
.result .e { font-size: 3rem; line-height:1; } .result .lab { font-size: 1.75rem; font-weight: 800; margin-top: 6px; }
.result .sub { color:#b9c0e6; font-size:.88rem; margin-top: 4px; }
.vote { display:inline-block; margin: 3px; padding: 3px 10px; border-radius: 10px; font-size:.74rem; background: rgba(255,255,255,.07);
        border: 1px solid rgba(255,255,255,.1); }
.vote.ok { background: rgba(52,211,153,.14); border-color: rgba(52,211,153,.45); }
.note { padding: 12px 16px; border-radius: 12px; background: rgba(34,211,238,.08); border: 1px solid rgba(34,211,238,.3);
        color:#bdf3fb; font-size:.9rem; margin: 6px 0 12px; }
.brand { font-size: 1.35rem; font-weight: 800; background: linear-gradient(90deg,#c4b5fd,#67e8f9);
         -webkit-background-clip:text; -webkit-text-fill-color:transparent; }

.stTabs [data-baseweb="tab-list"] { gap: 6px; background: rgba(255,255,255,.04); padding: 6px; border-radius: 14px; }
.stTabs [data-baseweb="tab"] { height: 42px; border-radius: 10px; padding: 0 18px; font-weight: 600; }
.stTabs [aria-selected="true"] { background: linear-gradient(135deg, rgba(139,92,246,.55), rgba(34,211,238,.3)); color:#fff; }
.stTabs [data-baseweb="tab-highlight"], .stTabs [data-baseweb="tab-border"] { display:none; }
.stButton > button { border-radius: 12px; font-weight: 600; border: 1px solid rgba(255,255,255,.16);
        background: rgba(255,255,255,.06); transition: all .15s; }
.stButton > button:hover { border-color:#8b5cf6; box-shadow: 0 0 0 3px rgba(139,92,246,.25); transform: translateY(-1px); }
[data-testid="stExpander"] { border-radius: 14px; border: 1px solid rgba(255,255,255,.1); background: rgba(255,255,255,.03); }
[data-testid="stDataFrame"] { border-radius: 12px; overflow: hidden; }
div[data-testid="stColumn"]:has(.sticky) { position: sticky; top: 3.6rem; align-self: flex-start; }
</style>
""", unsafe_allow_html=True)


# ───────────────────────────── helpers ─────────────────────────────
def hero(title, sub, chips):
    st.markdown(f'<div class="hero"><h1>{title}</h1><p>{sub}</p>' + "".join(f'<span class="chip">{c}</span>' for c in chips) + "</div>",
                unsafe_allow_html=True)


def kpi(col, label, value, sub="", tone="a"):
    col.markdown(f'<div class="kpi {tone}"><div class="l">{label}</div><div class="v">{value}</div><div class="s">{sub}</div></div>',
                 unsafe_allow_html=True)


def step(n, title, desc=""):
    st.markdown(f'<div class="step"><div class="n">{n}</div><div><div class="t">{title}</div><div class="d">{desc}</div></div></div>',
                unsafe_allow_html=True)


def style(fig, h=380, legend=True):
    fig.update_layout(height=h, template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      colorway=PALETTE, font=dict(family="Inter", color=INK), margin=dict(l=10, r=10, t=48, b=10),
                      legend_title_text="", showlegend=legend, title_font=dict(size=15))
    fig.update_xaxes(gridcolor=GRID, zerolinecolor=GRID)
    fig.update_yaxes(gridcolor=GRID, zerolinecolor=GRID)
    return fig


def show(fig, h=380, legend=True, key=None):
    st.plotly_chart(style(fig, h, legend), width="stretch", key=key, config={"displaylogo": False})


@st.cache_resource(show_spinner="🧠 Training the 10 models (one time only, under a minute)...")
def get_cls():
    return ml.load_classification()


@st.cache_resource(show_spinner="⚙️ Preparing the clustering models...")
def get_clu(raw):
    return ml.build_clustering(raw)


@st.cache_data(show_spinner=False)
def macro_rocs(_art, names):
    return {n: ml.roc_curves(_art, n)[1] for n in names}


# ═════════════════════════ Classification ═════════════════════════
def page_classification():
    art = get_cls()
    feats, table, classes, info = art["feats"], art["table"], art["classes"], art["info"]
    fmap = {f["name"]: f for f in feats}
    is_int = lambda f: float(f["min"]).is_integer() and float(f["max"]).is_integer() and f["max"] - f["min"] > 5
    best = table.index[0]
    hero("🔩 Steel Plate Fault Classifier",
         "Tell the app what a defect on a steel plate looks like, pick one of 10 machine-learning models, and it names the fault. "
         "Then compare all 10 models side by side.",
         [f"{info['n']:,} plates", f"{info['n_feat']} measurements", "7 fault types", "10 algorithms"])
    tab_pred, tab_cmp, tab_in = st.tabs(["🔮 Predict a fault", "🏆 Compare all 10 models", "🔬 Look inside a model"])

    # ---- session defaults (steel type A300/A400 is shown as one choice) ----
    def put(name, v):
        f = fmap[name]
        st.session_state[f"c_{name}"] = (int(round(v)) if f["binary"] or is_int(f) and f["max"] - f["min"] <= 255
                                         else round(float(v), 3) if f["max"] - f["min"] <= 255 else float(v))

    def defaults(row=None):
        for f in feats:
            v = f["med"] if row is None or pd.isna(row[f["name"]]) else row[f["name"]]
            if f["name"] not in ("TypeOfSteel_A300", "TypeOfSteel_A400"):
                put(f["name"], v)
        a300 = (fmap["TypeOfSteel_A300"]["med"] if row is None else row["TypeOfSteel_A300"])
        st.session_state["c_steel"] = "A300" if a300 == 1 else "A400"

    if "c_steel" not in st.session_state:
        defaults()

    def reset():
        defaults(); st.session_state.pop("actual", None)

    def load_sample():
        want = st.session_state.get("want", "Any")
        pool = art["samples"] if want == "Any" else art["samples"][art["samples"]["Actual"] == want]
        r = pool.sample(1).iloc[0]
        defaults(r); st.session_state["actual"] = r["Actual"]

    # ═══ Tab 1: predict ═══
    with tab_pred:
        step(1, "Pick a model", f"Not sure? Keep the default. {best} scored best on the test plates.")
        a, b, c1, c2 = st.columns([2.2, 2.2, 1.3, 1.1], vertical_alignment="bottom")
        model = a.selectbox("Model", list(table.index), format_func=lambda m: f"🏆 {m} (best)" if m == best else m, label_visibility="collapsed")
        b.selectbox("Quick start: load a real test plate of type", ["Any"] + classes, key="want",
                    format_func=lambda x: x if x == "Any" else f"{ml.FAULT_INFO[x][0]} {x}")
        c1.button("🎲 Load a real plate", on_click=load_sample, width="stretch")
        c2.button("↺ Reset", on_click=reset, width="stretch")
        st.markdown('<div class="note">💡 <b>Easiest way to try it:</b> press <b>Load a real plate</b>. It fills every box with a real measured plate, '
                    'and the app tells you the true fault so you can check the answer.</div>', unsafe_allow_html=True)

        step(2, "Describe the defect", "Move the sliders or type numbers. Hover the (?) on any box to see what it means.")
        left, right = st.columns([3, 1.6], gap="large")
        values = {}
        groups = {}
        for name, (lab, hlp, grp) in ml.FEATURE_INFO.items():
            groups.setdefault(grp, []).append(name)
        with left:
            for g, names in groups.items():
                with st.expander(g, expanded=g.startswith(("📏", "💡", "🔩"))):
                    cols = st.columns(3)
                    i = 0
                    for n in names:
                        lab, hlp, _ = ml.FEATURE_INFO[n]
                        f = fmap[n]
                        if n == "TypeOfSteel_A300":
                            ch = cols[i % 3].radio("Steel type", ["A300", "A400"], horizontal=True, key="c_steel",
                                                   help="Which kind of steel the plate is made of.")
                            values["TypeOfSteel_A300"], values["TypeOfSteel_A400"] = int(ch == "A300"), int(ch == "A400")
                            i += 1
                            continue
                        if n == "TypeOfSteel_A400":
                            continue
                        col, key = cols[i % 3], f"c_{n}"
                        i += 1
                        rng = f["max"] - f["min"]
                        tip = f"{hlp}  Seen in the data: {f['min']:g} to {f['max']:g}."
                        if f["binary"]:
                            values[n] = col.radio(lab, [0, 1], horizontal=True, key=key, help=tip)
                        elif rng <= 255:
                            if is_int(f):
                                values[n] = col.slider(lab, int(f["min"]), int(f["max"]), step=1, key=key, help=tip)
                            else:
                                lo, hi = math.floor(f["min"] * 1000) / 1000, math.ceil(f["max"] * 1000) / 1000
                                values[n] = col.slider(lab, lo, hi, step=0.001, key=key, help=tip)
                        else:
                            values[n] = col.number_input(lab, key=key, format="%.0f", step=1.0, help=tip)
        res = ml.cls_predict(art, model, values)
        label, probs = res["label"], res["probs"]
        conf = max(probs.values())
        with right:
            st.markdown('<span class="sticky"></span>', unsafe_allow_html=True)
            emoji, desc = ml.FAULT_INFO[label]
            st.markdown(f'<div class="result"><div class="e">{emoji}</div><div class="lab">{label.replace("_", " ")}</div>'
                        f'<div class="sub">{desc}</div></div>', unsafe_allow_html=True)
            g = go.Figure(go.Indicator(mode="gauge+number", value=conf * 100, number=dict(suffix="%", font=dict(size=34)),
                                       title=dict(text=f"Confidence ({model})", font=dict(size=13, color=MUTED)),
                                       gauge=dict(axis=dict(range=[0, 100]), bar=dict(color="#8b5cf6"), bgcolor="rgba(255,255,255,.05)",
                                                  steps=[dict(range=[0, 50], color="rgba(251,113,133,.25)"),
                                                         dict(range=[50, 80], color="rgba(251,191,36,.22)"),
                                                         dict(range=[80, 100], color="rgba(52,211,153,.25)")])))
            show(g, 210, False)
            if "actual" in st.session_state:
                ok = st.session_state["actual"] == label
                st.markdown(f"**Real fault of this plate:** {ml.FAULT_INFO[st.session_state['actual']][0]} {st.session_state['actual']} "
                            f"{'✅ correct!' if ok else '❌ the model got it wrong'}")
            p = pd.Series(probs).sort_values()
            f1 = px.bar(x=p.values, y=p.index, orientation="h", labels={"x": "Probability", "y": ""},
                        color=(p.index == label), color_discrete_map={True: "#8b5cf6", False: "#2a3157"})
            f1.update_layout(xaxis_tickformat=".0%", title="How likely is each fault?")
            show(f1, 280, False)
            agree = sum(v == label for v in res["votes"].values())
            st.markdown(f"**🗳️ {agree} of {len(res['votes'])} models agree**")
            st.markdown("".join(f'<span class="vote {"ok" if v == label else ""}"><b>{n}</b>: {v.replace("_", " ")}</span>'
                                for n, v in res["votes"].items()), unsafe_allow_html=True)

    # ═══ Tab 2: compare ═══
    with tab_cmp:
        top = table.iloc[0]
        k = st.columns(4)
        kpi(k[0], "Best model", best, "highest F1", "a")
        kpi(k[1], "Best accuracy", f"{table['Accuracy'].max():.1%}", table["Accuracy"].idxmax(), "b")
        kpi(k[2], "Best ROC-AUC", f"{table['ROC-AUC'].max():.3f}", table["ROC-AUC"].idxmax(), "c")
        kpi(k[3], "Test plates", f"{info['n_test']:,}", f"trained on {info['n_train']:,}", "d")
        st.write("")
        step("📋", "One table, all 10 algorithms", "Accuracy, Precision, Recall, F1 and ROC-AUC. Sorted by F1 (top = best). Precision, Recall, F1 and ROC-AUC are weighted averages over the 7 faults.")
        t = table.copy()
        t.insert(0, "Rank", range(1, len(t) + 1))
        cfg = {m: st.column_config.ProgressColumn(m, min_value=0.0, max_value=1.0, format="%.3f") for m in table.columns}
        cfg["Rank"] = st.column_config.NumberColumn("#", width="small")
        st.dataframe(t, column_config=cfg, width="stretch", height=len(t) * 36 + 40)
        st.caption("Bars fill up to 1.0. Longer bar = better.")

        l, r = st.columns(2)
        metrics = list(table.columns)
        long = table.reset_index().melt(id_vars="Model", var_name="Metric", value_name="Score")
        gb = px.bar(long, x="Model", y="Score", color="Metric", barmode="group", title="All five metrics, every model")
        gb.update_layout(yaxis_range=[0, 1.05], xaxis_tickangle=-35)
        with l:
            show(gb, 440)
        hm = px.imshow(table[metrics].T, text_auto=".2f", aspect="auto", color_continuous_scale=["#1b1f4a", "#8b5cf6", "#22d3ee"],
                       zmin=float(table.values.min()) - 0.05, zmax=1, title="Heatmap (brighter = better)")
        hm.update_xaxes(tickangle=-35)
        with r:
            show(hm, 440, False)

        l, r = st.columns(2)
        with l:
            sel = st.multiselect("Models to draw on the radar", list(table.index), default=list(table.index[:4]))
            rad = go.Figure()
            for i, m in enumerate(sel):
                v = table.loc[m, metrics].tolist()
                rad.add_trace(go.Scatterpolar(r=v + v[:1], theta=metrics + metrics[:1], name=m, fill="toself", opacity=.55,
                                              line=dict(color=PALETTE[i % 10])))
            rad.update_layout(polar=dict(bgcolor="rgba(0,0,0,0)", radialaxis=dict(range=[max(0, table.values.min() - .05), 1],
                                                                                   gridcolor=GRID), angularaxis=dict(gridcolor=GRID)),
                              title="Radar: which model is strongest where?")
            show(rad, 440)
        with r:
            st.write(""); st.write("")
            mr = macro_rocs(art, tuple(table.index))
            roc = go.Figure()
            roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(dash="dot", color=MUTED), name="Random guess"))
            for i, m in enumerate(table.index):
                grid, tpr, a_ = mr[m]
                roc.add_trace(go.Scatter(x=grid, y=tpr, mode="lines", name=f"{m} ({a_:.3f})", line=dict(color=PALETTE[i % 10], width=2)))
            roc.update_layout(title="ROC curves, macro-average of the 7 faults", xaxis_title="False positive rate",
                              yaxis_title="True positive rate")
            show(roc, 440)

    # ═══ Tab 3: inspect ═══
    with tab_in:
        pick = st.selectbox("Choose a model to inspect", list(table.index), key="inspect")
        l, r = st.columns(2)
        norm = st.toggle("Show confusion matrix as % of each real fault", value=False)
        cm = art["cms"][pick].astype(float)
        z = cm / cm.sum(1, keepdims=True) if norm else cm
        lab = [c.replace("_", " ") for c in classes]
        cmf = px.imshow(z, x=lab, y=lab, text_auto=".0%" if norm else True, color_continuous_scale=["#10143a", "#8b5cf6", "#22d3ee"],
                        labels=dict(x="Predicted fault", y="Real fault", color=""), title="Confusion matrix")
        cmf.update_layout(coloraxis_showscale=False)
        with l:
            show(cmf, 470, False)
        rep = art["reports"][pick]
        rl = rep.melt(id_vars=["Fault", "Plates"], value_vars=["Precision", "Recall", "F1"], var_name="Metric", value_name="Score")
        rb = px.bar(rl, x="Fault", y="Score", color="Metric", barmode="group", title="Precision / Recall / F1 for each fault")
        rb.update_layout(yaxis_range=[0, 1.05])
        with r:
            show(rb, 470)
        l, r = st.columns(2)
        curves, (g_, t_, a_) = ml.roc_curves(art, pick)
        rc = go.Figure()
        rc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(dash="dot", color=MUTED), name="Random guess"))
        for i, (cn, (fpr, tpr, au)) in enumerate(curves.items()):
            rc.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"{cn} ({au:.2f})", line=dict(color=PALETTE[i % 10])))
        rc.update_layout(title=f"ROC curve for each fault ({pick})", xaxis_title="False positive rate", yaxis_title="True positive rate")
        with l:
            show(rc, 400)
        imp = art["importance"].head(12).sort_values()
        names = [ml.FEATURE_INFO.get(n, (n,))[0] for n in imp.index]
        fi = px.bar(x=imp.values, y=names, orientation="h", labels={"x": "Importance", "y": ""},
                    title="What matters most? (Random Forest importance)", color=imp.values, color_continuous_scale=["#2a3157", "#8b5cf6", "#22d3ee"])
        fi.update_layout(coloraxis_showscale=False)
        with r:
            show(fi, 400, False)
        cc = art["class_counts"]
        dn = go.Figure(go.Pie(labels=[k.replace("_", " ") for k in cc], values=list(cc.values()), hole=.62, marker=dict(colors=PALETTE)))
        dn.update_layout(title="How many plates of each fault are in the data?")
        show(dn, 360)


# ═════════════════════════ Clustering ═════════════════════════
def get_raw():
    if os.path.exists(ml.CLU_CSV):
        with open(ml.CLU_CSV, "rb") as fh:
            return fh.read()
    if "clu_raw" in st.session_state:
        return st.session_state["clu_raw"]
    st.markdown('<div class="note">📂 <b>One thing is missing:</b> the customer file. Drop <b>marketing_campaign.csv</b> below '
                '(or put it in the <code>data/</code> folder and restart the app).</div>', unsafe_allow_html=True)
    up = st.file_uploader("Upload marketing_campaign.csv", type=["csv", "tsv", "txt"])
    if up:
        st.session_state["clu_raw"] = up.getvalue()
        st.rerun()
    return None


def page_clustering():
    hero("🧩 Customer Segmentation Studio",
         "Group customers by income, spending and shopping habits using two clustering algorithms. See how good each grouping is, "
         "how to pick the number of groups, and where every customer lands on a map.",
         ["K-Means", "Hierarchical (Ward)", "Silhouette · Davies-Bouldin · Calinski-Harabasz", "PCA + t-SNE maps"])
    raw = get_raw()
    if raw is None:
        return
    try:
        c = get_clu(raw)
    except ValueError as e:
        st.error(str(e)); st.session_state.pop("clu_raw", None); return

    keys = list(ml.ALGOS)
    step(1, "How many customer groups do you want?", f"The slider starts at {c.best_k}, the number with the best K-Means silhouette score.")
    k = st.slider("Number of segments (k)", 2, 10, c.best_k, label_visibility="collapsed")
    fits = {a: ml.fit(c, a, k) for a in keys}
    segs = [f"Segment {i + 1}" for i in range(k)]
    cmap = {s: PALETTE[i % 10] for i, s in enumerate(segs)}

    def scatter(emb, a, title, star=None):
        d = pd.DataFrame({"x": emb[:, 0], "y": emb[:, 1], "Segment": [segs[i] for i in fits[a]["labels"]]})
        fig = px.scatter(d, x="x", y="y", color="Segment", category_orders={"Segment": segs}, color_discrete_map=cmap, opacity=.8,
                         title=title, labels={"x": "", "y": ""})
        fig.update_traces(marker=dict(size=6, line=dict(width=0)))
        if star is not None:
            fig.add_trace(go.Scatter(x=[star[0]], y=[star[1]], mode="markers", name="This customer",
                                     marker=dict(symbol="star", size=22, color="#fff", line=dict(color="#000", width=1.5))))
        fig.update_xaxes(showticklabels=False); fig.update_yaxes(showticklabels=False)
        return fig

    tabs = st.tabs(["📊 Scores", "📐 Choose k", "🗺️ Cluster maps", "👥 Who is in each segment", "🔮 Place a new customer"])

    # ── Scores ──
    with tabs[0]:
        M = pd.DataFrame({ml.ALGOS[a]: fits[a]["metrics"] for a in keys}).T
        k1 = st.columns(3)
        wins = {"Silhouette": M["Silhouette"].idxmax(), "Davies-Bouldin": M["Davies-Bouldin"].idxmin(),
                "Calinski-Harabasz": M["Calinski-Harabasz"].idxmax()}
        for col, (m, w), tone in zip(k1, wins.items(), "abc"):
            kpi(col, f"{m} winner", w, f"{'lower' if m == 'Davies-Bouldin' else 'higher'} is better", tone)
        st.write("")
        step("📋", f"Quality scores at k = {k}", "Three standard ways to judge a clustering. The green cell is the better algorithm for that score.")
        sty = (M.style.format("{:.3f}", subset=["Silhouette", "Davies-Bouldin"]).format("{:,.1f}", subset=["Calinski-Harabasz"])
               .highlight_max(subset=["Silhouette", "Calinski-Harabasz"], color="#065f46")
               .highlight_min(subset=["Davies-Bouldin"], color="#065f46"))
        st.dataframe(sty, width="stretch")
        with st.expander("🤔 What do these three scores mean?"):
            st.markdown("- **Silhouette** (−1 to 1, higher is better): how well each customer fits its own segment versus the nearest other one.\n"
                        "- **Davies-Bouldin** (0 and up, **lower** is better): how much segments overlap. Compact, well-separated groups score low.\n"
                        "- **Calinski-Harabasz** (higher is better): ratio of between-group spread to within-group spread.")
        cols = st.columns(3)
        for col, m in zip(cols, M.columns):
            f = px.bar(M.reset_index(), x="index", y=m, color="index", color_discrete_sequence=PALETTE, text_auto=".3f",
                       labels={"index": "", m: m}, title=m + (" (lower = better)" if m == "Davies-Bouldin" else " (higher = better)"))
            f.update_traces(textposition="outside")
            with col:
                show(f, 330, False)
        # metrics over every k
        st.markdown("##### How each score changes with k")
        ks = ml.KRANGE
        allm = {a: pd.DataFrame({kk: ml.fit(c, a, kk)["metrics"] for kk in ks}).T for a in keys}
        cols = st.columns(3)
        for col, m in zip(cols, M.columns):
            f = go.Figure()
            for i, a in enumerate(keys):
                f.add_trace(go.Scatter(x=ks, y=allm[a][m], mode="lines+markers", name=ml.ALGOS[a], line=dict(color=PALETTE[i], width=3)))
            f.add_vline(x=k, line_dash="dot", line_color="#fff")
            f.update_layout(title=m, xaxis_title="k")
            with col:
                show(f, 310)

    # ── Choose k ──
    with tabs[1]:
        st.markdown('<div class="note">📐 <b>How to read these:</b> in the elbow chart, pick the k where the line stops dropping steeply (the "elbow"). '
                    'In the dendrogram, cut the tree with a horizontal line: the number of branches it crosses is the number of segments. The dotted line shows your current k.</div>',
                    unsafe_allow_html=True)
        l, r = st.columns(2)
        e = go.Figure(go.Scatter(x=list(c.inertia), y=list(c.inertia.values()), mode="lines+markers",
                                 line=dict(color=PALETTE[0], width=3), marker=dict(size=9), fill="tozeroy", fillcolor="rgba(139,92,246,.12)"))
        e.add_vline(x=k, line_dash="dot", line_color="#fff", annotation_text=f"k = {k}")
        e.update_layout(title="K-Means elbow curve", xaxis_title="Number of segments (k)", yaxis_title="Inertia (lower = tighter groups)")
        with l:
            show(e, 400, False)
        ic, dc, cut, colors = ml.dendro(c, k)
        d = go.Figure()
        shade = {}
        for x, y, col in zip(ic, dc, colors):
            if col not in shade:
                shade[col] = "#64748b" if col == "C0" else PALETTE[(len(shade)) % 10]
            d.add_trace(go.Scatter(x=x, y=y, mode="lines", line=dict(color=shade[col], width=1.6), hoverinfo="skip", showlegend=False))
        d.add_hline(y=cut, line_dash="dot", line_color="#fff", annotation_text=f"cut → {k} segments")
        d.update_layout(title="Hierarchical (Ward) dendrogram, last 40 merges", yaxis_title="Distance between merged groups",
                        xaxis=dict(showticklabels=False, title="Customer groups (leaves)"))
        with r:
            show(d, 400, False)
        s_df = pd.DataFrame({ml.ALGOS[a]: c.sil[a] for a in keys})
        s = go.Figure()
        for i, col in enumerate(s_df.columns):
            s.add_trace(go.Scatter(x=s_df.index, y=s_df[col], mode="lines+markers", name=col, line=dict(color=PALETTE[i], width=3)))
        s.add_vline(x=k, line_dash="dot", line_color="#fff")
        s.update_layout(title="Silhouette score for every k (the peak is the best k)", xaxis_title="k", yaxis_title="Silhouette")
        show(s, 340)
        st.caption(f"Best K-Means silhouette is at k = {c.best_k}.")

    # ── Maps ──
    with tabs[2]:
        st.markdown("##### PCA maps (instant): every dot is a customer, coloured by segment")
        cols = st.columns(2)
        for col, a in zip(cols, keys):
            with col:
                show(scatter(c.emb["PCA"], a, f"{ml.ALGOS[a]} · PCA"), 470)
        ev = c.pca.explained_variance_ratio_
        st.caption(f"The two PCA axes keep {ev.sum():.0%} of the information in the data.")
        st.markdown("##### t-SNE maps (shows tighter groups)")
        if "t-SNE" in c.emb or st.session_state.get("do_tsne"):
            with st.spinner("Computing t-SNE, about 30 seconds the first time..."):
                emb = ml.tsne(c)
            cols = st.columns(2)
            for col, a in zip(cols, keys):
                with col:
                    show(scatter(emb, a, f"{ml.ALGOS[a]} · t-SNE"), 470)
        else:
            st.button("✨ Compute t-SNE maps (takes ~30 s)", on_click=lambda: st.session_state.update(do_tsne=True))

    # ── Profiles ──
    with tabs[3]:
        a = keys[[ml.ALGOS[x] for x in keys].index(st.radio("Algorithm", [ml.ALGOS[x] for x in keys], horizontal=True))]
        T = fits[a]["table"]
        l, r = st.columns([1, 2])
        dn = go.Figure(go.Pie(labels=T["Segment"], values=T["Customers"], hole=.6, marker=dict(colors=[cmap[s] for s in T["Segment"]])))
        dn.update_layout(title="Segment sizes")
        with l:
            show(dn, 360)
        pc = [v for v in ml.PROFILE.values()]
        Z = (T[pc] - T[pc].min()) / (T[pc].max() - T[pc].min()).replace(0, 1)
        rad = go.Figure()
        for i, s in enumerate(T["Segment"]):
            v = Z.iloc[i].tolist()
            rad.add_trace(go.Scatterpolar(r=v + v[:1], theta=pc + pc[:1], name=s, fill="toself", opacity=.5, line=dict(color=cmap[s])))
        rad.update_layout(polar=dict(bgcolor="rgba(0,0,0,0)", radialaxis=dict(visible=False, range=[0, 1]), angularaxis=dict(gridcolor=GRID)),
                          title="Segment fingerprints (relative scale)")
        with r:
            show(rad, 360)
        st.dataframe(T.style.format({"Customers": "{:,.0f}", "Avg income": "{:,.0f}", "Avg total spend": "{:,.0f}", "Avg purchases": "{:.1f}",
                                     "Avg age": "{:.0f}", "Avg children": "{:.1f}"}), hide_index=True, width="stretch")

    # ── Predict ──
    with tabs[4]:
        step(1, "Choose the algorithm", "The customer joins the closest segment centre of the chosen algorithm.")
        a = keys[[ml.ALGOS[x] for x in keys].index(st.radio("Algorithm for placing", [ml.ALGOS[x] for x in keys], horizontal=True,
                                                            key="pa", label_visibility="collapsed"))]
        for f in c.fields:
            st.session_state.setdefault(f"u_{f['name']}", float(round(f["med"])) if f["kind"] == "money" else int(round(f["med"])))
        for n in c.cat_c:
            st.session_state.setdefault(f"u_{n}", c.cats[n]["mode"])

        def setv(r=None):
            for f in c.fields:
                v = f["med"] if r is None or pd.isna(r[f["name"]]) else r[f["name"]]
                st.session_state[f"u_{f['name']}"] = float(round(v)) if f["kind"] == "money" else int(round(v))
            for n in c.cat_c:
                st.session_state[f"u_{n}"] = c.cats[n]["mode"] if r is None else str(r[n])

        b1, b2, _ = st.columns([1.3, 1, 4])
        b1.button("🎲 Random customer", on_click=lambda: setv(c.f.sample(1).iloc[0]), width="stretch")
        b2.button("↺ Reset", on_click=setv, width="stretch")
        step(2, "Describe the customer", "Sliders show the range found in your data.")
        left, right = st.columns([3, 1.6], gap="large")
        vals = {}
        fm = {f["name"]: f for f in c.fields}
        sections = {"👤 About the customer": ["Income", "Age", "Kidhome", "Teenhome"],
                    "🛒 Money spent in the last 2 years": ["MntWines", "MntFruits", "MntMeatProducts", "MntFishProducts", "MntSweetProducts", "MntGoldProds"],
                    "🏬 Shopping habits": ["NumStorePurchases", "NumWebPurchases", "NumCatalogPurchases", "NumDealsPurchases", "NumWebVisitsMonth", "Recency"]}
        with left:
            for title, names in sections.items():
                with st.expander(title, expanded=True):
                    cc = st.columns(3)
                    j = 0
                    for n in names:
                        if n not in fm:
                            continue
                        f = fm[n]
                        if f["kind"] == "money":
                            vals[n] = cc[j % 3].number_input(f["label"], min_value=0.0, step=1000.0, format="%.0f", key=f"u_{n}",
                                                             help=f"In your data: {f['min']:,.0f} to {f['max']:,.0f}")
                        else:
                            hi = int(max(f["max"], st.session_state[f"u_{n}"]))
                            vals[n] = cc[j % 3].slider(f["label"], int(min(f["min"], 0)), hi, key=f"u_{n}",
                                                       help=f"In your data: {f['min']:g} to {f['max']:g}")
                        j += 1
                    if title.startswith("👤") and c.cat_c:
                        for n in c.cat_c:
                            vals[n] = cc[j % 3].selectbox(n.replace("_", " "), c.cats[n]["opts"], key=f"u_{n}")
                            j += 1
        res = ml.clu_predict(c, a, k, vals)
        fi = fits[a]
        with right:
            st.markdown('<span class="sticky"></span>', unsafe_allow_html=True)
            st.markdown(f'<div class="result"><div class="e">🎯</div><div class="lab">Segment {res["seg"] + 1}</div>'
                        f'<div class="sub">{res["tag"]}</div></div>', unsafe_allow_html=True)
            seg_row, dd = fi["table"].iloc[res["seg"]], res["derived"]
            cmpd = pd.DataFrame({"This customer": [dd["Income"], dd["TotalSpend"], dd["TotalPurchases"], dd["Age"]],
                                 "Segment average": [seg_row["Avg income"], seg_row["Avg total spend"], seg_row["Avg purchases"], seg_row["Avg age"]]},
                                index=["Income", "Total spend", "Purchases", "Age"])
            st.dataframe(cmpd.style.format("{:,.0f}"), width="stretch")
            dist = pd.Series(res["dist"], index=segs)
            f = px.bar(x=dist.index, y=dist.values, labels={"x": "", "y": "Distance"}, title="Distance to each segment (smaller = closer)",
                       color=(dist.index == segs[res["seg"]]), color_discrete_map={True: "#8b5cf6", False: "#2a3157"})
            show(f, 260, False)
        show(scatter(c.emb["PCA"], a, f"Where this customer lands (white star) · {ml.ALGOS[a]}", res["pca"]), 460)


# ═════════════════════════ Navigation ═════════════════════════
with st.sidebar:
    st.markdown('<div class="brand">🧠 ML Studio</div>', unsafe_allow_html=True)
    st.caption("Predict, compare, explore.")
    page = st.radio("What do you want to do?", ["🔩  Classify steel faults", "🧩  Segment customers"])
    st.divider()
    st.markdown("**🔩 Classification**\n\nCompare 10 algorithms and predict a plate's fault.\n\n"
                "**🧩 Clustering**\n\nK-Means vs Hierarchical with 3 quality scores, elbow, dendrogram, PCA and t-SNE.")

page_classification() if page.startswith("🔩") else page_clustering()
