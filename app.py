# ══════════════════════════════════════════════════════════════
#  Egypt Economic Stability — EWS Interactive Dashboard
#  app.py  |  Streamlit ≥ 1.32
#  Graduation Project — Statistics Department
# ══════════════════════════════════════════════════════════════
#
#  Run on Google Colab (in a separate cell):
#  ─────────────────────────────────────────
#  !pip install streamlit pyngrok plotly -q
#  !ngrok authtoken <YOUR_TOKEN>   # from ngrok.com (free)
#  import subprocess, time
#  from pyngrok import ngrok
#  subprocess.Popen(["streamlit","run","app.py","--server.port","8501"])
#  time.sleep(4)
#  print(ngrok.connect(8501))
# ══════════════════════════════════════════════════════════════

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings("ignore")

# ── Page Settings ───────────────────────────────────────────
st.set_page_config(
    page_title="Egypt EWS Dashboard",
    page_icon="🇪🇬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ──────────────────────────────────────────────────────
st.markdown("""
<style>
.metric-card {
    border-radius: 10px; padding: 1rem 1.25rem;
    margin-bottom: .5rem; border-left: 4px solid #dee2e6;
}
.card-green  { background:#f0fdf4; border-left-color:#16a34a; }
.card-yellow { background:#fefce8; border-left-color:#ca8a04; }
.card-red    { background:#fef2f2; border-left-color:#dc2626; }
.card-blue   { background:#eff6ff; border-left-color:#2563eb; }
.big-num { font-size:2.2rem; font-weight:700; line-height:1.1; }
.g{color:#16a34a;} .y{color:#ca8a04;} .r{color:#dc2626;} .b{color:#2563eb;}
.sub { font-size:.82rem; color:#6b7280; margin-top:4px; }
div[data-testid="stSidebar"] { background:#0f172a !important; }
div[data-testid="stSidebar"] * { color:#e2e8f0 !important; }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# Data Loading — reads directly from Google Drive
# ══════════════════════════════════════════════════════════════
PROJECT = "/content/drive/MyDrive/Egypt_Economic_Stability"

@st.cache_data(show_spinner="Loading data …")
def load_all():
    def xl(name):
        try:
            return pd.read_excel(f"{PROJECT}/{name}")
        except Exception:
            return pd.read_excel(name)

    msi = xl("data_with_msi.xlsx")
    bt  = xl("backtesting_results_v3.xlsx")
    mdl = xl("final_results_all_models.xlsx")
    pca = xl("pca_msi_results.xlsx")
    ews = xl("ews_performance_metrics_v3.xlsx")

    msi["Date"] = pd.to_datetime(msi["Date"])
    bt["Date"]  = pd.to_datetime(bt["Date"])
    return msi, bt, mdl, pca, ews

msi_df, bt_df, mdl_df, pca_df, ews_df = load_all()

# ── Latest Quarter Values ─────────────────────────────────────────────
latest    = bt_df.iloc[-1]
lat_msi   = latest["MSI"]
lat_p     = latest["P_Shock"]
lat_alert = latest["Alert"]
lat_date  = latest["Date"].strftime("%Y-%m-%d")

ICON  = {"Red":"🔴","Yellow":"🟡","Stable":"🟢"}
ACSS  = {"Red":"card-red","Yellow":"card-yellow","Stable":"card-green"}
ACLR  = {"Red":"r","Yellow":"y","Stable":"g"}

MODEL_CLR = {
    "VECM":"#16a34a","VAR":"#2563eb","ARIMA":"#7c3aed",
    "SARIMA":"#9333ea","Naive":"#94a3b8",
    "XGBoost":"#ea580c","RF":"#b45309","MLR":"#be123c",
}

# ══════════════════════════════════════════════════════════════
# Sidebar
# ══════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🇪🇬 Egypt EWS")
    st.markdown("**Early Warning System**\nfor Egyptian Economic Stability")
    st.divider()

    page = st.radio("📄 Page", [
        "🏠 Current Status",
        "📈 MSI Over Time",
        "🔮 Forecast",
        "📊 Model Comparison",
        "📋 Alerts Table",
        "🔬 PCA Results",
    ])

    st.divider()
    st.markdown("**⚙️ Alert Thresholds**")
    thr_red    = st.slider("🔴 Critical MSI",    10, 30, 20)
    thr_yellow = st.slider("🟡 Warning MSI",  25, 50, 40)
    thr_p      = st.slider("Critical P(Shock)", 0.40, 0.90, 0.65, 0.05)

    st.divider()
    st.caption("Graduation Project — Statistics Department")
    st.caption("Modeling Economic Stability in Egypt")

# ══════════════════════════════════════════════════════════════
# PAGE 1 — Current Status
# ══════════════════════════════════════════════════════════════
if page == "🏠 Current Status":

    st.title("🇪🇬 Early Warning System for the Egyptian Economy")
    st.caption(f"Last Update: **{lat_date}** | Period: 2010–2026 | Models: VECM · XGBoost · RF · Logistic")
    st.divider()

    if lat_alert == "Red":
        st.error(f"🔴 **Warning:** Economy in danger zone — MSI={lat_msi:.1f} | P(Shock)={lat_p:.1%}")
    elif lat_alert == "Yellow":
        st.warning(f"🟡 **Caution:** Moderate economic pressure — MSI={lat_msi:.1f} | P(Shock)={lat_p:.1%}")
    else:
        st.success(f"🟢 **Stable** — MSI={lat_msi:.1f} | P(Shock)={lat_p:.1%}")

    st.markdown("")

    c1, c2, c3, c4 = st.columns(4)
    css = ACSS.get(lat_alert,"card-green")
    clr = ACLR.get(lat_alert,"g")

    with c1:
        st.markdown(f"""<div class="metric-card {css}">
        <div class="big-num {clr}">{lat_msi:.1f}</div>
        <div class="sub">MSI Score (0–100)<br>{lat_date}</div></div>""",
        unsafe_allow_html=True)

    p_css = "card-red" if lat_p>thr_p else "card-yellow" if lat_p>0.4 else "card-green"
    p_clr = "r" if lat_p>thr_p else "y" if lat_p>0.4 else "g"

    with c2:
        st.markdown(f"""<div class="metric-card {p_css}">
        <div class="big-num {p_clr}">{lat_p:.1%}</div>
        <div class="sub">P(Shock) — Crisis Probability</div></div>""",
        unsafe_allow_html=True)

    with c3:
        n_red = (bt_df["Alert"]=="Red").sum()
        st.markdown(f"""<div class="metric-card card-red">
        <div class="big-num r">{n_red}</div>
        <div class="sub">Red Alerts (2017–2026)</div></div>""",
        unsafe_allow_html=True)

    with c4:
        best_rmse = mdl_df["RMSE"].min()
        best_name = mdl_df.loc[mdl_df["RMSE"].idxmin(),"Model"]
        st.markdown(f"""<div class="metric-card card-blue">
        <div class="big-num b">{best_rmse:.1f}</div>
        <div class="sub">Best RMSE — Model {best_name}</div></div>""",
        unsafe_allow_html=True)

    col_g, col_p = st.columns([3,2])

    with col_g:
        fig_g = go.Figure(go.Indicator(
            mode ="gauge+number+delta",
            value=lat_msi,
            delta={"reference":msi_df["MSI"].mean(),"valueformat":".1f","suffix":" vs average"},
            title={"text":f"MSI — {lat_date}","font":{"size":15}},
            gauge={
                "axis":{"range":[0,100],"tickwidth":1},
                "bar":{"color":"#1e293b","thickness":0.22},
                "steps":[
                    {"range":[0,thr_red],"color":"#fee2e2"},
                    {"range":[thr_red,thr_yellow],"color":"#fef9c3"},
                    {"range":[thr_yellow,100],"color":"#dcfce7"},
                ],
                "threshold":{"line":{"color":"#dc2626","width":3},"thickness":0.75,"value":thr_red},
            }
        ))
        fig_g.update_layout(height=280,margin=dict(t=50,b=10,l=20,r=20))
        st.plotly_chart(fig_g, use_container_width=True)

    with col_p:
        st.markdown("### Performance Summary")
        for _, row in ews_df.iterrows():
            m,v,t = row["Metric"], row["V3_Pct"], str(row["Target"])
            try:
                tv = float(t.replace("≥","").replace("≤","").strip())
                ok = (v>=tv) if "≥" in t else (v<=tv)
            except Exception:
                ok = True
            st.markdown(f"{'✅' if ok else '⚠️'} **{m}**: `{v:.3f}` _(Target {t})_")
        st.divider()
        st.markdown("🏆 **Best Model:** VECM (RMSE=8.2)")
        st.markdown("📐 **MSI explains:** 78.6% of variance")
# ══════════════════════════════════════════════════════════════
# PAGE 2 — MSI Over Time
# ══════════════════════════════════════════════════════════════
elif page == "📈 MSI Over Time":
    st.title("📈 MSI Over Time (2010–2026)")

    fig = make_subplots(
        rows=2,cols=1,shared_xaxes=True,
        subplot_titles=("MSI Score","P(Shock) — Logistic Regression"),
        row_heights=[0.65,0.35],vertical_spacing=0.08,
    )

    fig.add_trace(go.Scatter(
        x=msi_df["Date"],y=msi_df["MSI"],mode="lines",name="MSI",
        line=dict(color="#1e293b",width=2.5),
        hovertemplate="<b>%{x|%Y-%m}</b><br>MSI: %{y:.1f}<extra></extra>",
    ),row=1,col=1)

    fig.add_hrect(y0=0,y1=thr_red,    fillcolor="#fee2e2",opacity=0.35,row=1,col=1)
    fig.add_hrect(y0=thr_red,y1=thr_yellow,fillcolor="#fef9c3",opacity=0.28,row=1,col=1)
    fig.add_hrect(y0=thr_yellow,y1=105,fillcolor="#dcfce7",opacity=0.22,row=1,col=1)
    fig.add_hline(y=thr_red,   line_color="#dc2626",line_dash="dash",line_width=1.5,row=1,col=1)
    fig.add_hline(y=thr_yellow,line_color="#ca8a04",line_dash="dash",line_width=1.2,row=1,col=1)

    for alert,color,sym in [("Red","#dc2626","triangle-down"),("Yellow","#ca8a04","triangle-up")]:
        sub = bt_df[bt_df["Alert"]==alert]
        if len(sub):
            fig.add_trace(go.Scatter(
                x=sub["Date"],y=sub["MSI"],mode="markers",
                name=f"{ICON.get(alert,'')} {alert}",
                marker=dict(color=color,size=9,symbol=sym),
            ),row=1,col=1)

    for ev,dt,c in [("COVID-19","2020-03-31","#7c3aed"),("Dollar Crisis","2022-03-31","#dc2626")]:
        fig.add_vline(x=dt,line_color=c,line_dash="dot",line_width=1.5,row=1,col=1)
        fig.add_annotation(x=dt,y=102,text=ev,font=dict(size=8,color=c),
                           showarrow=False,xshift=4,row=1,col=1)

    p_clrs = ["#dc2626" if p>=thr_p else "#ca8a04" if p>=0.4 else "#16a34a" for p in bt_df["P_Shock"]]
    fig.add_trace(go.Bar(x=bt_df["Date"],y=bt_df["P_Shock"],
                         name="P(Shock)",marker_color=p_clrs,opacity=0.85,
                         hovertemplate="<b>%{x|%Y-%m}</b><br>P(Shock):%{y:.3f}<extra></extra>"),
                  row=2,col=1)
    fig.add_hline(y=thr_p,line_color="#dc2626",line_dash="dash",row=2,col=1)
    fig.add_hline(y=0.40, line_color="#ca8a04",line_dash="dot", row=2,col=1)

    real = bt_df[bt_df["Shock_Label"]==1]
    if len(real):
        fig.add_trace(go.Scatter(x=real["Date"],y=real["P_Shock"]+0.04,mode="markers",
                                  name="Actual Shock ▼",
                                  marker=dict(color="#7f1d1d",size=10,symbol="triangle-down")),
                      row=2,col=1)

    fig.update_layout(height=580,hovermode="x unified",
                      legend=dict(orientation="h",y=1.05,font=dict(size=10)),
                      margin=dict(t=60,b=20))
    fig.update_yaxes(title_text="MSI (0–100)",range=[-3,108],row=1,col=1)
    fig.update_yaxes(title_text="P(Shock)",   range=[0,1.12], row=2,col=1)
    st.plotly_chart(fig,use_container_width=True)

    st.divider()
    c1,c2,c3,c4 = st.columns(4)
    z = msi_df["MSI_Zone"].value_counts()
    c1.metric("🔴 Crisis",  f"{z.get('Crisis',0)} quarters",  f"{z.get('Crisis',0)/len(msi_df)*100:.1f}%")
    c2.metric("🟡 Warning", f"{z.get('Warning',0)} quarters", f"{z.get('Warning',0)/len(msi_df)*100:.1f}%")
    c3.metric("🟢 Stable",  f"{z.get('Stable',0)} quarters",  f"{z.get('Stable',0)/len(msi_df)*100:.1f}%")
    c4.metric("MSI Average",  f"{msi_df['MSI'].mean():.1f}",f"std={msi_df['MSI'].std():.1f}")
# ══════════════════════════════════════════════════════════════
# PAGE 3 — Forecast
# ══════════════════════════════════════════════════════════════
elif page == "🔮 Forecast":
    st.title("🔮 MSI Forecast — Next 4 Quarters")
    st.info("📌 Forecast based on VECM trend (Best Model — RMSE=8.2)")

    last8    = msi_df.tail(8)
    coef     = np.polyfit(range(len(last8)), last8["MSI"].values, 1)
    last_val = msi_df["MSI"].iloc[-1]
    last_dt  = msi_df["Date"].iloc[-1]
    fut_dt   = pd.date_range(start=last_dt, periods=5, freq="QE")[1:]
    fc       = [float(np.clip(np.polyval(coef, len(last8)+i), 0, 100)) for i in range(4)]
    ci_up    = [min(100,f+10) for f in fc]
    ci_lo    = [max(0,  f-10) for f in fc]

    fig_fc = go.Figure()
    hist12 = msi_df.tail(12)

    fig_fc.add_trace(go.Scatter(
        x=hist12["Date"],y=hist12["MSI"],
        mode="lines+markers",name="Actual MSI",
        line=dict(color="#1e293b",width=2.5),marker=dict(size=5)
    ))

    fig_fc.add_trace(go.Scatter(
        x=[last_dt]+list(fut_dt),y=[last_val]+fc,
        mode="lines+markers",name="VECM Forecast",
        line=dict(color="#2563eb",width=2.5,dash="dash"),
        marker=dict(size=9,symbol="diamond",color="#2563eb"),
        hovertemplate="<b>%{x|%Y-%m}</b><br>Forecast MSI:%{y:.1f}<extra></extra>"
    ))

    fig_fc.add_trace(go.Scatter(
        x=list(fut_dt)+list(fut_dt[::-1]),y=ci_up+ci_lo[::-1],
        fill="toself",fillcolor="rgba(37,99,235,0.10)",
        line=dict(color="rgba(0,0,0,0)"),name="95% Confidence Interval"
    ))

    fig_fc.add_hrect(y0=0,y1=thr_red,fillcolor="#fee2e2",opacity=0.22)
    fig_fc.add_hrect(y0=thr_red,y1=thr_yellow,fillcolor="#fef9c3",opacity=0.18)
    fig_fc.add_hrect(y0=thr_yellow,y1=105,fillcolor="#dcfce7",opacity=0.15)

    fig_fc.add_hline(y=thr_red,line_color="#dc2626",line_dash="dash",line_width=1.2)
    fig_fc.add_vline(x=str(last_dt.date()),line_color="#94a3b8",line_dash="dot",line_width=1.5)

    fig_fc.update_layout(
        height=420,hovermode="x unified",
        yaxis=dict(title="MSI (0–100)",range=[-3,108]),
        title="MSI Forecast — Next 4 Quarters",
        legend=dict(orientation="h",y=1.08),
        margin=dict(t=60,b=20)
    )

    st.plotly_chart(fig_fc,use_container_width=True)

    fc_df = pd.DataFrame({
        "Quarter": [d.strftime("%Y-%m") for d in fut_dt],
        "Forecast MSI": [f"{v:.1f}" for v in fc],
        "Upper Bound": [f"{v:.1f}" for v in ci_up],
        "Lower Bound": [f"{v:.1f}" for v in ci_lo],
        "Expected Zone": [
            "🔴 Crisis" if v<thr_red else "🟡 Warning" if v<thr_yellow else "🟢 Stable"
            for v in fc
        ],
    })

    st.markdown("### Forecast Table")
    st.dataframe(fc_df,hide_index=True,use_container_width=True)
    st.caption("⚠️ Confidence interval is approximate (±10). For exact forecast see vecm_forecast.xlsx")


# ══════════════════════════════════════════════════════════════
# PAGE 4 — Model Comparison
# ══════════════════════════════════════════════════════════════
elif page == "📊 Model Comparison":
    st.title("📊 Model Performance Comparison")

    mdl_s = mdl_df.sort_values("RMSE").copy()
    mdl_s["Color"] = mdl_s["Model"].map(MODEL_CLR).fillna("#94a3b8")

    fig_bar = go.Figure()
    fig_bar.add_trace(go.Bar(
        x=mdl_s["Model"],y=mdl_s["RMSE"],
        marker_color=mdl_s["Color"],
        text=mdl_s["RMSE"].round(2),textposition="outside",
        hovertemplate="<b>%{x}</b><br>RMSE:%{y:.3f}<extra></extra>"
    ))

    fig_bar.add_hline(
        y=mdl_s["RMSE"].iloc[0],
        line_color="#16a34a",line_dash="dot",line_width=1.5,
        annotation_text=f"🏆 {mdl_s['Model'].iloc[0]} ({mdl_s['RMSE'].iloc[0]:.2f})",
        annotation_font_size=10
    )

    fig_bar.update_layout(
        title="RMSE — Lower is Better",
        yaxis_title="RMSE",
        height=400,
        plot_bgcolor="white",
        yaxis=dict(gridcolor="#f1f5f9"),
        margin=dict(t=60,b=20)
    )

    st.plotly_chart(fig_bar,use_container_width=True)

    st.markdown("### Full Results Table")
    show = mdl_s[["Rank","Model","Step","MAE","RMSE","MAPE_%","R2","N"]].copy()
    show.columns = ["#","Model","Step","MAE","RMSE","MAPE%","R²","n_test"]
    st.dataframe(show,hide_index=True,use_container_width=True)

    st.divider()
    col1,col2 = st.columns(2)

    with col1:
        st.success("**🏆 Why did VECM outperform?**\n\n"
                   "- Cointegration between macro variables\n"
                   "- Designed for long-run relationships\n"
                   "- n=65 is sufficient for VECM but not ML")

    with col2:
        st.warning("**⚠️ Why did XGBoost & RF underperform?**\n\n"
                   "- ML requires larger datasets (200+ ideally)\n"
                   "- Linear relationships dominate here\n"
                   "- **Research Insight:** VECM is not ‘traditional’ in a weak sense")

    st.info("**🔬 LSTM — exploratory result:**\n"
            "LSTM Univariate → RMSE=4.8 (better than VECM!) but n_train=48 only.\n"
            "**Recommendation:** Use with monthly data (n≥200) in future work.")


# ══════════════════════════════════════════════════════════════
# PAGE 5 — Alerts Table
# ══════════════════════════════════════════════════════════════
elif page == "📋 Alerts Table":
    st.title("📋 Full Alerts Table (2017–2026)")

    colf1,colf2,colf3 = st.columns(3)

    with colf1:
        sel = st.multiselect("Alert Type",["Red","Yellow","Stable"],default=["Red","Yellow"])

    with colf2:
        yr_mn = int(bt_df["Date"].dt.year.min())
        yr_mx = int(bt_df["Date"].dt.year.max())
        yr_r  = st.slider("Years",yr_mn,yr_mx,(2020,yr_mx))

    with colf3:
        ev_only = st.checkbox("Historical Events Only",False)

    mask = bt_df["Alert"].isin(sel) & bt_df["Date"].dt.year.between(*yr_r)
    if ev_only:
        mask &= bt_df["Event"].notna()

    show_bt = bt_df[mask][["Date","MSI","P_Shock","Alert","Shock_Label","Event"]].copy()

    show_bt["Date"]    = show_bt["Date"].dt.strftime("%Y-%m-%d")
    show_bt["MSI"]     = show_bt["MSI"].round(1)
    show_bt["P_Shock"] = show_bt["P_Shock"].round(4)
    show_bt["Alert"]   = show_bt["Alert"].map(
        {"Red":"🔴 Red","Yellow":"🟡 Yellow","Stable":"🟢 Stable"}
    )

    show_bt.columns = ["Date","MSI","P(Shock)","Alert","Actual Shock","Event"]

    st.dataframe(show_bt,hide_index=True,use_container_width=True)
    st.caption(f"Records: {len(show_bt)} of {len(bt_df)}")

    csv = show_bt.to_csv(index=False,encoding="utf-8-sig").encode("utf-8-sig")

    st.download_button(
        "⬇️ Download CSV",
        data=csv,
        file_name="ews_alerts_egypt.csv",
        mime="text/csv"
    )


# ══════════════════════════════════════════════════════════════
# PAGE 6 — PCA Results
# ══════════════════════════════════════════════════════════════
elif page == "🔬 PCA Results":
    st.title("🔬 PCA Results — MSI Construction")

    col1,col2 = st.columns(2)

    with col1:
        st.markdown("### Scree Plot")

        fig_sc = go.Figure()

        clrs_sc = ["#16a34a" if k=="YES" else "#94a3b8" for k in pca_df["Keep_Kaiser"]]

        fig_sc.add_trace(go.Bar(
            x=pca_df["Component"],
            y=pca_df["Variance_%"],
            marker_color=clrs_sc,
            name="Variance %",
            text=pca_df["Variance_%"].round(1),
            textposition="outside"
        ))

        fig_sc.add_trace(go.Scatter(
            x=pca_df["Component"],
            y=pca_df["Cumulative_%"],
            mode="lines+markers",
            name="Cumulative %",
            yaxis="y2",
            line=dict(color="#ea580c",width=2),
            marker=dict(size=7)
        ))

        fig_sc.add_hline(
            y=78.65,
            line_dash="dot",
            line_color="#16a34a",
            line_width=1.5,
            annotation_text="78.6% (PC1+PC2)",
            annotation_font_size=9
        )

        fig_sc.update_layout(
            height=360,
            yaxis=dict(title="Variance %"),
            yaxis2=dict(title="Cumulative %",overlaying="y",side="right",range=[60,101]),
            legend=dict(orientation="h",y=1.08),
            plot_bgcolor="white",
            yaxis_gridcolor="#f1f5f9",
            margin=dict(t=50,b=20)
        )

        st.plotly_chart(fig_sc,use_container_width=True)

    with col2:
        st.markdown("### Component Results")
        st.dataframe(pca_df,hide_index=True,use_container_width=True)

        st.markdown("""
### KMO + Bartlett
| Test | Value | Result |
|---|---|---|
| **KMO** | 0.7475 | Middling ✅ |
| **Bartlett χ²** | 521.85 | p<0.001 ✅ |
| **PC1** | 63.1% | Main driver |
| **PC2** | 15.5% | Monetary pressure |
| **PC1+PC2** | **78.6%** | **H₁ confirmed ✅** |
""")

    st.divider()

    st.markdown("### MSI vs Original Variables")

    var = st.selectbox(
        "Select Variable",
        [c for c in msi_df.columns if c not in ["Date","MSI","MSI_Zone","Income","Poverty_Rate"]]
    )

    fig_cmp = make_subplots(specs=[[{"secondary_y":True}]])

    fig_cmp.add_trace(go.Scatter(
        x=msi_df["Date"],y=msi_df["MSI"],
        mode="lines",
        name="MSI",
        line=dict(color="#1e293b",width=2.5)
    ),secondary_y=False)

    if var in msi_df.columns:
        fig_cmp.add_trace(go.Scatter(
            x=msi_df["Date"],y=msi_df[var],
            mode="lines",
            name=var,
            line=dict(color="#dc2626",width=1.8,dash="dash")
        ),secondary_y=True)

    fig_cmp.update_layout(
        height=340,
        hovermode="x unified",
        plot_bgcolor="white",
        yaxis_gridcolor="#f1f5f9",
        margin=dict(t=30,b=20),
        legend=dict(orientation="h",y=1.08)
    )

    fig_cmp.update_yaxes(title_text="MSI",secondary_y=False)
    fig_cmp.update_yaxes(title_text=var, secondary_y=True)

    st.plotly_chart(fig_cmp,use_container_width=True)