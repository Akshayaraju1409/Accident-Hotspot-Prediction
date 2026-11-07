"""M11: Streamlit dashboard.  Run:  streamlit run app.py"""
import datetime as dt
import folium
import pandas as pd
import streamlit as st
from streamlit_folium import st_folium
import config as C
from predict import Service, fetch_open_meteo, WeatherError

st.set_page_config(page_title="Accident Hotspot & Disaster Risk", layout="wide")
COLORS = {"Low": "#2e9e5b", "Medium": "#f0a30a", "High": "#d62828"}


@st.cache_resource
def svc():
    return Service()


try:
    S = svc()
except FileNotFoundError:
    st.error("Models not found. Run `python train.py` first."); st.stop()

st.title("Accident hotspot & weather-risk dashboard")
with st.sidebar:
    st.header("Forecast request")
    city = st.selectbox("City", S.cities, index=S.cities.index("Bangalore"))
    day = st.date_input("Date", dt.date(2024, 12, 15))
    hour = st.slider("Start hour (forecast covers the following hours)", 0, 23, 12)
    horizon = st.slider("Horizon (hours)", 1, 6, 6)
    wmode = st.radio("Weather", ["Scenario", "Auto (Open-Meteo live forecast)"],
                     help="Auto needs internet and a date within the next 2 days.")
    weather = st.selectbox("Weather scenario", ["clear", "rain", "fog"], index=1) if wmode == "Scenario" else None
    density = st.select_slider("Traffic density", ["low", "medium", "high"], value="medium")
    go = st.button("Run forecast", type="primary")

tab_map, tab_tbl, tab_why, tab_eval = st.tabs(["Map & alerts", "Ranked cells", "Why (SHAP)", "Model quality & data notes"])
start = pd.Timestamp(dt.datetime.combine(day, dt.time(hour)))

res = None
if go or "last" in st.session_state:
    if go:
        try:
            w = fetch_open_meteo(city, start, horizon) if weather is None else weather
            st.session_state["last"] = dict(res=S.forecast(city, start, w, density, horizon), city=city, start=start,
                                            w=w, density=density, horizon=horizon)
        except WeatherError as e:   # design rule: no weather -> no alert
            st.session_state.pop("last", None); st.error(f"{e} No alert was produced.")
        except ValueError as e:
            st.session_state.pop("last", None); st.error(str(e))
    L = st.session_state.get("last")
    res = L["res"] if L else None

with tab_map:
    if res is None:
        st.info("Choose a city, time and weather in the sidebar, then press **Run forecast**.")
    else:
        L = st.session_state["last"]
        hrs = sorted(res.hour_ts.unique())
        h = st.select_slider("Forecast hour", options=hrs, format_func=lambda t: pd.Timestamp(t).strftime("%d %b %H:%M"))
        sub = res[res.hour_ts == h]
        c1, c2, c3, c4 = st.columns(4)
        vc = sub.alert.value_counts()
        for col, lv in zip((c1, c2, c3), ("High", "Medium", "Low")):
            col.metric(f"{lv} cells", int(vc.get(lv, 0)))
        c4.metric("Weather", str(sub.weather.iloc[0]) + (" (hazardous)" if sub.hazardous.iloc[0] else ""))
        m = folium.Map([sub.lat_c.mean(), sub.lon_c.mean()], zoom_start=10, tiles="cartodbpositron")
        fa = folium.FeatureGroup("Alert level per cell", show=True)
        for r in sub.itertuples():
            folium.Rectangle([[r.lat_min, r.lon_min], [r.lat_min + r.dlat, r.lon_min + r.dlon]], color=COLORS[r.alert],
                             weight=0.5, fill=True, fill_opacity=0.55 if r.alert != "Low" else 0.18,
                             tooltip=f"cell {r.cell_id} | {r.alert} | P(crash, 6h block)={r.p_crash_block:.3f} "
                                     f"[{r.p_lo:.3f}-{r.p_hi:.3f}] | confidence {r.confidence}").add_to(fa)
        fa.add_to(m)
        hs = S.hotspots.merge(S.cells[["cell_id", "city"]], on="cell_id", suffixes=("", "_c"))
        hs = hs[hs.city == L["city"]]
        fg = folium.FeatureGroup("Gi* hotspots (FDR-significant)", show=False)
        for r in hs[hs.gi_hot == 1].itertuples():
            folium.Rectangle([[r.lat_min, r.lon_min], [r.lat_min + r.dlat, r.lon_min + r.dlon]], color="#6a00f4", weight=3, fill=False).add_to(fg)
        fg.add_to(m)
        fd = folium.FeatureGroup("DBSCAN hotspot cells", show=False)
        for r in hs[hs.db_hot == 1].itertuples():
            folium.Rectangle([[r.lat_min, r.lon_min], [r.lat_min + r.dlat, r.lon_min + r.dlon]], color="#0077b6", weight=3, dash_array="5", fill=False).add_to(fd)
        fd.add_to(m)
        act = S.actual_crashes(L["city"], L["start"], L["horizon"])
        fc = folium.FeatureGroup(f"Actual crashes in window (backtest: {len(act)})", show=True)
        for r in act.itertuples():
            folium.CircleMarker([r.latitude, r.longitude], radius=5, color="black", fill=True, tooltip=f"{r.ts:%H:%M} {r.accident_severity}").add_to(fc)
        fc.add_to(m); folium.LayerControl().add_to(m)
        st_folium(m, height=520, use_container_width=True, returned_objects=[])
        n_sig = int(hs.gi_hot.sum())
        st.caption(f"Hotspot layers use the full data period. Gi* found **{n_sig}** significant cell(s) in {L['city']} after FDR control; "
                   f"DBSCAN marked **{int(hs.db_hot.sum())}**. Alert thresholds: Medium ≥ {sub.thr_medium.iloc[0]:.2f}, High ≥ {sub.thr_high.iloc[0]:.2f} "
                   f"(risk relative to the city's average cell × conditions risk). P(crash) is for the 6-hour block containing the hour.")

with tab_tbl:
    if res is not None:
        top = res.sort_values("score", ascending=False).head(25)[
            ["hour_ts", "cell_id", "alert", "score", "rel_risk", "p_crash_block", "p_lo", "p_hi", "confidence", "cond_risk", "p_severe"]]
        st.dataframe(top.rename(columns={"p_severe": "P(severe | crash)"}), hide_index=True, width="stretch")
        sev_auc = S.m["severity_auc_val"]
        st.caption(f"P(severe | crash) has validation AUC {sev_auc:.2f} - effectively no skill in this dataset; do not rely on it.")

with tab_why:
    if res is None:
        st.info("Run a forecast first.")
    else:
        L = st.session_state["last"]
        top = res.sort_values("score", ascending=False).head(10)
        pick = st.selectbox("Cell-hour", top.index, format_func=lambda i: f"cell {top.loc[i,'cell_id']} @ {top.loc[i,'hour_ts']:%H:%M} - {top.loc[i,'alert']}")
        r = res.loc[pick]
        w = L["w"][r.hour_ts] if isinstance(L["w"], dict) else L["w"]
        ex = S.explain(L["city"], r.hour_ts, int(r.cell_id), w, L["density"], top=6)
        ex["direction"] = ex.shap.apply(lambda v: "raises risk" if v > 0 else "lowers risk")
        ex["value"] = ex["value"].astype(str)
        st.dataframe(ex[["model", "reason", "value", "direction", "shap"]], hide_index=True, width="stretch")
        st.caption("SHAP values from the crash-propensity model (where/when) and the conditions model (weather, traffic). Reasons shown are top by relative size within each model.")

with tab_eval:
    M = S.metrics; tm = M["test_metrics"]
    st.subheader("Held-out test period (time-based split)")
    df_ = pd.DataFrame(tm).T[["pr_auc", "pr_auc_lift", "roc_auc", "recall_top10pct", "f1", "brier", "brier_ref", "ece"]]
    st.dataframe(df_.round(4), width="stretch")
    st.image(str(C.ART / "eval_plots.png"))
    wc = pd.DataFrame(M["within_city_lift"]).T if "within_city_lift" in M else None
    st.subheader("Does the model beat chance inside a city?")
    if wc is not None:
        st.dataframe(wc.round(3), width="stretch")
        st.warning(f"Overall PR-AUC lift of about {tm['Balanced LightGBM ensemble (final)']['pr_auc_lift']:.2f} is explained by city size alone "
                   f"(city-only oracle lift {M['city_only_oracle_lift']:.2f}). Within a city, lift is about 1 and inside a shuffled-score null band.")
    st.subheader("Hotspot analysis")
    st.json({k: v for k, v in M["hotspots_full_period"].items()})
    st.subheader("Hotspot persistence (feature-window hotspots vs later crashes)")
    st.json(M["hotspot_persistence"])
    st.subheader("Cleaning report"); st.json(M["cleaning"])
