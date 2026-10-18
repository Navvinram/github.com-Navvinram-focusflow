import datetime
import os

import altair as alt
import joblib
import pandas as pd
import streamlit as st

from recommendations import best_and_risk_hours, build_tips, fmt_hour, what_if
from scheduler import build_plan, hourly_forecast
from train_models import (
    BUNDLE_VERSION,
    CLASSIFIER_NAMES,
    REGRESSOR_NAMES,
    best_classifier,
    best_regressor,
    metrics_table,
    train_and_save_models,
)

TEAL, PURPLE = "#0F9D8F", "#8B7CF6"
MODEL_PATH = "focusflow_models.joblib"
VIDEO_PATH = "focusflow_explainer.mp4"


def model_bar(series, title, best_name, ascending):
    """Vertical bars sorted best-first, with the best model in teal."""
    df = series.rename("Value").rename_axis("Model").reset_index()
    df["Best"] = df["Model"] == best_name
    order = list(series.sort_values(ascending=ascending).index)
    chart = (
        alt.Chart(df)
        .mark_bar()
        .encode(
            x=alt.X("Model:N", sort=order, title=None, axis=alt.Axis(labelAngle=0, labelLimit=140)),
            y=alt.Y("Value:Q", title=title),
            color=alt.condition("datum.Best", alt.value(TEAL), alt.value(PURPLE)),
            tooltip=["Model", alt.Tooltip("Value:Q", format=".3f")],
        )
        .properties(height=260)
    )
    st.altair_chart(chart, width="stretch")


@st.cache_resource
def load_models():
    if os.path.exists(MODEL_PATH):
        bundle = joblib.load(MODEL_PATH)
        if isinstance(bundle, dict) and bundle.get("version") == BUNDLE_VERSION:
            return bundle
    return train_and_save_models(MODEL_PATH)


st.set_page_config(page_title="FocusFlow", page_icon="🧠", layout="wide")
st.title("🧠 FocusFlow")
st.caption("ML-powered daily planner for focus and distraction")

models = load_models()
reg_table = metrics_table(models["regression_metrics"], REGRESSOR_NAMES)
clf_table = metrics_table(models["classification_metrics"], CLASSIFIER_NAMES)
best_reg_name = best_regressor(models)
best_clf_name = best_classifier(models)

with st.sidebar:
    st.header("Your day")
    sleep = st.slider("Sleep (hours)", 3.0, 10.0, 6.5, step=0.5)
    phone = st.slider("Phone usage (hours)", 0.0, 10.0, 2.5, step=0.5)
    stress = st.slider("Stress", 1, 5, 3)
    energy = st.slider("Energy", 1, 5, 4)
    motivation = st.slider("Motivation", 1, 5, 4)

    st.header("Models")
    reg_names = list(REGRESSOR_NAMES.keys())
    clf_names = list(CLASSIFIER_NAMES.keys())
    selected_regressor_name = st.selectbox(
        "Productivity model", reg_names, index=reg_names.index(best_reg_name)
    )
    selected_classifier_name = st.selectbox(
        "Distraction model", clf_names, index=clf_names.index(best_clf_name)
    )
    st.caption("The best models are selected by default.")

state = {
    "sleep_duration": sleep,
    "phone_usage": phone,
    "stress": stress,
    "energy": energy,
    "motivation": motivation,
}
today = datetime.date.today().weekday()
reg = models["regressors"][REGRESSOR_NAMES[selected_regressor_name]]
clf = models["classifiers"][CLASSIFIER_NAMES[selected_classifier_name]]
forecast = hourly_forecast(state, today, reg, clf)
best_hours, risk_hours = best_and_risk_hours(forecast)

plan_tab, actions_tab, compare_tab, video_tab = st.tabs(
    ["📈 Today's plan", "💡 Actions to improve", "📊 Regression comparison", "🎬 Video"]
)

# ---------------------------------------------------------------- plan
with plan_tab:
    c1, c2, c3 = st.columns(3)
    c1.metric("Average productivity", f"{forecast['productivity'].mean():.0f}/100")
    c2.metric("Average distraction risk", f"{forecast['distraction'].mean():.0%}")
    c3.metric("Best focus hours", ", ".join(fmt_hour(h) for h in best_hours))
    hour_labels = [fmt_hour(h) for h in forecast["hour"]]
    curve = pd.DataFrame(
        {
            "Hour": hour_labels * 2,
            "Series": ["Productivity (0-100)"] * len(forecast) + ["Distraction risk (%)"] * len(forecast),
            "Value": list(forecast["productivity"]) + list(100 * forecast["distraction"]),
        }
    )
    st.altair_chart(
        alt.Chart(curve)
        .mark_line(point=True, strokeWidth=3)
        .encode(
            x=alt.X("Hour:N", sort=hour_labels, title=None, axis=alt.Axis(labelAngle=0)),
            y=alt.Y("Value:Q", scale=alt.Scale(domain=[0, 100]), title=None),
            color=alt.Color(
                "Series:N",
                scale=alt.Scale(domain=["Productivity (0-100)", "Distraction risk (%)"], range=[TEAL, "#EF6461"]),
                legend=alt.Legend(orient="bottom", title=None),
            ),
            tooltip=["Hour", "Series", alt.Tooltip("Value:Q", format=".0f")],
        )
        .properties(height=340),
        width="stretch",
    )
    st.subheader("Best focus hours")
    plan = build_plan(forecast)
    st.dataframe(
        plan.head(5)[["hour", "productivity", "distraction", "focus_score"]].style.format(
            {"productivity": "{:.1f}", "distraction": "{:.0%}", "focus_score": "{:.1f}"}
        ),
        width="stretch",
    )

# ------------------------------------------------------------- actions
with actions_tab:
    improve_tips, avoid_tips = build_tips(state, forecast)
    gains = what_if(
        state,
        today,
        models["regressors"]["linear"],
        models["classifiers"]["logistic"],
    )

    st.subheader("What will improve your productivity the most?")
    if gains.empty:
        st.info("All inputs are already at their best values.")
    else:
        top = gains.iloc[0]
        st.success(
            f"Biggest lever: **{top['Action'].lower()}**. "
            f"The model predicts about **+{top['Productivity gain (points)']:.1f} productivity points**."
        )
        gain_chart = gains[["Short", "Productivity gain (points)"]].rename(
            columns={"Short": "Action", "Productivity gain (points)": "Gain"}
        )
        gain_chart["Top"] = gain_chart.index == 0
        st.altair_chart(
            alt.Chart(gain_chart)
            .mark_bar()
            .encode(
                x=alt.X("Gain:Q", title="Productivity gain (points)"),
                y=alt.Y("Action:N", sort="-x", title=None, axis=alt.Axis(labelLimit=220)),
                color=alt.condition("datum.Top", alt.value(TEAL), alt.value(PURPLE)),
                tooltip=["Action", alt.Tooltip("Gain:Q", format=".1f")],
            )
            .properties(height=260),
            width="stretch",
        )
        st.dataframe(
            gains.drop(columns="Short").style.format(
                {
                    "Productivity gain (points)": "{:+.1f}",
                    "Distraction risk change (points)": "{:+.1f}",
                }
            ),
            width="stretch",
            hide_index=True,
        )

    left, right = st.columns(2)
    with left:
        st.subheader("✅ Improve productivity")
        for tip in improve_tips:
            st.markdown(f"- {tip}")
    with right:
        st.subheader("🚫 Avoid distractions")
        for tip in avoid_tips:
            st.markdown(f"- {tip}")
    st.caption(
        "Gains are estimated with the Linear and Logistic Regression models, whose coefficients are the "
        "easiest to interpret. They are trained on synthetic data, so they show how the method works "
        "and are not personal or medical advice."
    )

# ------------------------------------------------------------- compare
with compare_tab:
    st.subheader("Which regression model is best?")
    best_row = reg_table.loc[best_reg_name]
    worst_name = reg_table["RMSE"].idxmax()
    st.success(
        f"**{best_reg_name}** is the best: lowest error (MAE {best_row['MAE']:.2f}, "
        f"RMSE {best_row['RMSE']:.2f}) and highest R² ({best_row['R2']:.3f}) on the held-out test data."
    )
    st.caption(
        f"{worst_name} is the weakest here (R² {reg_table.loc[worst_name, 'R2']:.3f}). "
        "Productivity rises and falls through the day in a curve, and a straight line cannot follow a curve."
    )
    styled = (
        reg_table.style.format("{:.3f}")
        .highlight_min(subset=["MAE", "RMSE"], color="#bfe8e2")
        .highlight_max(subset=["R2"], color="#bfe8e2")
    )
    st.dataframe(styled, width="stretch")
    st.caption("Lower MAE and RMSE are better. Higher R² is better. Best values are highlighted.")

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("**RMSE (lower is better)**")
        model_bar(reg_table["RMSE"], "RMSE", reg_table["RMSE"].idxmin(), ascending=True)
    with col_b:
        st.markdown("**R² (higher is better)**")
        model_bar(reg_table["R2"], "R²", reg_table["R2"].idxmax(), ascending=False)

    st.markdown(f"**Actual vs predicted productivity: {best_reg_name}**")
    preds = models["regression_preds"][REGRESSOR_NAMES[best_reg_name]]
    scatter = pd.DataFrame({"Actual": preds["actual"], "Predicted": preds["predicted"]})
    domain = alt.Scale(domain=[0, 100])
    points = (
        alt.Chart(scatter)
        .mark_circle(opacity=0.6, size=45, color=PURPLE)
        .encode(
            x=alt.X("Actual:Q", scale=domain, title="Actual productivity"),
            y=alt.Y("Predicted:Q", scale=domain, title="Predicted productivity"),
            tooltip=[alt.Tooltip("Actual:Q", format=".1f"), alt.Tooltip("Predicted:Q", format=".1f")],
        )
    )
    perfect = (
        alt.Chart(pd.DataFrame({"v": [0, 100]}))
        .mark_line(color=TEAL, strokeDash=[6, 4], strokeWidth=3)
        .encode(x=alt.X("v:Q", title="Actual productivity"), y=alt.Y("v:Q", title="Predicted productivity"))
    )
    st.altair_chart((points + perfect).properties(height=380), width="stretch")
    st.caption("The dashed line is a perfect prediction. The closer the points sit to it, the better the model.")

    st.divider()
    st.subheader("Distraction classifiers")
    st.success(f"**{best_clf_name}** has the highest ROC-AUC ({clf_table.loc[best_clf_name, 'ROC-AUC']:.3f}).")
    st.dataframe(
        clf_table.style.format("{:.3f}").highlight_max(color="#bfe8e2"), width="stretch"
    )
    st.caption("Higher scores are better. Best value in each column is highlighted.")

# --------------------------------------------------------------- video
with video_tab:
    st.subheader("FocusFlow productivity coach")
    if os.path.exists(VIDEO_PATH):
        st.video(VIDEO_PATH)
        st.caption(
            "A short animated explainer generated from this app's own models: how check-in data "
            "becomes an hourly forecast and a list of actions."
        )
    else:
        st.info("Run `python make_video.py` to create the explainer video.")
