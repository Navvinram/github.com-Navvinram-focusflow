"""Render focusflow_explainer.mp4: a short animated explainer built from the app's own models.

Needs: pip install matplotlib   and   ffmpeg on the PATH.
"""
import os

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter, FuncAnimation

from recommendations import best_and_risk_hours, fmt_hour, what_if
from scheduler import hourly_forecast
from train_models import (BUNDLE_VERSION, CLASSIFIER_NAMES, REGRESSOR_NAMES,
                          best_classifier, best_regressor, train_and_save_models)

OUT = "focusflow_explainer.mp4"
FPS = 15
BG, FG, MUTED = "#1E1B3A", "#FFFFFF", "#B8B5D6"
TEAL, AMBER, RED, PURPLE = "#2DD4BF", "#F59E0B", "#EF6461", "#8B7CF6"
SCENES = [("title", 3), ("checkin", 5), ("forecast", 9), ("actions", 8), ("outro", 4)]

if os.path.exists("focusflow_models.joblib"):
    models = joblib.load("focusflow_models.joblib")
    if models.get("version") != BUNDLE_VERSION:
        models = train_and_save_models()
else:
    models = train_and_save_models()

STATE = dict(sleep_duration=5.5, phone_usage=4.2, stress=4, energy=2, motivation=2)
DOW = 2
best_reg, best_clf = best_regressor(models), best_classifier(models)
reg = models["regressors"][REGRESSOR_NAMES[best_reg]]
clf = models["classifiers"][CLASSIFIER_NAMES[best_clf]]
forecast = hourly_forecast(STATE, DOW, reg, clf)
best_hours, risk_hours = best_and_risk_hours(forecast)
gains = what_if(STATE, DOW, models["regressors"]["linear"], models["classifiers"]["logistic"])
top_action = gains.iloc[0]["Action"]

fig = plt.figure(figsize=(12.8, 7.2), dpi=100, facecolor=BG)


def ease(p):
    p = min(1.0, max(0.0, p))
    return 1 - (1 - p) ** 3


def header(text, sub=None):
    fig.text(0.06, 0.90, text, color=FG, fontsize=34, fontweight="bold", va="center")
    if sub:
        fig.text(0.06, 0.835, sub, color=MUTED, fontsize=18, va="center")


def styled_axes(rect):
    ax = fig.add_axes(rect, facecolor=BG)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=14)
    return ax


def scene_title(p):
    a = ease(p * 2.5)
    fig.text(0.5, 0.58, "FocusFlow", color=FG, fontsize=80, fontweight="bold", ha="center", alpha=a)
    fig.text(0.5, 0.44, "Your AI productivity coach", color=AMBER, fontsize=32, ha="center", alpha=a)
    fig.text(0.5, 0.32, "Predict your best hours. Plan your day. Beat distractions.",
             color=MUTED, fontsize=20, ha="center", alpha=ease(p * 2.5 - 0.6))


def scene_checkin(p):
    header("1. Tell it about your day", "A quick daily check-in is all the model needs")
    items = [("Sleep", STATE["sleep_duration"] / 10, f"{STATE['sleep_duration']:g} h", AMBER),
             ("Phone use", STATE["phone_usage"] / 10, f"{STATE['phone_usage']:g} h", RED),
             ("Stress", STATE["stress"] / 5, f"{STATE['stress']} / 5", RED),
             ("Energy", STATE["energy"] / 5, f"{STATE['energy']} / 5", AMBER),
             ("Motivation", STATE["motivation"] / 5, f"{STATE['motivation']} / 5", AMBER)]
    for i, (name, frac, label, color) in enumerate(items):
        y = 0.68 - i * 0.115
        g = ease(p * 1.6 - i * 0.12)
        fig.text(0.08, y, name, color=FG, fontsize=22, va="center")
        fig.patches.append(plt.Rectangle((0.28, y - 0.03), 0.55, 0.06, transform=fig.transFigure,
                                         facecolor="#2F2B55", edgecolor="none"))
        fig.patches.append(plt.Rectangle((0.28, y - 0.03), 0.55 * frac * g, 0.06, transform=fig.transFigure,
                                         facecolor=color, edgecolor="none"))
        fig.text(0.86, y, label, color=FG, fontsize=22, va="center", alpha=g)


def scene_forecast(p):
    header("2. See your hour-by-hour forecast",
           (f"{best_reg} predicts productivity and distraction risk" if best_reg == best_clf
            else f"{best_reg} predicts productivity, {best_clf} predicts distraction risk"))
    ax = styled_axes([0.08, 0.20, 0.86, 0.55])
    hours = forecast["hour"].to_numpy()
    n = max(2, int(ease(p * 1.4) * len(hours)))
    ax.plot(hours[:n], forecast["productivity"].to_numpy()[:n], color=TEAL, lw=5, label="Productivity (0-100)")
    ax.plot(hours[:n], 100 * forecast["distraction"].to_numpy()[:n], color=RED, lw=5, label="Distraction risk (%)")
    ax.set_xlim(hours.min() - 0.3, hours.max() + 0.3)
    ax.set_ylim(0, 100)
    ax.set_xticks(hours[::2])
    ax.set_xticklabels([fmt_hour(h) for h in hours[::2]])
    ax.grid(axis="y", color="#36325E", lw=1)
    if p > 0.62:
        a = ease((p - 0.62) * 4)
        for h in best_hours:
            ax.axvspan(h - 0.5, h + 0.5, color=TEAL, alpha=0.22 * a, lw=0)
        for h in risk_hours:
            ax.axvspan(h - 0.5, h + 0.5, color=RED, alpha=0.22 * a, lw=0)
    leg = ax.legend(loc="upper right", fontsize=15, facecolor=BG, edgecolor=MUTED, labelcolor=FG)
    leg.get_frame().set_alpha(0.9)
    if p > 0.62:
        fig.text(0.08, 0.10, "Best focus hours: " + ", ".join(fmt_hour(h) for h in best_hours),
                 color=TEAL, fontsize=22, fontweight="bold", alpha=ease((p - 0.62) * 4))
        fig.text(0.08, 0.04, "Distraction peaks: " + " and ".join(fmt_hour(h) for h in risk_hours),
                 color=RED, fontsize=22, fontweight="bold", alpha=ease((p - 0.62) * 4))


def scene_actions(p):
    header("3. Know what to change", "Predicted gain in productivity (points) for one small improvement")
    ax = styled_axes([0.27, 0.12, 0.66, 0.62])
    g = gains.iloc[::-1]
    vals = g["Productivity gain (points)"].to_numpy() * ease(p * 1.5)
    colors = [TEAL if i == len(g) - 1 else PURPLE for i in range(len(g))]
    ax.barh(g["Short"], vals, color=colors, height=0.62)
    ax.set_xlim(0, g["Productivity gain (points)"].max() * 1.25)
    ax.tick_params(axis="y", labelsize=20, colors=FG)
    ax.set_xlabel("Points gained", color=MUTED, fontsize=15)
    ax.grid(axis="x", color="#36325E", lw=1)
    for y, (v, full) in enumerate(zip(vals, g["Productivity gain (points)"])):
        if p > 0.5:
            ax.text(v + 0.1, y, f"+{full:.1f}", color=FG, fontsize=20, va="center", alpha=ease((p - 0.5) * 3))


def scene_outro(p):
    a = ease(p * 2)
    fig.text(0.5, 0.80, "Plan your day around your best hours", color=FG, fontsize=38,
             fontweight="bold", ha="center", alpha=a)
    lines = [(TEAL, f"Hardest task in your best hours: {', '.join(fmt_hour(h) for h in best_hours)}"),
             (RED, "Phone away and a break in your risk hours"),
             (AMBER, f"Biggest lever today: {top_action.lower()}")]
    for i, (c, t) in enumerate(lines):
        fig.text(0.12, 0.60 - i * 0.14, "●  " + t, color=c, fontsize=26, va="center",
                 alpha=ease(p * 2.2 - 0.4 * (i + 1) * 0.5))
    fig.text(0.5, 0.10, "FocusFlow  |  trained on synthetic data for demonstration", color=MUTED,
             fontsize=16, ha="center")


DRAW = {"title": scene_title, "checkin": scene_checkin, "forecast": scene_forecast,
        "actions": scene_actions, "outro": scene_outro}
bounds, t0 = [], 0
for name, dur in SCENES:
    bounds.append((name, t0, t0 + dur))
    t0 += dur
TOTAL = int(t0 * FPS)


def render(frame):
    t = frame / FPS
    for name, a, b in bounds:
        if a <= t < b or (name == "outro" and t >= a):
            fig.clf()
            fig.patch.set_facecolor(BG)
            DRAW[name]((t - a) / (b - a))
            return


anim = FuncAnimation(fig, render, frames=TOTAL, interval=1000 / FPS)
anim.save(OUT, writer=FFMpegWriter(fps=FPS, codec="libx264", bitrate=1800,
                                   extra_args=["-pix_fmt", "yuv420p", "-movflags", "+faststart"]))
print(f"wrote {OUT}: {TOTAL / FPS:.0f}s")
