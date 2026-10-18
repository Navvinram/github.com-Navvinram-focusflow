"""Turn model predictions into concrete actions."""
import pandas as pd

from scheduler import build_plan, hourly_forecast

# (label, short label, feature, change, lowest allowed, highest allowed)
LEVERS = [
    ("Sleep 1 more hour", "Sleep +1 hour", "sleep_duration", +1.0, 3.0, 10.0),
    ("Cut phone use by 1 hour", "Phone -1 hour", "phone_usage", -1.0, 0.0, 10.0),
    ("Lower stress by 1 point", "Stress -1", "stress", -1, 1, 5),
    ("Raise energy by 1 point", "Energy +1", "energy", +1, 1, 5),
    ("Raise motivation by 1 point", "Motivation +1", "motivation", +1, 1, 5),
]


def fmt_hour(h):
    return f"{int(h) % 12 or 12} {'AM' if h < 12 else 'PM'}"


def best_and_risk_hours(forecast, n_best=3, n_risk=2):
    best = sorted(build_plan(forecast).head(n_best)["hour"].tolist())
    risk = sorted(forecast.nlargest(n_risk, "distraction")["hour"].tolist())
    return best, risk


def what_if(state, day_of_week, reg, clf):
    """Re-run the forecast with one input improved at a time and report the change.

    Pass the Linear and Logistic Regression models here: their effects are easy to
    interpret and stable, while tree models give noisy one-input-at-a-time estimates.
    """
    base = hourly_forecast(state, day_of_week, reg, clf)
    rows = []
    for label, short, key, delta, lo, hi in LEVERS:
        changed = dict(state)
        changed[key] = min(hi, max(lo, state[key] + delta))
        if changed[key] == state[key]:
            continue  # already at the limit
        f = hourly_forecast(changed, day_of_week, reg, clf)
        rows.append({
            "Action": label,
            "Short": short,
            "Productivity gain (points)": f["productivity"].mean() - base["productivity"].mean(),
            "Distraction risk change (points)": 100 * (f["distraction"].mean() - base["distraction"].mean()),
        })
    return pd.DataFrame(rows).sort_values("Productivity gain (points)", ascending=False).reset_index(drop=True)


def build_tips(state, forecast):
    """Rule-based advice. Returns (improve_productivity, avoid_distractions)."""
    best, risk = best_and_risk_hours(forecast)
    best_txt = ", ".join(fmt_hour(h) for h in best)
    risk_txt = " and ".join(fmt_hour(h) for h in risk)

    improve = [f"Do your hardest, highest-priority task in your best hours ({best_txt})."]
    if state["sleep_duration"] < 7:
        improve.append(f"You slept {state['sleep_duration']:g} hours. Aim for 7-8 hours tonight, and keep a fixed bedtime.")
    if state["energy"] <= 2:
        improve.append("Energy is low: take a 10-minute walk, drink water and have a proper meal before deep work.")
    if state["motivation"] <= 2:
        improve.append("Motivation is low: pick one small task and work on it for just 10 minutes to get started.")
    if state["stress"] >= 4:
        improve.append("Stress is high: spend 5 minutes on slow breathing, then split big tasks into smaller steps.")
    improve.append(f"Use the low-focus hours ({risk_txt}) for emails, admin and light revision.")
    improve.append("Work in 50-minute blocks with a 10-minute break between them.")

    avoid = [f"Distraction risk peaks around {risk_txt}. Plan a break or a light task then."]
    if state["phone_usage"] >= 3:
        avoid.append(f"Phone use is {state['phone_usage']:g} hours. Keep the phone in another room or on Do Not Disturb during focus blocks.")
    else:
        avoid.append("Keep the phone face down and silent during focus blocks.")
    avoid.append("Close social media tabs and chat apps before you start, and check them only at break times.")
    avoid.append("Write down distracting thoughts on paper and deal with them after the block.")
    avoid.append("Set one clear goal for each block, for example 'finish questions 1 to 5'.")
    return improve, avoid
