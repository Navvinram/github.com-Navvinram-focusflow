import pandas as pd

from train_models import FEATURES


def hourly_forecast(state, day_of_week, reg, clf, hours=range(8, 21)):
    """Score the same user state for every hour of the day."""
    rows = [{**state, "hour": h, "day_of_week": day_of_week} for h in hours]
    X = pd.DataFrame(rows)[FEATURES]
    return pd.DataFrame({
        "hour": X["hour"],
        "productivity": reg.predict(X),
        "distraction": clf.predict_proba(X)[:, 1],
    })


def build_plan(forecast):
    """Rank hours by focus score = productivity x (1 - distraction risk)."""
    plan = forecast.copy()
    plan["focus_score"] = plan["productivity"] * (1 - plan["distraction"])
    return plan.sort_values("focus_score", ascending=False).reset_index(drop=True)


def assign_tasks(tasks, forecast):
    """Greedy: hardest, highest-priority tasks get the best free hours."""
    free = build_plan(forecast)["hour"].tolist()
    plan = []
    for t in sorted(tasks, key=lambda t: (-t["priority"], -t["difficulty"])):
        slots, free = free[:t["hours"]], free[t["hours"]:]
        plan.append((t["name"], sorted(slots)))
    return plan
