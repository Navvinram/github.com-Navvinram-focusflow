import numpy as np, pandas as pd

def generate(n_users=50, days=30, seed=42):
    rng = np.random.default_rng(seed)
    rows = []
    for u in range(n_users):
        base = rng.normal(60, 8)                      # personal baseline
        for d in range(days):
            sleep = np.clip(rng.normal(6.8, 1.2), 3, 10)
            phone = np.clip(rng.normal(4, 1.5), 0.5, 10)
            stress, energy, motiv = rng.integers(1, 6, 3)
            for hour in range(8, 21):
                circadian = 12 * np.exp(-((hour - 10) ** 2) / 8) - 10 * np.exp(-((hour - 15) ** 2) / 6)
                score = (base + circadian + 4 * (sleep - 7) - 2.5 * (phone - 4)
                         + 3 * (energy - 3) + 2 * (motiv - 3) - 2 * (stress - 3)
                         + rng.normal(0, 4))
                score = float(np.clip(score, 0, 100))
                p_dis = 1 / (1 + np.exp((score - 50) / 10))
                rows.append([u, d, hour, d % 7, sleep, phone, stress, energy, motiv,
                             score, int(rng.random() < p_dis)])
    cols = ["user_id","day","hour","day_of_week","sleep_duration","phone_usage",
            "stress","energy","motivation","productivity_score","distraction"]
    return pd.DataFrame(rows, columns=cols)
