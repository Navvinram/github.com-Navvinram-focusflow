import sqlite3
SCHEMA = """
CREATE TABLE IF NOT EXISTS users(id INTEGER PRIMARY KEY, name TEXT, occupation TEXT, wake_up_time TEXT, sleep_time TEXT);
CREATE TABLE IF NOT EXISTS daily_checkins(id INTEGER PRIMARY KEY, user_id INT, date TEXT, sleep_duration REAL,
  phone_usage REAL, stress INT, energy INT, motivation INT);
CREATE TABLE IF NOT EXISTS tasks(id INTEGER PRIMARY KEY, user_id INT, task_name TEXT, priority INT, difficulty INT, duration REAL, status TEXT);
CREATE TABLE IF NOT EXISTS task_feedback(id INTEGER PRIMARY KEY, task_id INT, focus_score REAL, completed INT, feedback_date TEXT);
"""
def init(path="focusflow.db"):
    con = sqlite3.connect(path); con.executescript(SCHEMA); return con

def save_checkin(con, user_id, date, **v):
    con.execute("INSERT INTO daily_checkins(user_id,date,sleep_duration,phone_usage,stress,energy,motivation) VALUES(?,?,?,?,?,?,?)",
                (user_id, date, v["sleep_duration"], v["phone_usage"], v["stress"], v["energy"], v["motivation"]))
    con.commit()
