import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import List

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ---------------------------------------------------------------------------
# Database helpers
# ---------------------------------------------------------------------------

DB_PATH = Path("click_logs.db")


def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS scored_clicks (
            id                     INTEGER PRIMARY KEY AUTOINCREMENT,
            ip                     INTEGER,
            app                    INTEGER,
            device                 INTEGER,
            os                     INTEGER,
            channel                INTEGER,
            click_time             TEXT,
            conversion_probability REAL,
            risk_score             REAL,
            decision               TEXT
        )
    """)
    conn.commit()
    conn.close()


init_db()


def save_click(click, conversion_probability: float, risk_score: float, decision: str):
    """Insert a single scored click into the database."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        """
        INSERT INTO scored_clicks (
            ip, app, device, os, channel, click_time,
            conversion_probability, risk_score, decision
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """,
        (
            click.ip,
            click.app,
            click.device,
            click.os,
            click.channel,
            click.click_time,
            conversion_probability,
            risk_score,
            decision,
        ),
    )
    conn.commit()
    conn.close()


def save_clicks_batch(data_list: list):
    """
    Insert multiple scored clicks in a single DB round-trip via executemany.

    Each element of data_list must be a tuple:
        (ip, app, device, os, channel, click_time,
         conversion_probability, risk_score, decision)
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.executemany(
        """
        INSERT INTO scored_clicks (
            ip, app, device, os, channel, click_time,
            conversion_probability, risk_score, decision
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """,
        data_list,
    )
    conn.commit()
    conn.close()


# ---------------------------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------------------------

app = FastAPI(title="Click Fraud Detection API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Model & metadata loading (done once at startup)
# ---------------------------------------------------------------------------

MODEL_PATH = "models/fraud_model.pkl"
ISO_FOREST_PATH = "models/isolation_forest.pkl"
SCALER_PATH = "models/anomaly_scaler.pkl"
METADATA_PATH = "models/metadata.json"

model = joblib.load(MODEL_PATH)
iso_forest = joblib.load(ISO_FOREST_PATH)
scaler = joblib.load(SCALER_PATH)

with open(METADATA_PATH, "r", encoding="utf-8") as f:
    metadata = json.load(f)

model_features = metadata["model_features"]
anomaly_features = metadata["anomaly_features"]

# Pre-sort feature importances once so every request is O(1)
_fi_sorted: list[dict] = [
    {"feature": feat, "importance": round(float(imp), 6)}
    for feat, imp in sorted(
        zip(model_features, model.feature_importances_),
        key=lambda x: x[1],
        reverse=True,
    )
]
TOP_N_FEATURES = 5


# ---------------------------------------------------------------------------
# Input schema
# ---------------------------------------------------------------------------


class ClickInput(BaseModel):
    ip: int
    app: int
    device: int
    os: int
    channel: int
    click_time: str


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------


def _base_row(click: ClickInput) -> dict:
    """
    Build the raw feature dict for one click.
    Aggregate/historical counts default to 1 (single real-time click, no window).
    """
    click_dt = datetime.fromisoformat(click.click_time)
    return {
        "ip": click.ip,
        "app": click.app,
        "device": click.device,
        "os": click.os,
        "channel": click.channel,
        # Temporal features
        "hour": click_dt.hour,
        "day": click_dt.day,
        "minute": click_dt.minute,
        "dayofweek": click_dt.weekday(),
        # Aggregate count defaults
        "ip_click_count": 1,
        "app_click_count": 1,
        "device_click_count": 1,
        "os_click_count": 1,
        "channel_click_count": 1,
        "ip_hour_click_count": 1,
        "ip_app_click_count": 1,
        "ip_device_click_count": 1,
        "ip_channel_click_count": 1,
        "ip_unique_apps": 1,
        "ip_unique_devices": 1,
        "ip_unique_os": 1,
        "ip_unique_channels": 1,
        "time_since_prev_click_sec": -1,
        "very_fast_repeat_click": 0,
    }


def _compute_anomaly(row: dict) -> tuple[float, int]:
    """
    Score one click with the Isolation Forest.

    Steps
    -----
    1. Extract the anomaly feature subset and scale it.
    2. decision_function() → raw score  (more negative = more anomalous).
    3. Normalize:  anomaly_score = clip(0.5 − raw_score, 0, 1)
       •  raw ≈  0   → score ≈ 0.5  (borderline)
       •  raw ≪  0   → score → 1.0  (highly anomalous)
       •  raw ≫  0   → score → 0.0  (clearly normal)
    4. anomaly_flag = 1 if predict() == −1 (outlier), else 0.

    Returns
    -------
    anomaly_score : float ∈ [0, 1]
    anomaly_flag  : int   ∈ {0, 1}
    """
    X_anom = pd.DataFrame([{k: row[k] for k in anomaly_features}])
    X_scaled = scaler.transform(X_anom)

    raw_score = float(iso_forest.decision_function(X_scaled)[0])
    anomaly_score = float(np.clip(0.5 - raw_score, 0.0, 1.0))

    prediction = iso_forest.predict(X_scaled)[0]  # -1 = outlier, 1 = inlier
    anomaly_flag = 1 if prediction == -1 else 0

    return anomaly_score, anomaly_flag


def build_features(click: ClickInput) -> tuple[pd.DataFrame, float, int]:
    """
    Assemble the full feature DataFrame ready for the XGBoost model.

    Returns
    -------
    X             : pd.DataFrame  — shape (1, len(model_features))
    anomaly_score : float ∈ [0, 1]
    anomaly_flag  : int   ∈ {0, 1}
    """
    row = _base_row(click)
    anomaly_score, anomaly_flag = _compute_anomaly(row)
    row["anomaly_score"] = anomaly_score
    row["anomaly_flag"] = anomaly_flag
    df: pd.DataFrame = pd.DataFrame([row])
    X: pd.DataFrame = df.reindex(columns=model_features)
    return X, anomaly_score, anomaly_flag


# ---------------------------------------------------------------------------
# Scoring helpers
# ---------------------------------------------------------------------------


def decision_from_score(score: float) -> str:
    """
    Map a risk score to a routing decision using the metadata thresholds:
      block   ≥ 0.80
      monitor ≥ 0.50
      allow   < 0.50
    """
    if score >= 0.80:
        return "block"
    elif score >= 0.50:
        return "monitor"
    return "allow"


def _score_single(click: ClickInput) -> dict:
    """
    Core per-click scoring logic shared by /score-click and /score-batch.
    Does NOT write to the database — persistence is the caller's responsibility.
    """
    X, anomaly_score, anomaly_flag = build_features(click)

    conversion_probability = float(model.predict_proba(X)[0, 1])
    low_quality_score = 1.0 - conversion_probability

    # Weighted risk formula from metadata.json:
    #   risk = 0.70 * low_quality_score + 0.30 * anomaly_score
    risk_score = float(
        np.clip(
            0.70 * low_quality_score + 0.30 * anomaly_score,
            0.0,
            1.0,
        )
    )

    decision = decision_from_score(risk_score)

    return {
        "conversion_probability": round(conversion_probability, 6),
        "risk_score": round(risk_score, 6),
        "anomaly_score": round(anomaly_score, 6),
        "anomaly_flag": anomaly_flag,
        "decision": decision,
        "feature_importance": _fi_sorted[:TOP_N_FEATURES],
    }


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.get("/")
def home():
    return {"message": "Click Fraud Detection API is running"}


# ── Single click ────────────────────────────────────────────────────────────


@app.post("/score-click")
def score_click(click: ClickInput):
    """Score a single click, persist it, and return the full result."""
    result = _score_single(click)
    save_click(
        click,
        result["conversion_probability"],
        result["risk_score"],
        result["decision"],
    )
    return result


# ── Batch scoring ───────────────────────────────────────────────────────────


@app.post("/score-batch")
def score_batch(clicks: List[ClickInput]):
    """
    Score a list of clicks.
    All rows are persisted to the DB in a single batch insert for efficiency.
    """
    results = []
    batch_rows = []

    for click in clicks:
        result = _score_single(click)
        results.append(result)
        batch_rows.append(
            (
                click.ip,
                click.app,
                click.device,
                click.os,
                click.channel,
                click.click_time,
                result["conversion_probability"],
                result["risk_score"],
                result["decision"],
            )
        )

    if batch_rows:
        save_clicks_batch(batch_rows)

    return {"results": results, "total": len(results)}


# ── History ─────────────────────────────────────────────────────────────────


@app.get("/clicks")
def get_clicks(limit: int = Query(default=500, ge=1, le=10_000)):
    """
    Return the *limit* most recent scored clicks, newest first.
    The `id` field is returned as a string to be JSON-safe for large integers.
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT id, ip, app, device, os, channel, click_time,
               conversion_probability, risk_score, decision
        FROM   scored_clicks
        ORDER  BY id DESC
        LIMIT  ?
    """,
        (limit,),
    )
    rows = cursor.fetchall()
    columns = [
        "id",
        "ip",
        "app",
        "device",
        "os",
        "channel",
        "click_time",
        "conversion_probability",
        "risk_score",
        "decision",
    ]
    conn.close()

    return [{**dict(zip(columns, row)), "id": str(row[0])} for row in rows]


# ── Aggregate statistics ─────────────────────────────────────────────────────


@app.get("/stats")
def get_stats():
    """Return aggregate statistics computed from the scored_clicks table."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT
            COUNT(*)                                                 AS total,
            SUM(CASE WHEN decision = 'allow'   THEN 1 ELSE 0 END)  AS allow_count,
            SUM(CASE WHEN decision = 'monitor' THEN 1 ELSE 0 END)  AS monitor_count,
            SUM(CASE WHEN decision = 'block'   THEN 1 ELSE 0 END)  AS block_count,
            AVG(risk_score)                                          AS avg_risk_score,
            AVG(conversion_probability)                              AS avg_conversion_probability
        FROM scored_clicks
    """)
    row = cursor.fetchone()
    conn.close()

    total = row[0] or 0
    allow_count = row[1] or 0
    monitor_count = row[2] or 0
    block_count = row[3] or 0
    fraud_rate = round(block_count / total, 6) if total > 0 else 0.0

    return {
        "total": total,
        "allow_count": allow_count,
        "monitor_count": monitor_count,
        "block_count": block_count,
        "fraud_rate": fraud_rate,
        "avg_risk_score": round(row[4] or 0.0, 6),
        "avg_conversion_probability": round(row[5] or 0.0, 6),
    }
