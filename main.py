# main.py
# ATP Match Prediction API
# Run locally: uvicorn main:app --reload
# Docs at: http://localhost:8000/docs

import json
import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional

# ── Load model + metadata ──────────────────────────────────────────────────
app = FastAPI(title="ATPPredict API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # tighten this when you go to production
    allow_methods=["*"],
    allow_headers=["*"],
)

model = joblib.load("xgb_model.pkl")

with open("xgb_model_meta.json") as f:
    meta = json.load(f)

FEATURE_NAMES = meta["feature_names"]  # 109 features in exact order

# ── Input schema ───────────────────────────────────────────────────────────
# These are the features your model was trained on.
# The app sends these; we run them through the model and return a probability.

class MatchFeatures(BaseModel):
    # Player basics
    p1_ht: float = 185.0
    p2_ht: float = 185.0
    p1_age: float = 26.0
    p2_age: float = 26.0

    # Rankings
    p1_rank: float = 50.0
    p2_rank: float = 50.0
    p1_rank_points: float = 900.0
    p2_rank_points: float = 900.0

    # Elo ratings
    p1_elo: float = 1500.0
    p2_elo: float = 1500.0
    elo_diff: float = 0.0
    p1_elo_surface: float = 1500.0
    p2_elo_surface: float = 1500.0
    elo_surface_diff: float = 0.0
    p1_elo_surf_dec: float = 1500.0
    p2_elo_surf_dec: float = 1500.0
    elo_surf_dec_diff: float = 0.0
    elo_win_prob: float = 0.5
    elo_surface_win_prob: float = 0.5
    elo_surf_dec_win_prob: float = 0.5

    # Rank diffs
    rank_diff: float = 0.0
    rank_points_diff: float = 0.0
    age_diff: float = 0.0
    height_diff: float = 0.0

    # Win rates (long window = last 30, short = last 10)
    p1_win_rate: float = 0.5
    p2_win_rate: float = 0.5
    win_rate_diff: float = 0.0
    p1_surface_wr: float = 0.5
    p2_surface_wr: float = 0.5
    surface_wr_diff: float = 0.0
    p1_win_rate_l: float = 0.5
    p2_win_rate_l: float = 0.5
    win_rate_diff_l: float = 0.0
    p1_surface_wr_l: float = 0.5
    p2_surface_wr_l: float = 0.5
    surface_wr_diff_l: float = 0.0

    # Form
    p1_form_momentum: float = 0.0
    p2_form_momentum: float = 0.0
    form_momentum_diff: float = 0.0

    # Head to head
    p1_h2h_wr: float = 0.5
    p2_h2h_wr: float = 0.5
    h2h_diff: float = 0.0
    h2h_n: float = 0.0
    p1_h2h_wr_recent: float = 0.5
    p2_h2h_wr_recent: float = 0.5
    h2h_recent_diff: float = 0.0
    h2h_best_diff: float = 0.0

    # Rest & fatigue
    p1_days_rest: float = 2.0
    p2_days_rest: float = 2.0
    days_rest_diff: float = 0.0
    p1_fatigue: float = 0.0
    p2_fatigue: float = 0.0
    fatigue_diff: float = 0.0

    # Rank trajectory
    p1_rank_traj: float = 0.0
    p2_rank_traj: float = 0.0
    rank_traj_diff: float = 0.0

    # Rank tier flags
    p1_top10: float = 0.0
    p2_top10: float = 0.0
    p1_top50: float = 0.0
    p2_top50: float = 0.0
    rank_points_ratio: float = 1.0

    # Round flags
    is_final: float = 0.0
    is_semifinal: float = 0.0
    is_qf: float = 0.0
    is_late_round: float = 0.0
    is_bo5: float = 0.0

    # Interaction features (bo5)
    elo_diff_x_bo5: float = 0.0
    fatigue_diff_x_bo5: float = 0.0
    win_rate_diff_x_bo5: float = 0.0
    age_diff_x_bo5: float = 0.0

    # Interaction features (surface)
    rank_diff_x_carpet: float = 0.0
    win_rate_diff_x_carpet: float = 0.0
    elo_diff_x_carpet: float = 0.0
    rank_diff_x_clay: float = 0.0
    win_rate_diff_x_clay: float = 0.0
    elo_diff_x_clay: float = 0.0
    rank_diff_x_grass: float = 0.0
    win_rate_diff_x_grass: float = 0.0
    elo_diff_x_grass: float = 0.0
    rank_diff_x_hard: float = 0.0
    win_rate_diff_x_hard: float = 0.0
    elo_diff_x_hard: float = 0.0
    rank_diff_x_unknown: float = 0.0
    win_rate_diff_x_unknown: float = 0.0
    elo_diff_x_unknown: float = 0.0

    # Interaction features (tournament level)
    surface_wr_diff_x_grandslam: float = 0.0
    surface_wr_diff_x_masters: float = 0.0
    win_rate_diff_x_grandslam: float = 0.0
    win_rate_diff_x_masters: float = 0.0
    elo_diff_x_grandslam: float = 0.0
    elo_diff_x_masters: float = 0.0

    # Cross features
    h2h_diff_x_surface_wr_diff: float = 0.0
    rank_traj_x_rest: float = 0.0
    elo_x_surface_wr_diff: float = 0.0

    # Surface one-hot
    surface_Carpet: float = 0.0
    surface_Clay: float = 0.0
    surface_Grass: float = 0.0
    surface_Hard: float = 1.0
    surface_Unknown: float = 0.0

    # Tournament level one-hot
    tourney_level_A: float = 0.0
    tourney_level_D: float = 0.0
    tourney_level_F: float = 0.0
    tourney_level_G: float = 0.0
    tourney_level_M: float = 0.0
    tourney_level_O: float = 0.0

    # Encoded categoricals
    round_enc: float = 1.0
    tourney_id_enc: float = 0.0
    draw_size: float = 32.0
    best_of: float = 3.0


# ── Helper: build a simple feature vector from basic player info ───────────
# This lets the app send just the simple stats and we compute the rest.

class SimpleMatchRequest(BaseModel):
    # Player 1
    p1_name: str
    p1_rank: int
    p1_rank_points: float
    p1_elo: float
    p1_elo_surface: float
    p1_age: float
    p1_ht: float
    p1_win_rate: float        # last 30 matches
    p1_win_rate_short: float  # last 10 matches
    p1_surface_wr: float
    p1_days_rest: int
    p1_fatigue: float
    p1_form_momentum: float
    p1_h2h_wr: float
    p1_h2h_wr_recent: float
    p1_rank_traj: float

    # Player 2
    p2_name: str
    p2_rank: int
    p2_rank_points: float
    p2_elo: float
    p2_elo_surface: float
    p2_age: float
    p2_ht: float
    p2_win_rate: float
    p2_win_rate_short: float
    p2_surface_wr: float
    p2_days_rest: int
    p2_fatigue: float
    p2_form_momentum: float
    p2_h2h_wr: float
    p2_h2h_wr_recent: float
    p2_rank_traj: float

    # Match context
    surface: str              # "Hard", "Clay", "Grass", "Carpet"
    tourney_level: str        # "G", "M", "A", "F", "D", "O"
    round: str                # "F", "SF", "QF", "R16", "R32", "R64", "R128", "RR"
    draw_size: int = 32
    best_of: int = 3
    h2h_n: int = 0
    h2h_best_diff: float = 0.0
    tourney_id_enc: float = 0.0


def build_features(r: SimpleMatchRequest) -> pd.DataFrame:
    """Compute all 109 features from the simple request."""

    # Diffs
    elo_diff        = r.p1_elo - r.p2_elo
    rank_diff       = r.p1_rank - r.p2_rank
    rank_pts_diff   = r.p1_rank_points - r.p2_rank_points
    age_diff        = r.p1_age - r.p2_age
    height_diff     = r.p1_ht - r.p2_ht
    elo_surf_diff   = r.p1_elo_surface - r.p2_elo_surface
    wr_diff         = r.p1_win_rate - r.p2_win_rate
    surf_wr_diff    = r.p1_surface_wr - r.p2_surface_wr
    wr_diff_l       = r.p1_win_rate_short - r.p2_win_rate_short
    h2h_diff        = r.p1_h2h_wr - r.p2_h2h_wr
    days_rest_diff  = r.p1_days_rest - r.p2_days_rest
    fatigue_diff    = r.p1_fatigue - r.p2_fatigue
    momentum_diff   = r.p1_form_momentum - r.p2_form_momentum
    rank_traj_diff  = r.p1_rank_traj - r.p2_rank_traj
    h2h_recent_diff = r.p1_h2h_wr_recent - r.p2_h2h_wr_recent

    # Elo win probabilities (logistic)
    def elo_prob(e1, e2): return 1 / (1 + 10 ** ((e2 - e1) / 400))
    elo_win_prob         = elo_prob(r.p1_elo, r.p2_elo)
    elo_surface_win_prob = elo_prob(r.p1_elo_surface, r.p2_elo_surface)
    elo_surf_dec_win_prob = elo_surface_win_prob  # simplified

    # Surface one-hot
    surf = r.surface
    surface_Carpet  = float(surf == "Carpet")
    surface_Clay    = float(surf == "Clay")
    surface_Grass   = float(surf == "Grass")
    surface_Hard    = float(surf == "Hard")
    surface_Unknown = float(surf not in ["Carpet", "Clay", "Grass", "Hard"])

    # Tournament level one-hot
    lvl = r.tourney_level
    tourney_level_A = float(lvl == "A")
    tourney_level_D = float(lvl == "D")
    tourney_level_F = float(lvl == "F")
    tourney_level_G = float(lvl == "G")
    tourney_level_M = float(lvl == "M")
    tourney_level_O = float(lvl == "O")

    # Round encoding (higher = later round)
    round_map = {"R128": 1, "R64": 2, "R32": 3, "R16": 4,
                 "QF": 5, "SF": 6, "F": 7, "RR": 3}
    round_enc   = float(round_map.get(r.round, 3))
    is_final    = float(r.round == "F")
    is_semifinal = float(r.round == "SF")
    is_qf       = float(r.round == "QF")
    is_late_round = float(r.round in ["QF", "SF", "F"])
    is_bo5      = float(r.best_of == 5)

    # Rank tier flags
    p1_top10  = float(r.p1_rank <= 10)
    p2_top10  = float(r.p2_rank <= 10)
    p1_top50  = float(r.p1_rank <= 50)
    p2_top50  = float(r.p2_rank <= 50)
    rank_pts_ratio = r.p1_rank_points / max(r.p2_rank_points, 1)

    # Interaction features
    is_grandslam = float(lvl == "G")
    is_masters   = float(lvl == "M")

    feat = {
        "p1_ht": r.p1_ht, "p2_ht": r.p2_ht,
        "p1_age": r.p1_age, "p2_age": r.p2_age,
        "p1_rank": r.p1_rank, "p2_rank": r.p2_rank,
        "p1_rank_points": r.p1_rank_points, "p2_rank_points": r.p2_rank_points,
        "p1_elo": r.p1_elo, "p2_elo": r.p2_elo,
        "elo_diff": elo_diff,
        "p1_elo_surface": r.p1_elo_surface, "p2_elo_surface": r.p2_elo_surface,
        "elo_surface_diff": elo_surf_diff,
        "p1_elo_surf_dec": r.p1_elo_surface, "p2_elo_surf_dec": r.p2_elo_surface,
        "elo_surf_dec_diff": elo_surf_diff,
        "elo_win_prob": elo_win_prob,
        "elo_surface_win_prob": elo_surface_win_prob,
        "elo_surf_dec_win_prob": elo_surf_dec_win_prob,
        "rank_diff": rank_diff, "rank_points_diff": rank_pts_diff,
        "age_diff": age_diff, "height_diff": height_diff,
        "p1_win_rate": r.p1_win_rate, "p2_win_rate": r.p2_win_rate,
        "win_rate_diff": wr_diff,
        "p1_surface_wr": r.p1_surface_wr, "p2_surface_wr": r.p2_surface_wr,
        "surface_wr_diff": surf_wr_diff,
        "p1_win_rate_l": r.p1_win_rate_short, "p2_win_rate_l": r.p2_win_rate_short,
        "win_rate_diff_l": wr_diff_l,
        "p1_surface_wr_l": r.p1_surface_wr, "p2_surface_wr_l": r.p2_surface_wr,
        "surface_wr_diff_l": surf_wr_diff,
        "p1_form_momentum": r.p1_form_momentum, "p2_form_momentum": r.p2_form_momentum,
        "form_momentum_diff": momentum_diff,
        "p1_h2h_wr": r.p1_h2h_wr, "p2_h2h_wr": r.p2_h2h_wr,
        "h2h_diff": h2h_diff, "h2h_n": float(r.h2h_n),
        "p1_h2h_wr_recent": r.p1_h2h_wr_recent, "p2_h2h_wr_recent": r.p2_h2h_wr_recent,
        "h2h_recent_diff": h2h_recent_diff, "h2h_best_diff": r.h2h_best_diff,
        "p1_days_rest": float(r.p1_days_rest), "p2_days_rest": float(r.p2_days_rest),
        "days_rest_diff": days_rest_diff,
        "p1_fatigue": r.p1_fatigue, "p2_fatigue": r.p2_fatigue,
        "fatigue_diff": fatigue_diff,
        "p1_rank_traj": r.p1_rank_traj, "p2_rank_traj": r.p2_rank_traj,
        "rank_traj_diff": rank_traj_diff,
        "p1_top10": p1_top10, "p2_top10": p2_top10,
        "p1_top50": p1_top50, "p2_top50": p2_top50,
        "rank_points_ratio": rank_pts_ratio,
        "is_final": is_final, "is_semifinal": is_semifinal,
        "is_qf": is_qf, "is_late_round": is_late_round, "is_bo5": is_bo5,
        "elo_diff_x_bo5": elo_diff * is_bo5,
        "fatigue_diff_x_bo5": fatigue_diff * is_bo5,
        "win_rate_diff_x_bo5": wr_diff * is_bo5,
        "age_diff_x_bo5": age_diff * is_bo5,
        "rank_diff_x_carpet": rank_diff * surface_Carpet,
        "win_rate_diff_x_carpet": wr_diff * surface_Carpet,
        "elo_diff_x_carpet": elo_diff * surface_Carpet,
        "rank_diff_x_clay": rank_diff * surface_Clay,
        "win_rate_diff_x_clay": wr_diff * surface_Clay,
        "elo_diff_x_clay": elo_diff * surface_Clay,
        "rank_diff_x_grass": rank_diff * surface_Grass,
        "win_rate_diff_x_grass": wr_diff * surface_Grass,
        "elo_diff_x_grass": elo_diff * surface_Grass,
        "rank_diff_x_hard": rank_diff * surface_Hard,
        "win_rate_diff_x_hard": wr_diff * surface_Hard,
        "elo_diff_x_hard": elo_diff * surface_Hard,
        "rank_diff_x_unknown": rank_diff * surface_Unknown,
        "win_rate_diff_x_unknown": wr_diff * surface_Unknown,
        "elo_diff_x_unknown": elo_diff * surface_Unknown,
        "surface_wr_diff_x_grandslam": surf_wr_diff * is_grandslam,
        "surface_wr_diff_x_masters": surf_wr_diff * is_masters,
        "win_rate_diff_x_grandslam": wr_diff * is_grandslam,
        "win_rate_diff_x_masters": wr_diff * is_masters,
        "elo_diff_x_grandslam": elo_diff * is_grandslam,
        "elo_diff_x_masters": elo_diff * is_masters,
        "h2h_diff_x_surface_wr_diff": h2h_diff * surf_wr_diff,
        "rank_traj_x_rest": rank_traj_diff * days_rest_diff,
        "elo_x_surface_wr_diff": elo_diff * surf_wr_diff,
        "surface_Carpet": surface_Carpet, "surface_Clay": surface_Clay,
        "surface_Grass": surface_Grass, "surface_Hard": surface_Hard,
        "surface_Unknown": surface_Unknown,
        "tourney_level_A": tourney_level_A, "tourney_level_D": tourney_level_D,
        "tourney_level_F": tourney_level_F, "tourney_level_G": tourney_level_G,
        "tourney_level_M": tourney_level_M, "tourney_level_O": tourney_level_O,
        "round_enc": round_enc,
        "tourney_id_enc": r.tourney_id_enc,
        "draw_size": float(r.draw_size),
        "best_of": float(r.best_of),
    }

    return pd.DataFrame([feat])[FEATURE_NAMES]


# ── Routes ─────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {
        "name": "ATPPredict API",
        "version": "1.0.0",
        "model_auc": meta["cv_auc"],
        "features": len(FEATURE_NAMES),
        "docs": "/docs",
    }


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(req: SimpleMatchRequest):
    """
    Main prediction endpoint.
    Send basic player stats → get win probability back.
    """
    try:
        df = build_features(req)
        prob = float(model.predict_proba(df)[0][1])

        # Determine signal
        p = max(prob, 1 - prob)
        if p >= 0.70:
            signal = "green"
            signal_label = "Bet"
        elif p >= 0.57:
            signal = "amber"
            signal_label = "Risky"
        else:
            signal = "red"
            signal_label = "Skip"

        # Confidence level 1-5
        if p >= 0.85: confidence = 5
        elif p >= 0.78: confidence = 4
        elif p >= 0.70: confidence = 3
        elif p >= 0.62: confidence = 2
        else: confidence = 1

        return {
            "p1_name": req.p1_name,
            "p2_name": req.p2_name,
            "p1_win_prob": round(prob, 4),
            "p2_win_prob": round(1 - prob, 4),
            "signal": signal,
            "signal_label": signal_label,
            "confidence": confidence,
            "surface": req.surface,
            "tourney_level": req.tourney_level,
            "round": req.round,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/batch")
def predict_batch(matches: list[SimpleMatchRequest]):
    """Predict multiple matches at once — used by the app to load all today's matches."""
    return [predict(m) for m in matches]


@app.get("/model/info")
def model_info():
    """Returns model metadata — useful for the app's 'about' screen."""
    return {
        "cv_auc": meta["cv_auc"],
        "best_n_trees": meta["best_n_trees"],
        "hyperparameters": meta["hyperparameters"],
        "top_features": sorted(
            meta["feature_importances"].items(),
            key=lambda x: x[1],
            reverse=True
        )[:15],
    }

    # ── ADD THIS TO THE BOTTOM OF YOUR main.py ────────────────────────────────
# Player stats scraper — fetches from tennisratio.com and ATP rankings

import re
import urllib.request
from urllib.parse import quote

@app.get("/player/{name}")
def get_player_stats(name: str):
    """
    Fetch current player stats by name.
    Usage: GET /player/CarlosAlcaraz or /player/Carlos%20Alcaraz
    Returns rank, Elo, win rates, surface stats ready for the predict endpoint.
    """
    try:
        # Format name for URL (remove spaces, keep capitals)
        clean = name.strip().replace(' ', '')
        url = f"https://www.tennisratio.com/players/{clean}.html"

        req = urllib.request.Request(url, headers={
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        with urllib.request.urlopen(req, timeout=10) as resp:
            html = resp.read().decode('utf-8', errors='ignore')

        stats = _parse_tennisratio(html, name)
        return stats

    except Exception as e:
        # Return safe defaults if scraping fails
        return {
            "name": name,
            "found": False,
            "error": str(e),
            "rank": 50,
            "rank_points": 900.0,
            "elo": 1800.0,
            "elo_surface": 1800.0,
            "age": 26.0,
            "height": 185,
            "win_rate_long": 0.60,
            "win_rate_short": 0.60,
            "surface_wr": 0.60,
            "days_rest": 3,
            "fatigue": 0.1,
        }


def _parse_tennisratio(html: str, name: str) -> dict:
    """Parse tennisratio.com player page for the stats we need."""

    def find(pattern, default=None):
        m = re.search(pattern, html, re.IGNORECASE | re.DOTALL)
        return m.group(1).strip() if m else default

    def find_float(pattern, default=0.0):
        try:
            v = find(pattern)
            return float(v.replace('%','').replace(',','')) if v else default
        except:
            return default

    # Rank
    rank = int(find_float(r'ranked world No\.\s*(\d+)', 50))

    # Win rates — look for "past 10 matches" and "past 52 weeks"
    wr_52 = find_float(r'past 52 weeks.*?(\d+\.\d+)%\s*win rate', 60.0) / 100
    wr_10_raw = find(r'past 10 matches.*?(\d+)-(\d+)\s*record')
    if wr_10_raw:
        m = re.search(r'past 10 matches.*?(\d+)-(\d+)\s*record', html, re.IGNORECASE)
        if m:
            w, l = int(m.group(1)), int(m.group(2))
            wr_10 = w / max(w + l, 1)
        else:
            wr_10 = wr_52
    else:
        wr_10 = wr_52

    # Surface win rate — try to find clay/hard/grass based on surface
    # We return all three and let the caller pick the right one
    clay_rec = re.search(r'clay.*?(\d+)-(\d+)', html, re.IGNORECASE)
    hard_rec = re.search(r'hard.*?(\d+)-(\d+)', html, re.IGNORECASE)
    grass_rec = re.search(r'grass.*?(\d+)-(\d+)', html, re.IGNORECASE)

    def wr_from_rec(m):
        if not m: return 0.60
        w, l = int(m.group(1)), int(m.group(2))
        return round(w / max(w + l, 1), 3)

    clay_wr  = wr_from_rec(clay_rec)
    hard_wr  = wr_from_rec(hard_rec)
    grass_wr = wr_from_rec(grass_rec)

    # Elo — tennisratio shows ELO score (different scale, ~10000-13000)
    # We map it back to standard ~1500-2300 scale
    elo_raw = find_float(r'ELO score of ([\d,]+)', 0)
    if elo_raw > 5000:
        # tennisratio uses a different scale, normalize to ~1500-2300
        elo = round(1500 + (elo_raw - 8000) / 20, 0)
        elo = max(1200, min(2400, elo))
    elif elo_raw > 0:
        elo = elo_raw
    else:
        # Estimate from rank
        elo = max(1200, 2300 - (rank * 4))

    # Age — look for birth year or age mentions
    age_match = re.search(r'born.*?(\d{4})', html, re.IGNORECASE)
    if age_match:
        birth_year = int(age_match.group(1))
        from datetime import datetime
        age = round(datetime.now().year - birth_year + 0.5, 1)
    else:
        age = 26.0

    # Days rest — check when last match was
    last_match = find(r'last match was (\d+) (?:day|week|month)', None)
    if last_match:
        n = int(last_match)
        unit = find(r'last match was \d+ (day|week|month)', 'day')
        if 'week' in unit: days_rest = n * 7
        elif 'month' in unit: days_rest = n * 30
        else: days_rest = n
    else:
        days_rest = 3

    # Rank points — estimate from rank if not found directly
    rank_points_map = {
        1: 11000, 2: 9000, 3: 7500, 4: 6500, 5: 5800,
        10: 3800, 20: 2200, 30: 1600, 50: 1000, 100: 500
    }
    rank_pts = 500.0
    for r_threshold in sorted(rank_points_map.keys()):
        if rank <= r_threshold:
            rank_pts = float(rank_points_map[r_threshold])
            break

    return {
        "name": name,
        "found": True,
        "rank": rank,
        "rank_points": rank_pts,
        "elo": float(elo),
        "elo_surface": float(elo),  # caller adjusts per surface
        "age": age,
        "height": 185,  # not on tennisratio, user fills manually
        "win_rate_long": round(wr_52, 3),
        "win_rate_short": round(wr_10, 3),
        "surface_wr": {
            "Clay": clay_wr,
            "Hard": hard_wr,
            "Grass": grass_wr,
            "Carpet": round((hard_wr + 0.60) / 2, 3),
        },
        "days_rest": min(days_rest, 14),
        "fatigue": round(max(0, 0.3 - (days_rest * 0.05)), 2),
        "source": "tennisratio.com",
    }