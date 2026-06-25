# main.py
# ATP Match Prediction API  — Phase 1 (feature parity at inference)
# Run locally: uvicorn main:app --reload
# Docs:        http://localhost:8000/docs
#
# WHAT CHANGED vs the old main.py
# ───────────────────────────────
# The old /predict trusted the app/scraper to send Elo, win rates, fatigue,
# days_rest, etc. Those are exactly the features the model leans on most
# (days_rest_diff + fatigue ≈ 35% of importance), and the scraper produced
# them on the wrong scale (fatigue as a 0–0.3 float vs the model's integer
# match-count; days_rest capped at 14 vs the model's 30). That silently broke
# the model in production.
#
# Now the backend computes every stateful feature itself from the replayed
# match history (inference_state.pkl), using the SAME code that built the
# training data (feature_core.compute_features via the Predictor class). The
# app only needs to send player IDs, current rank/age/height, and match info.

import os
import json
from datetime import datetime
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from inference import Predictor

# ── App ──────────────────────────────────────────────────────────────────────
app = FastAPI(title="ATPPredict API", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # tighten when you go to production
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Load model + state ONCE at startup ───────────────────────────────────────
predictor = Predictor(
    state_path="inference_state.pkl",
    model_path="xgb_model.pkl",
    meta_path="xgb_model_meta.json",
    medians_path="feature_medians_xgb.json",   # produced by train_model_xgb.py
)

with open("xgb_model_meta.json") as f:
    meta = json.load(f)


# ── Request schemas ──────────────────────────────────────────────────────────
# Note how short this is now. No Elo / win-rate / fatigue / rest fields —
# the backend derives all of those from state. The app sends IDs + the few
# things that genuinely change match to match (rank, age) plus match context.

class PredictRequest(BaseModel):
    # players — IDs MUST match the ids in cleaned_atp_matches.csv.
    # Use /player/search to resolve a name to an id.
    p1_id: int
    p2_id: int
    p1_name: Optional[str] = None   # display only
    p2_name: Optional[str] = None

    # current stats — optional. If omitted, the player's last-known values
    # from the training history are used. Always pass current rank/points if
    # you have them; height rarely changes so the stored value is fine.
    p1_rank: Optional[float] = None
    p2_rank: Optional[float] = None
    p1_rank_points: Optional[float] = None
    p2_rank_points: Optional[float] = None
    p1_age: Optional[float] = None
    p2_age: Optional[float] = None
    p1_ht: Optional[float] = None
    p2_ht: Optional[float] = None

    # match context
    surface: str                       # "Hard","Clay","Grass","Carpet"
    tourney_level: str                 # "G","M","A","F","D","O"
    round: str                         # "F","SF","QF","R16","R32","R64","R128","RR"
    draw_size: int = 32
    best_of: int = 3
    tourney_id: Optional[str] = None   # raw tourney id, e.g. "2025-520"
    tourney_date: Optional[int] = None # yyyymmdd; defaults to today


class ResultUpdate(BaseModel):
    """Record a finished match so Elo / rest / fatigue stay current.
    days_rest_diff is the model's #1 feature, so keeping this fresh matters."""
    winner_id: int
    loser_id: int
    surface: str
    tourney_level: str = "A"
    tourney_date: Optional[int] = None
    winner_rank_points: Optional[float] = None
    loser_rank_points: Optional[float] = None


# ── helpers ──────────────────────────────────────────────────────────────────

def _today() -> int:
    return int(datetime.now().strftime("%Y%m%d"))


def _signal_and_confidence(prob: float):
    p = max(prob, 1 - prob)
    if p >= 0.70:
        signal, label = "green", "Bet"
    elif p >= 0.57:
        signal, label = "amber", "Risky"
    else:
        signal, label = "red", "Skip"
    if   p >= 0.85: confidence = 5
    elif p >= 0.78: confidence = 4
    elif p >= 0.70: confidence = 3
    elif p >= 0.62: confidence = 2
    else:           confidence = 1
    return signal, label, confidence


def _run_prediction(req: PredictRequest) -> dict:
    match = {
        "surface":      req.surface,
        "tourney_level": req.tourney_level,
        "round":        req.round,
        "draw_size":    req.draw_size,
        "best_of":      req.best_of,
        "tourney_id":   req.tourney_id or "",
        "tourney_date": req.tourney_date or _today(),
    }
    p1 = {"id": req.p1_id, "rank": req.p1_rank, "rank_points": req.p1_rank_points,
          "age": req.p1_age, "ht": req.p1_ht}
    p2 = {"id": req.p2_id, "rank": req.p2_rank, "rank_points": req.p2_rank_points,
          "age": req.p2_age, "ht": req.p2_ht}

    res  = predictor.predict(p1, p2, match)
    prob = res["p1_win_prob"]
    signal, label, confidence = _signal_and_confidence(prob)

    return {
        "p1_name": req.p1_name, "p2_name": req.p2_name,
        "p1_win_prob": round(prob, 4),
        "p2_win_prob": round(1 - prob, 4),
        "signal": signal, "signal_label": label, "confidence": confidence,
        "surface": req.surface, "tourney_level": req.tourney_level, "round": req.round,
        # monitoring: a handful (undefined H2H) is normal; a spike toward ~65
        # means an id didn't resolve or the state is stale.
        "features_filled_from_median": res["features_filled_from_median"],
    }


# ── Prediction routes ────────────────────────────────────────────────────────

@app.post("/predict")
def predict(req: PredictRequest):
    try:
        return _run_prediction(req)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/batch")
def predict_batch(matches: list[PredictRequest]):
    return [_run_prediction(m) for m in matches]


@app.post("/matches/result")
def record_result(upd: ResultUpdate):
    """Apply a finished match to the live state, then persist it.
    NOTE: Railway's disk is ephemeral across redeploys — the saved file
    survives restarts but not a new deploy. Rebuild inference_state.pkl from
    cleaned_atp_matches.csv on each deploy; in-session updates fill the gap."""
    try:
        predictor.update_with_result(
            upd.winner_id, upd.loser_id,
            {"surface": upd.surface, "tourney_level": upd.tourney_level,
             "tourney_date": upd.tourney_date or _today(),
             "winner_rank_points": upd.winner_rank_points,
             "loser_rank_points": upd.loser_rank_points},
        )
        predictor.save("inference_state.pkl")
        return {"status": "recorded"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Player lookup (replaces the name-based scraper for id resolution) ─────────

@app.get("/player/search")
def player_search(name: str):
    """Resolve a name to the player id(s) the model knows, with last-known
    stats from the training history. The admin panel uses this to pick the
    right id before sending a match to /predict."""
    hits = predictor.resolve_player(name)
    out = []
    for pid, prof in sorted(hits, key=lambda h: h[1].get("date", 0), reverse=True):
        out.append({
            "id": pid,
            "name": prof.get("name", ""),
            "last_rank": prof.get("rank"),
            "last_rank_points": prof.get("rank_points"),
            "age": prof.get("age"),
            "ht": prof.get("ht"),
            "last_seen": prof.get("date"),
        })
    return {"query": name, "matches": out[:20]}


# ── Match storage (unchanged) ────────────────────────────────────────────────

MATCHES_FILE = "matches_store.json"

def load_matches() -> list:
    if not os.path.exists(MATCHES_FILE):
        return []
    with open(MATCHES_FILE, "r") as f:
        return json.load(f)

def save_matches(matches: list):
    with open(MATCHES_FILE, "w") as f:
        json.dump(matches, f, indent=2)


@app.get("/matches")
def get_matches(tourney: str = None, signal: str = None, surface: str = None):
    matches = load_matches()
    if tourney:
        matches = [m for m in matches if tourney.lower() in m.get("tourney", "").lower()]
    if signal:
        matches = [m for m in matches if m.get("signal") == signal]
    if surface:
        matches = [m for m in matches if m.get("surface", "").lower() == surface.lower()]
    return matches


@app.post("/matches")
def save_matches_endpoint(matches: list[dict]):
    existing = load_matches()
    incoming = set(m.get("tourney", "") for m in matches)
    existing = [m for m in existing if m.get("tourney", "") not in incoming]
    updated = existing + matches
    save_matches(updated)
    return {"saved": len(matches), "total": len(updated)}


@app.delete("/matches")
def delete_tournament_matches(tourney: str):
    existing = load_matches()
    updated = [m for m in existing if m.get("tourney", "") != tourney]
    save_matches(updated)
    return {"deleted": len(existing) - len(updated), "remaining": len(updated)}


@app.get("/matches/summary")
def matches_summary():
    matches = load_matches()
    tourneys = {}
    for m in matches:
        t = m.get("tourney", "Unknown")
        tourneys.setdefault(t, {"total": 0, "green": 0, "amber": 0, "red": 0})
        tourneys[t]["total"] += 1
        tourneys[t][m.get("signal", "red")] += 1
    return {
        "total_matches": len(matches),
        "tournaments": tourneys,
        "signals": {
            "green": sum(1 for m in matches if m.get("signal") == "green"),
            "amber": sum(1 for m in matches if m.get("signal") == "amber"),
            "red":   sum(1 for m in matches if m.get("signal") == "red"),
        },
    }


# ── Info routes ──────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {
        "name": "ATPPredict API", "version": "2.0.0",
        "model_auc": meta["cv_auc"], "features": len(meta["feature_names"]),
        "state_last_date": predictor.meta["last_date"],
        "players_tracked": len(predictor.players),
        "docs": "/docs",
    }


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/model/info")
def model_info():
    return {
        "cv_auc": meta["cv_auc"],
        "best_n_trees": meta["best_n_trees"],
        "hyperparameters": meta["hyperparameters"],
        "top_features": sorted(meta["feature_importances"].items(),
                               key=lambda x: x[1], reverse=True)[:15],
    }
