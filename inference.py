"""
inference.py
============
Live prediction for a single upcoming match, using the persisted Elo/history
state. The feature vector is built by feature_core.compute_features() — the
SAME function the training pipeline uses — so the model sees exactly what it
was trained on. No silent median-fill collapse.

Typical backend usage:

    predictor = Predictor("inference_state.pkl",
                          model_path="xgb_model.pkl",
                          meta_path="xgb_model_meta.json",
                          medians_path="feature_medians_xgb.json")

    result = predictor.predict(
        p1={"id": 104745, "rank": 2,  "rank_points": 8000, "age": 37.1, "ht": 185},
        p2={"id": 206173, "rank": 14, "rank_points": 2800, "age": 22.4, "ht": 188},
        match={"surface": "Hard", "tourney_level": "M", "round": "QF",
               "best_of": 3, "draw_size": 56, "tourney_id": "2025-1536",
               "tourney_date": 20250320},
    )
    # result["p1_win_prob"], result["features_filled_from_median"], ...

After a match finishes, keep state fresh:

    predictor.update_with_result(winner_id=104745, loser_id=206173, match={...})
    predictor.save("inference_state.pkl")
"""

import json
import pickle
import numpy as np
import pandas as pd

from feature_core import EloTracker, PlayerHistory, compute_features


class Predictor:
    def __init__(self, state_path, model_path=None, meta_path=None,
                 medians_path=None):
        with open(state_path, "rb") as f:
            state = pickle.load(f)

        self.elo     = EloTracker.from_state(state["elo"])
        self.history = PlayerHistory.from_state(state["history"])
        self.players = state["players"]
        self.meta    = state["meta"]

        self.surface_cols  = self.meta["surface_cols"]
        self.level_cols    = self.meta["level_cols"]
        self.encodings     = self.meta["encodings"]

        # Model + the exact feature order it expects
        self.model = None
        self.feature_names = None
        if model_path:
            with open(model_path, "rb") as f:
                self.model = pickle.load(f)
        if meta_path:
            with open(meta_path, "r", encoding="utf-8") as f:
                m = json.load(f)
            self.feature_names = m["feature_names"]

        # Training medians — used to fill the few features that are genuinely
        # unknown for a new tournament (e.g. unseen tourney_id_enc).
        self.medians = None
        if medians_path:
            self.medians = pd.read_json(medians_path, typ="series")

    # ── context builder ───────────────────────────────────────────────────────

    def _build_ctx(self, match):
        surface = match.get("surface", "Unknown")
        level   = match.get("tourney_level", "A")

        surface_onehot = {c: int(c == f"surface_{surface}")
                          for c in self.surface_cols}
        # unknown surface that wasn't in training maps to surface_Unknown
        if f"surface_{surface}" not in self.surface_cols and \
                "surface_Unknown" in surface_onehot:
            surface_onehot["surface_Unknown"] = 1

        level_onehot = {c: int(c == f"tourney_level_{level}")
                        for c in self.level_cols}

        round_str = str(match.get("round", "")).strip().upper()
        round_map = self.encodings.get("round", {})
        round_enc = round_map.get(round_str, round_map.get("Unknown", np.nan))

        tid_map = self.encodings.get("tourney_id", {})
        tid_enc = tid_map.get(str(match.get("tourney_id", "")), np.nan)

        # tourney_name_enc is dropped during feature selection, so its value
        # never reaches the model — pass NaN as a placeholder for parity.
        ctx = {
            "surface_onehot": surface_onehot,
            "level_onehot":   level_onehot,
            "round_str":      round_str,
            "round_enc":      round_enc,
            "tourney_name_enc": np.nan,
            "tourney_id_enc": tid_enc,
            "draw_size":      match.get("draw_size", np.nan),
            "best_of":        match.get("best_of", 3),
            "tourney_date":   match.get("tourney_date", self.meta["last_date"]),
        }
        return ctx

    def _player_raw(self, p):
        """Merge caller-supplied stats over the stored profile. Caller values
        win (they're current); profile fills slow-changing fields like height."""
        prof = self.players.get(int(p["id"]), {})
        def pick(k, default=np.nan):
            return p[k] if k in p and p[k] is not None else prof.get(k, default)
        return {
            "id":          int(p["id"]),
            "seed":        p.get("seed", np.nan),
            "hand":        pick("hand", "Unknown"),
            "ht":          pick("ht"),
            "ioc":         pick("ioc", ""),
            "age":         pick("age"),
            "rank":        pick("rank"),
            "rank_points": pick("rank_points"),
        }

    # ── feature row ───────────────────────────────────────────────────────────

    def build_feature_row(self, p1, p2, match):
        """Returns the full feature dict (p1 in slot 1). compute_features is the
        same function training uses, so this is guaranteed consistent."""
        ctx = self._build_ctx(match)
        p1_raw = self._player_raw(p1)
        p2_raw = self._player_raw(p2)
        surface = match.get("surface", "Unknown")
        if f"surface_{surface}" not in self.surface_cols:
            surface = "Unknown"
        tourney_date = ctx["tourney_date"]
        return compute_features(self.elo, self.history,
                                int(p1["id"]), int(p2["id"]),
                                surface, tourney_date, p1_raw, p2_raw, ctx)

    # ── predict ───────────────────────────────────────────────────────────────

    def predict(self, p1, p2, match):
        feat = self.build_feature_row(p1, p2, match)

        if self.feature_names is None:
            return {"features": feat,
                    "note": "no model/meta loaded — feature vector only"}

        row = pd.DataFrame([feat]).reindex(columns=self.feature_names)

        missing_before = int(row.isna().sum().sum())
        filled_cols = row.columns[row.isna().any()].tolist()
        if self.medians is not None:
            for c in self.feature_names:
                if c in self.medians.index:
                    row[c] = row[c].fillna(self.medians[c])
        still_na = int(row.isna().sum().sum())
        if still_na:
            row = row.fillna(0.0)

        result = {
            "features_total": len(self.feature_names),
            "features_filled_from_median": missing_before,
            "filled_columns": filled_cols,
        }
        if self.model is not None:
            prob = float(self.model.predict_proba(row[self.feature_names])[:, 1][0])
            result["p1_win_prob"] = prob
            result["p2_win_prob"] = 1.0 - prob
        return result

    # ── keep state current ────────────────────────────────────────────────────

    def update_with_result(self, winner_id, loser_id, match):
        """Apply a finished match to the state so Elo/form stay current."""
        surface = match.get("surface", "Unknown")
        if f"surface_{surface}" not in self.surface_cols:
            surface = "Unknown"
        level = match.get("tourney_level", "A")
        tourney_date = match.get("tourney_date", self.meta["last_date"])
        wrp = match.get("winner_rank_points", np.nan)
        lrp = match.get("loser_rank_points",  np.nan)

        self.history.record_match(int(winner_id), int(loser_id), surface,
                                  tourney_date, wrp, lrp)
        try:
            dp = pd.to_datetime(str(int(tourney_date)), format="%Y%m%d")
        except Exception:
            dp = None
        self.elo.update(int(winner_id), int(loser_id), surface, level, dp)
        if int(tourney_date) > self.meta["last_date"]:
            self.meta["last_date"] = int(tourney_date)

    def save(self, path):
        state = {
            "elo":     self.elo.to_state(),
            "history": self.history.to_state(),
            "players": self.players,
            "meta":    self.meta,
        }
        with open(path, "wb") as f:
            pickle.dump(state, f)

    # ── helpers ───────────────────────────────────────────────────────────────

    def resolve_player(self, name):
        """Find a player id by (sub)name match — convenience for the admin panel."""
        name_l = name.lower()
        hits = [(pid, pr) for pid, pr in self.players.items()
                if name_l in str(pr.get("name", "")).lower()]
        return hits
