"""
feature_core.py
===============
Single source of truth for ATP match feature construction.

BOTH the training pipeline (feature_engineering_xgb.py) and the live
inference layer (inference.py) call compute_features() in here. Because
they share one code path, the feature vector the model sees in production
is guaranteed identical to the one it was trained on. This is the whole
point of Phase 1: no silent divergence between offline and live.

It also adds serialization (to_state / from_state) so the Elo + history
state can be persisted to disk after replaying all historical matches,
then loaded by the backend and updated as new results arrive.

The EloTracker and PlayerHistory classes below are copied VERBATIM from
the original feature_engineering_xgb.py (only serialization methods are
added). Do not edit feature logic here without re-running the parity test.
"""

import numpy as np
import pandas as pd
from collections import defaultdict, deque


# ── helpers (copied verbatim) ────────────────────────────────────────────────

def label_encode_series(s):
    """Simple label encoder for a pandas Series."""
    s = s.fillna("Unknown").astype(str)
    vals = s.unique()
    mapping = {v: i for i, v in enumerate(vals)}
    return s.map(mapping), mapping


ELO_BASE    = 1500.0
ELO_K_MAX   = 32.0
ELO_K_MIN   = 16.0
ELO_K_DECAY = 30

TOURNEY_K_MULT = {
    'G': 1.5, 'M': 1.25, 'F': 1.25, 'A': 1.0, 'D': 0.75, 'O': 0.75,
}

DAYS_REST_CAP = 30


def k_factor(games_played, tourney_level='A'):
    t    = min(games_played, ELO_K_DECAY) / ELO_K_DECAY
    base = ELO_K_MAX - t * (ELO_K_MAX - ELO_K_MIN)
    return base * TOURNEY_K_MULT.get(tourney_level, 1.0)


def elo_expected(rating_a, rating_b):
    return 1.0 / (1.0 + 10 ** ((rating_b - rating_a) / 400.0))


def safe_div(a, b):
    try:
        if np.isnan(a) or np.isnan(b) or b == 0:
            return np.nan
        return a / b
    except Exception:
        return np.nan


def _isnan(x):
    return isinstance(x, float) and np.isnan(x)


# ── Elo system ───────────────────────────────────────────────────────────────

class EloTracker:
    def __init__(self):
        self.ratings         = defaultdict(lambda: ELO_BASE)
        self.surface_ratings = defaultdict(lambda: ELO_BASE)
        self.games_played    = defaultdict(int)
        self.surface_games   = defaultdict(int)
        self.surface_history = defaultdict(list)
        self.last_update_date = {}

    def get(self, player_id):
        return self.ratings[player_id]

    def get_surface(self, player_id, surface):
        return self.surface_ratings[(player_id, surface)]

    def get_surface_decayed(self, player_id, surface, current_date,
                            half_life_days=365):
        history = self.surface_history[(player_id, surface)]
        if not history:
            return ELO_BASE
        try:
            cur = pd.to_datetime(str(int(current_date)), format='%Y%m%d')
            total_w, total_rating = 0.0, 0.0
            for date_val, rating in history:
                age_days = (cur - date_val).days
                weight   = 0.5 ** (age_days / half_life_days)
                total_w       += weight
                total_rating  += weight * rating
            return total_rating / total_w if total_w > 0 else ELO_BASE
        except Exception:
            return self.surface_ratings[(player_id, surface)]

    def update(self, winner_id, loser_id, surface, tourney_level='A', match_date=None):
        rw = self.ratings[winner_id]
        rl = self.ratings[loser_id]
        ew = elo_expected(rw, rl)
        kw = k_factor(self.games_played[winner_id], tourney_level)
        kl = k_factor(self.games_played[loser_id],  tourney_level)
        self.ratings[winner_id] += kw * (1 - ew)
        self.ratings[loser_id]  += kl * (0 - (1 - ew))
        self.games_played[winner_id] += 1
        self.games_played[loser_id]  += 1

        sw   = self.surface_ratings[(winner_id, surface)]
        sl   = self.surface_ratings[(loser_id,  surface)]
        es   = elo_expected(sw, sl)
        ks_w = k_factor(self.surface_games[(winner_id, surface)], tourney_level)
        ks_l = k_factor(self.surface_games[(loser_id,  surface)], tourney_level)
        self.surface_ratings[(winner_id, surface)] += ks_w * (1 - es)
        self.surface_ratings[(loser_id,  surface)] += ks_l * (0 - (1 - es))
        self.surface_games[(winner_id, surface)] += 1
        self.surface_games[(loser_id,  surface)] += 1
        if match_date is not None:
            self.surface_history[(winner_id, surface)].append(
                (match_date, self.surface_ratings[(winner_id, surface)]))
            self.surface_history[(loser_id, surface)].append(
                (match_date, self.surface_ratings[(loser_id, surface)]))

    # ── serialization ────────────────────────────────────────────────────────
    # Keys that are tuples (player_id, surface) are stored as-is inside the
    # plain dicts; pickle handles tuple keys fine. We only strip the lambda
    # defaults so the structures are picklable, and rebuild them on load.

    def to_state(self):
        return {
            "ratings":         dict(self.ratings),
            "surface_ratings": dict(self.surface_ratings),
            "games_played":    dict(self.games_played),
            "surface_games":   dict(self.surface_games),
            "surface_history": dict(self.surface_history),
            "last_update_date": dict(self.last_update_date),
        }

    @classmethod
    def from_state(cls, state):
        obj = cls()
        obj.ratings.update(state["ratings"])
        obj.surface_ratings.update(state["surface_ratings"])
        obj.games_played.update(state["games_played"])
        obj.surface_games.update(state["surface_games"])
        for k, v in state["surface_history"].items():
            obj.surface_history[k] = list(v)
        obj.last_update_date.update(state.get("last_update_date", {}))
        return obj


# ── Player history ───────────────────────────────────────────────────────────

class PlayerHistory:
    def __init__(self, window_short=10, window_long=30):
        self.ws = window_short
        self.wl = window_long
        self.recent_results_s  = defaultdict(lambda: deque(maxlen=self.ws))
        self.surface_results_s = defaultdict(lambda: deque(maxlen=self.ws))
        self.recent_results_l  = defaultdict(lambda: deque(maxlen=self.wl))
        self.surface_results_l = defaultdict(lambda: deque(maxlen=self.wl))
        self.h2h               = defaultdict(lambda: [0, 0])
        self.h2h_dated         = defaultdict(list)
        self.last_match_date  = {}
        self.rank_history     = defaultdict(lambda: deque(maxlen=50))
        self.match_dates      = defaultdict(lambda: deque(maxlen=20))

    def get_win_rate(self, pid, long=False):
        buf = self.recent_results_l[pid] if long else self.recent_results_s[pid]
        return sum(buf) / len(buf) if buf else np.nan

    def get_surface_wr(self, pid, surface, long=False):
        buf = (self.surface_results_l[(pid, surface)] if long
               else self.surface_results_s[(pid, surface)])
        return sum(buf) / len(buf) if buf else np.nan

    def get_h2h_wr(self, pid, opp):
        rec = self.h2h[(pid, opp)]
        return rec[0] / rec[1] if rec[1] > 0 else np.nan

    def get_h2h_n(self, pid, opp):
        return self.h2h[(pid, opp)][1] + self.h2h[(opp, pid)][1]

    def get_h2h_wr_recent(self, pid, opp, current_date, years=3):
        records = self.h2h_dated[(pid, opp)]
        if not records:
            return np.nan
        try:
            cur = pd.to_datetime(str(int(current_date)), format='%Y%m%d')
            cutoff = cur - pd.DateOffset(years=years)
            recent = [(d, w) for d, w in records if d >= cutoff]
            if not recent:
                return np.nan
            return sum(w for _, w in recent) / len(recent)
        except Exception:
            return np.nan

    def get_days_rest(self, pid, current_date):
        if pid not in self.last_match_date:
            return np.nan
        try:
            cur  = pd.to_datetime(str(int(current_date)), format='%Y%m%d')
            prev = pd.to_datetime(str(int(self.last_match_date[pid])), format='%Y%m%d')
            return (cur - prev).days
        except Exception:
            return np.nan

    def get_fatigue(self, pid, current_date, window_days=14):
        dates = self.match_dates[pid]
        if not dates:
            return 0
        try:
            cur = pd.to_datetime(str(int(current_date)), format='%Y%m%d')
            return sum(1 for d in dates if (cur - d).days <= window_days)
        except Exception:
            return 0

    def get_rank_trajectory(self, pid, current_date, current_rank_points):
        history = self.rank_history[pid]
        if not history:
            return np.nan
        try:
            cur_date = pd.to_datetime(str(int(current_date)), format='%Y%m%d')
            for past_date, past_pts in reversed(history):
                if (cur_date - past_date).days >= 60:
                    return current_rank_points - past_pts
            return np.nan
        except Exception:
            return np.nan

    def record_match(self, winner_id, loser_id, surface, date,
                     winner_rank_points, loser_rank_points):
        for buf in [self.recent_results_s[winner_id], self.recent_results_l[winner_id]]:
            buf.append(1)
        for buf in [self.recent_results_s[loser_id], self.recent_results_l[loser_id]]:
            buf.append(0)
        for buf in [self.surface_results_s[(winner_id, surface)],
                    self.surface_results_l[(winner_id, surface)]]:
            buf.append(1)
        for buf in [self.surface_results_s[(loser_id, surface)],
                    self.surface_results_l[(loser_id, surface)]]:
            buf.append(0)
        self.h2h[(winner_id, loser_id)][0] += 1
        self.h2h[(winner_id, loser_id)][1] += 1
        self.h2h[(loser_id,  winner_id)][1] += 1
        self.last_match_date[winner_id] = date
        self.last_match_date[loser_id]  = date
        try:
            dp = pd.to_datetime(str(int(date)), format='%Y%m%d')
            self.h2h_dated[(winner_id, loser_id)].append((dp, 1))
            self.h2h_dated[(loser_id,  winner_id)].append((dp, 0))
            self.match_dates[winner_id].append(dp)
            self.match_dates[loser_id].append(dp)
            if not np.isnan(winner_rank_points):
                self.rank_history[winner_id].append((dp, winner_rank_points))
            if not np.isnan(loser_rank_points):
                self.rank_history[loser_id].append((dp, loser_rank_points))
        except Exception:
            pass

    # ── serialization ────────────────────────────────────────────────────────

    def to_state(self):
        def deqmap(d):
            return {k: list(v) for k, v in d.items()}
        return {
            "ws": self.ws, "wl": self.wl,
            "recent_results_s":  deqmap(self.recent_results_s),
            "surface_results_s": deqmap(self.surface_results_s),
            "recent_results_l":  deqmap(self.recent_results_l),
            "surface_results_l": deqmap(self.surface_results_l),
            "h2h":        {k: list(v) for k, v in self.h2h.items()},
            "h2h_dated":  {k: list(v) for k, v in self.h2h_dated.items()},
            "last_match_date": dict(self.last_match_date),
            "rank_history":    deqmap(self.rank_history),
            "match_dates":     deqmap(self.match_dates),
        }

    @classmethod
    def from_state(cls, state):
        obj = cls(window_short=state["ws"], window_long=state["wl"])
        for k, v in state["recent_results_s"].items():
            obj.recent_results_s[k] = deque(v, maxlen=obj.ws)
        for k, v in state["surface_results_s"].items():
            obj.surface_results_s[k] = deque(v, maxlen=obj.ws)
        for k, v in state["recent_results_l"].items():
            obj.recent_results_l[k] = deque(v, maxlen=obj.wl)
        for k, v in state["surface_results_l"].items():
            obj.surface_results_l[k] = deque(v, maxlen=obj.wl)
        for k, v in state["h2h"].items():
            obj.h2h[k] = list(v)
        for k, v in state["h2h_dated"].items():
            obj.h2h_dated[k] = list(v)
        obj.last_match_date.update(state["last_match_date"])
        for k, v in state["rank_history"].items():
            obj.rank_history[k] = deque(v, maxlen=50)
        for k, v in state["match_dates"].items():
            obj.match_dates[k] = deque(v, maxlen=20)
        return obj


# ── THE shared feature builder ───────────────────────────────────────────────
# Extracted verbatim from feature_engineering_xgb.py's inner loop. The ONLY
# change is that p1/p2 are passed in already assigned (the random flip stays
# in the training script), and match-level context is passed via `ctx` instead
# of read from a DataFrame row. Query-by-id is equivalent to the original
# fetch-winner/loser-then-assign because Elo/history getters are keyed by id.

def compute_features(elo, history, p1_id, p2_id, surface_name, tourney_date,
                     p1_raw, p2_raw, ctx):
    """
    elo, history : state objects (read-only here)
    p1_id, p2_id : player ids already assigned to slots
    surface_name : e.g. 'Hard'
    tourney_date : int yyyymmdd
    p1_raw,p2_raw: dict of raw player fields keyed by shared field name
                   (id, seed, hand, ht, ioc, age, rank, rank_points)
    ctx          : dict with one-hot + encoded match-level fields:
                   surface_onehot {col:0/1}, level_onehot {col:0/1},
                   round_str, round_enc, tourney_name_enc, tourney_id_enc,
                   draw_size, best_of, tourney_date
    returns      : feat dict (one row of features, no target)
    """
    feat = {}

    def diff(a, b):
        return (a - b) if not (np.isnan(a) or np.isnan(b)) else np.nan

    # ── Elo reads ─────────────────────────────────────────────────────────────
    p1_elo    = elo.get(p1_id)
    p2_elo    = elo.get(p2_id)
    p1_elo_s  = elo.get_surface(p1_id, surface_name)
    p2_elo_s  = elo.get_surface(p2_id, surface_name)
    p1_elo_sd = elo.get_surface_decayed(p1_id, surface_name, tourney_date)
    p2_elo_sd = elo.get_surface_decayed(p2_id, surface_name, tourney_date)

    # form
    p1_wr_s  = history.get_win_rate(p1_id, long=False)
    p2_wr_s  = history.get_win_rate(p2_id, long=False)
    p1_wr_l  = history.get_win_rate(p1_id, long=True)
    p2_wr_l  = history.get_win_rate(p2_id, long=True)
    p1_swr_s = history.get_surface_wr(p1_id, surface_name, long=False)
    p2_swr_s = history.get_surface_wr(p2_id, surface_name, long=False)
    p1_swr_l = history.get_surface_wr(p1_id, surface_name, long=True)
    p2_swr_l = history.get_surface_wr(p2_id, surface_name, long=True)

    # h2h
    p1_h2h   = history.get_h2h_wr(p1_id, p2_id)
    p2_h2h   = history.get_h2h_wr(p2_id, p1_id)
    h2h_n    = history.get_h2h_n(p1_id, p2_id)
    p1_h2h_r = history.get_h2h_wr_recent(p1_id, p2_id, tourney_date)
    p2_h2h_r = history.get_h2h_wr_recent(p2_id, p1_id, tourney_date)

    # rest / fatigue
    p1_rest_raw = history.get_days_rest(p1_id, tourney_date)
    p2_rest_raw = history.get_days_rest(p2_id, tourney_date)
    p1_rest = min(p1_rest_raw, DAYS_REST_CAP) if not _isnan(p1_rest_raw) else np.nan
    p2_rest = min(p2_rest_raw, DAYS_REST_CAP) if not _isnan(p2_rest_raw) else np.nan
    p1_fatigue = history.get_fatigue(p1_id, tourney_date)
    p2_fatigue = history.get_fatigue(p2_id, tourney_date)

    # trajectory
    p1_rp = p1_raw.get("rank_points", np.nan)
    p2_rp = p2_raw.get("rank_points", np.nan)
    p1_rp_t = p1_rp if not _isnan(p1_rp) else 0
    p2_rp_t = p2_rp if not _isnan(p2_rp) else 0
    p1_traj = history.get_rank_trajectory(p1_id, tourney_date, p1_rp_t)
    p2_traj = history.get_rank_trajectory(p2_id, tourney_date, p2_rp_t)

    # ── raw player fields ─────────────────────────────────────────────────────
    for field in ["id", "seed", "hand", "ht", "ioc", "age", "rank", "rank_points"]:
        feat["p1_" + field] = p1_raw.get(field, np.nan)
        feat["p2_" + field] = p2_raw.get(field, np.nan)

    p1_rank     = feat.get("p1_rank",        np.nan)
    p2_rank     = feat.get("p2_rank",        np.nan)
    p1_rank_pts = feat.get("p1_rank_points", np.nan)
    p2_rank_pts = feat.get("p2_rank_points", np.nan)
    p1_age      = feat.get("p1_age",         np.nan)
    p2_age      = feat.get("p2_age",         np.nan)
    p1_ht       = feat.get("p1_ht",          np.nan)
    p2_ht       = feat.get("p2_ht",          np.nan)

    # ── Elo features ──────────────────────────────────────────────────────────
    feat['p1_elo']               = p1_elo
    feat['p2_elo']               = p2_elo
    feat['elo_diff']             = diff(p1_elo, p2_elo)
    feat['p1_elo_surface']       = p1_elo_s
    feat['p2_elo_surface']       = p2_elo_s
    feat['elo_surface_diff']     = diff(p1_elo_s, p2_elo_s)
    feat['p1_elo_surf_dec']      = p1_elo_sd
    feat['p2_elo_surf_dec']      = p2_elo_sd
    feat['elo_surf_dec_diff']    = diff(p1_elo_sd, p2_elo_sd)
    feat['elo_win_prob']         = elo_expected(p1_elo, p2_elo)
    feat['elo_surface_win_prob'] = elo_expected(p1_elo_s, p2_elo_s)
    feat['elo_surf_dec_win_prob'] = elo_expected(p1_elo_sd, p2_elo_sd)

    # ── diff features ─────────────────────────────────────────────────────────
    feat['rank_diff']        = diff(p1_rank,     p2_rank)
    feat['rank_points_diff'] = diff(p1_rank_pts, p2_rank_pts)
    feat['age_diff']         = diff(p1_age,      p2_age)
    feat['height_diff']      = diff(p1_ht,       p2_ht)

    # ── short-window form ─────────────────────────────────────────────────────
    feat['p1_win_rate']     = p1_wr_s
    feat['p2_win_rate']     = p2_wr_s
    feat['win_rate_diff']   = diff(p1_wr_s, p2_wr_s)
    feat['p1_surface_wr']   = p1_swr_s
    feat['p2_surface_wr']   = p2_swr_s
    feat['surface_wr_diff'] = diff(p1_swr_s, p2_swr_s)

    # ── long-window form ──────────────────────────────────────────────────────
    feat['p1_win_rate_l']     = p1_wr_l
    feat['p2_win_rate_l']     = p2_wr_l
    feat['win_rate_diff_l']   = diff(p1_wr_l, p2_wr_l)
    feat['p1_surface_wr_l']   = p1_swr_l
    feat['p2_surface_wr_l']   = p2_swr_l
    feat['surface_wr_diff_l'] = diff(p1_swr_l, p2_swr_l)

    feat['p1_form_momentum'] = diff(p1_wr_s, p1_wr_l)
    feat['p2_form_momentum'] = diff(p2_wr_s, p2_wr_l)
    feat['form_momentum_diff'] = diff(feat['p1_form_momentum'],
                                      feat['p2_form_momentum'])

    # ── h2h ───────────────────────────────────────────────────────────────────
    feat['p1_h2h_wr']      = p1_h2h
    feat['p2_h2h_wr']      = p2_h2h
    feat['h2h_diff']       = diff(p1_h2h, p2_h2h)
    feat['h2h_n']          = h2h_n
    feat['p1_h2h_wr_recent'] = p1_h2h_r
    feat['p2_h2h_wr_recent'] = p2_h2h_r
    feat['h2h_recent_diff']  = diff(p1_h2h_r, p2_h2h_r)
    p1_h2h_best = p1_h2h_r if not _isnan(p1_h2h_r) else p1_h2h
    p2_h2h_best = p2_h2h_r if not _isnan(p2_h2h_r) else p2_h2h
    feat['h2h_best_diff']  = diff(p1_h2h_best, p2_h2h_best)

    # ── rest & trajectory ─────────────────────────────────────────────────────
    feat['p1_days_rest']   = p1_rest
    feat['p2_days_rest']   = p2_rest
    feat['days_rest_diff'] = diff(p1_rest, p2_rest)

    feat['p1_fatigue']     = p1_fatigue
    feat['p2_fatigue']     = p2_fatigue
    feat['fatigue_diff']   = float(p1_fatigue - p2_fatigue)

    feat['p1_rank_traj']   = p1_traj
    feat['p2_rank_traj']   = p2_traj
    feat['rank_traj_diff'] = diff(p1_traj, p2_traj)

    # ── buckets ───────────────────────────────────────────────────────────────
    feat['p1_top10'] = int(not np.isnan(p1_rank) and p1_rank <= 10)
    feat['p2_top10'] = int(not np.isnan(p2_rank) and p2_rank <= 10)
    feat['p1_top50'] = int(not np.isnan(p1_rank) and p1_rank <= 50)
    feat['p2_top50'] = int(not np.isnan(p2_rank) and p2_rank <= 50)
    feat['rank_points_ratio'] = safe_div(p1_rank_pts, p2_rank_pts)

    # ── round pressure flags ──────────────────────────────────────────────────
    round_val = str(ctx.get('round_str', '')).strip().upper()
    feat['is_final']     = int(round_val == 'F')
    feat['is_semifinal'] = int(round_val in ('SF',))
    feat['is_qf']        = int(round_val in ('QF',))
    feat['is_late_round'] = int(round_val in ('F', 'SF', 'QF'))

    # ── best_of interactions ──────────────────────────────────────────────────
    best_of_val = float(ctx.get('best_of', 3) or 3)
    is_bo5 = int(best_of_val >= 5)
    feat['is_bo5'] = is_bo5
    feat['elo_diff_x_bo5']      = feat['elo_diff']      * is_bo5 \
        if not np.isnan(feat['elo_diff'])      else np.nan
    feat['fatigue_diff_x_bo5']  = feat['fatigue_diff']  * is_bo5
    feat['win_rate_diff_x_bo5'] = feat['win_rate_diff'] * is_bo5 \
        if not np.isnan(feat['win_rate_diff']) else np.nan
    feat['age_diff_x_bo5']      = feat['age_diff']      * is_bo5 \
        if not np.isnan(feat['age_diff'])      else np.nan

    # ── surface interactions ──────────────────────────────────────────────────
    surface_onehot = ctx['surface_onehot']
    for s_col, s_val in surface_onehot.items():
        s_name = s_col.replace('surface_', '').lower()
        feat[f'rank_diff_x_{s_name}']     = feat['rank_diff']     * s_val \
            if not np.isnan(feat['rank_diff'])     else np.nan
        feat[f'win_rate_diff_x_{s_name}'] = feat['win_rate_diff'] * s_val \
            if not np.isnan(feat['win_rate_diff']) else np.nan
        feat[f'elo_diff_x_{s_name}']       = feat['elo_diff']      * s_val \
            if not np.isnan(feat['elo_diff'])      else np.nan

    # ── tourney-level interactions ────────────────────────────────────────────
    level_onehot = ctx['level_onehot']
    gs_val = level_onehot.get('tourney_level_G', 0)
    m_val  = level_onehot.get('tourney_level_M', 0)
    feat['surface_wr_diff_x_grandslam']  = feat['surface_wr_diff'] * gs_val \
        if not np.isnan(feat['surface_wr_diff']) else np.nan
    feat['surface_wr_diff_x_masters']    = feat['surface_wr_diff'] * m_val \
        if not np.isnan(feat['surface_wr_diff']) else np.nan
    feat['win_rate_diff_x_grandslam']    = feat['win_rate_diff']   * gs_val \
        if not np.isnan(feat['win_rate_diff'])   else np.nan
    feat['win_rate_diff_x_masters']      = feat['win_rate_diff']   * m_val \
        if not np.isnan(feat['win_rate_diff'])   else np.nan
    feat['elo_diff_x_grandslam']         = feat['elo_diff']        * gs_val \
        if not np.isnan(feat['elo_diff'])        else np.nan
    feat['elo_diff_x_masters']           = feat['elo_diff']        * m_val \
        if not np.isnan(feat['elo_diff'])        else np.nan

    feat['h2h_diff_x_surface_wr_diff'] = feat['h2h_diff'] * \
        (feat['surface_wr_diff'] if not np.isnan(feat.get('surface_wr_diff', np.nan)) else 0)
    feat['rank_traj_x_rest'] = feat['rank_traj_diff'] * feat['days_rest_diff'] \
        if not (np.isnan(feat['rank_traj_diff']) or np.isnan(feat['days_rest_diff'])) \
        else np.nan
    feat['elo_x_surface_wr_diff'] = feat['elo_diff'] * \
        (feat['surface_wr_diff'] if not np.isnan(feat.get('surface_wr_diff', np.nan)) else 0)

    # ── match-level fields ────────────────────────────────────────────────────
    for col, val in surface_onehot.items():
        feat[col] = val
    for col, val in level_onehot.items():
        feat[col] = val
    for c in ['round_enc', 'tourney_name_enc', 'tourney_id_enc',
              'draw_size', 'best_of', 'tourney_date']:
        if c in ctx:
            feat[c] = ctx[c]

    return feat
