#!/usr/bin/env python3
"""
search_engine.py
----------------
Two-layer vein template search. This is the ONLY module that calls
match_templates() from gabor.py. No other file calls it directly.

Layer 1 — Signature pre-filter (RAM, ~0.5ms for 3 000 templates):
    Euclidean distance on 16-float signatures. Keeps candidates below
    L1_THRESHOLD.

Layer 2 — Parallel MNHD (4 Pi 5 cores, ~2ms per template):
    match_templates() via multiprocessing.Pool, created once and reused.
"""

import numpy as np
from multiprocessing import Pool

from gabor import match_templates, MATCH_THRESHOLD
from db_manager import compute_signature, get_all_signatures, \
    get_templates_by_ids, get_username


# ---------------------------------------------------------------------------
# Module-level worker — MUST be at module level for Pool pickling
# ---------------------------------------------------------------------------

def _match_worker(args):
    """
    Module-level function required for multiprocessing.Pool pickling.
    args: (template_dict, probe_dict)
    Returns: float MNHD score
    """
    template, probe = args
    return match_templates(template, probe)


# ---------------------------------------------------------------------------
# SearchEngine
# ---------------------------------------------------------------------------

class SearchEngine:

    L1_THRESHOLD = 0.25   # Euclidean distance on 16-float signatures.
                          # 0.25 is more permissive than 0.20 — better for
                          # small databases where signatures cluster tightly.
    TOP_K        = 80     # Safety cap: max candidates passed to Layer 2.

    def __init__(self, n_workers: int = 4):
        """
        Initialise the search engine.
        Creates the multiprocessing pool (once, kept alive for session).
        Loads signature cache from DB.
        Call this after init_db().
        """
        self._pool         = Pool(processes=n_workers)
        self._sig_matrix   = None
        self._template_ids = []
        self._user_ids     = []
        self.refresh_cache()

    def refresh_cache(self):
        """
        Reload all signatures from DB into RAM.
        Call this after every enrollment so new users are searchable immediately.
        """
        data = get_all_signatures()

        if len(data['template_ids']) == 0:
            self._sig_matrix   = np.zeros((0, 16), dtype=np.float32)
            self._template_ids = []
            self._user_ids     = []
            return

        self._sig_matrix   = data['matrix']
        self._template_ids = data['template_ids']
        self._user_ids     = data['user_ids']

    def identify(self, probe_veincode: dict):
        """
        Identify a probe VeinCode against all enrolled users.

        Returns: (username: str, score: float) if accepted
                 (None, best_score: float)     if rejected

        Steps:
        1. Compute probe signature
        2. Layer 1: euclidean distance filter on RAM signature matrix
        3. Layer 2: parallel MNHD on candidates
        4. Per-user aggregation: min score across their templates
        5. Decision: if best_score <= MATCH_THRESHOLD -> accepted
        """
        if self._sig_matrix.shape[0] == 0:
            return None, 1.0

        return self._run_search(probe_veincode)

    def _run_search(self, probe_veincode: dict):
        """Execute the two-layer search and return (username|None, score)."""
        probe_sig = compute_signature(probe_veincode['VR'])
        dists     = np.linalg.norm(self._sig_matrix - probe_sig, axis=1)

        sorted_idx = np.argsort(dists)
        candidates = [i for i in sorted_idx if dists[i] < self.L1_THRESHOLD]
        candidates = candidates[:self.TOP_K]

        # Fallback: always evaluate at least the top 10 closest signatures
        # to reduce false-reject rate on small databases (< 10 users).
        if len(candidates) == 0:
            candidates = sorted_idx[:min(10, len(sorted_idx))].tolist()

        candidate_template_ids = [self._template_ids[i] for i in candidates]
        candidate_user_ids     = [self._user_ids[i]     for i in candidates]
        templates              = get_templates_by_ids(candidate_template_ids)

        args   = [(t, probe_veincode) for t in templates]
        scores = self._pool.map(_match_worker, args)

        user_best = self._aggregate_per_user(candidate_user_ids, scores)

        best_user_id = min(user_best, key=user_best.get)
        best_score   = user_best[best_user_id]

        if best_score <= MATCH_THRESHOLD:
            username = get_username(best_user_id)
            return username, best_score

        return None, best_score

    def _aggregate_per_user(self, user_ids: list, scores: list) -> dict:
        """
        Weighted aggregation per user: 0.7 * min(scores) + 0.3 * mean(scores).
        Rewards users who match consistently across multiple templates, not just
        via a single lucky minimum match.
        """
        user_scores_all: dict = {}
        for user_id, score in zip(user_ids, scores):
            user_scores_all.setdefault(user_id, []).append(score)

        user_best = {}
        for user_id, score_list in user_scores_all.items():
            min_s  = min(score_list)
            mean_s = sum(score_list) / len(score_list)
            user_best[user_id] = 0.7 * min_s + 0.3 * mean_s
        return user_best

    def shutdown(self):
        """Terminate the multiprocessing pool cleanly. Call on app exit."""
        self._pool.terminate()
        self._pool.join()
