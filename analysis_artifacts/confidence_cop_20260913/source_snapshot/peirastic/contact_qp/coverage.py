"""Measured path coverage, independent from the accepted reference clock."""
from __future__ import annotations

import math


class PathCoverage:
    def __init__(self, length_m, *, max_gap_s=math.inf):
        if not math.isfinite(length_m) or length_m <= 0:
            raise ValueError("positive finite path length required")
        self.length_m = float(length_m)
        if max_gap_s <= 0 or math.isnan(max_gap_s):
            raise ValueError("positive coverage time-gap bound required")
        self.max_gap_s = max_gap_s
        self.intervals = []
        self.previous = None
        self.last_time_s = -math.inf
        self.invalid_forward_travel_m = 0.
        self.maximum_path_m = 0.

    def observe(self, path_m, required_valid, time_s, *, measurement_valid=True):
        if not measurement_valid or not math.isfinite(path_m) or not math.isfinite(time_s):
            self.previous = None
            return False
        if time_s <= self.last_time_s:
            return False
        path = min(self.length_m, max(0., float(path_m)))
        old = self.previous
        if time_s-self.last_time_s > self.max_gap_s:
            old = None
        self.previous, self.last_time_s = (path, bool(required_valid)), float(time_s)
        self.maximum_path_m = max(self.maximum_path_m, path)
        if old is not None and path > old[0]:
            if old[1] and required_valid:
                self._add(old[0], path)
            else:
                self.invalid_forward_travel_m += path-old[0]
        return True

    def _add(self, lo, hi):
        merged = []
        for a, b in self.intervals:
            if b < lo:
                merged.append((a, b))
            elif a > hi:
                merged.append((lo, hi)); lo, hi = a, b
            else:
                lo, hi = min(lo, a), max(hi, b)
        merged.append((lo, hi))
        self.intervals = merged

    @property
    def valid_length_m(self):
        return sum(hi-lo for lo, hi in self.intervals)

    @property
    def missing_length_m(self):
        return self.length_m-self.valid_length_m
