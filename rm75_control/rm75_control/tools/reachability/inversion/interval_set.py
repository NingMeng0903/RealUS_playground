"""1-D closed-interval algebra for rail-base / y_shift reachability queries."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Interval:
    lo: float
    hi: float

    def __post_init__(self) -> None:
        if self.lo > self.hi:
            raise ValueError(f"invalid interval [{self.lo}, {self.hi}]")

    @property
    def length(self) -> float:
        return float(self.hi - self.lo)

    def contains(self, x: float) -> bool:
        return self.lo <= float(x) <= self.hi

    def intersect(self, other: "Interval") -> "Interval | None":
        lo = max(self.lo, other.lo)
        hi = min(self.hi, other.hi)
        if lo > hi:
            return None
        return Interval(lo, hi)


class IntervalSet:
    """Sorted, merged list of closed intervals on the real line."""

    def __init__(self, intervals: list[Interval] | None = None) -> None:
        self.intervals: list[Interval] = self._normalize(intervals or [])

    @classmethod
    def from_pairs(cls, pairs: list[tuple[float, float]]) -> "IntervalSet":
        return cls([Interval(float(a), float(b)) for a, b in pairs])

    @classmethod
    def none(cls) -> "IntervalSet":
        return cls([])

    @classmethod
    def full(cls, lo: float = -np.inf, hi: float = np.inf) -> "IntervalSet":
        return cls([Interval(float(lo), float(hi))])

    @staticmethod
    def _normalize(intervals: list[Interval]) -> list[Interval]:
        if not intervals:
            return []
        sorted_iv = sorted(intervals, key=lambda iv: iv.lo)
        merged: list[Interval] = [sorted_iv[0]]
        for iv in sorted_iv[1:]:
            last = merged[-1]
            if iv.lo <= last.hi + 1e-12:
                merged[-1] = Interval(last.lo, max(last.hi, iv.hi))
            else:
                merged.append(iv)
        return merged

    @property
    def empty(self) -> bool:
        return len(self.intervals) == 0

    def __bool__(self) -> bool:
        return not self.empty

    def total_length(self) -> float:
        return float(sum(iv.length for iv in self.intervals))

    def intersect(self, other: "IntervalSet") -> "IntervalSet":
        if self.empty or other.empty:
            return IntervalSet.none()
        out: list[Interval] = []
        j = 0
        for a in self.intervals:
            while j < len(other.intervals) and other.intervals[j].hi < a.lo:
                j += 1
            k = j
            while k < len(other.intervals) and other.intervals[k].lo <= a.hi:
                b = other.intervals[k]
                lo = max(a.lo, b.lo)
                hi = min(a.hi, b.hi)
                if lo <= hi:
                    out.append(Interval(lo, hi))
                k += 1
        return IntervalSet(out)

    def union(self, other: "IntervalSet") -> "IntervalSet":
        return IntervalSet(self.intervals + other.intervals)

    def contains_value(self, x: float) -> bool:
        x = float(x)
        for iv in self.intervals:
            if iv.contains(x):
                return True
        return False

    def sample_grid(self, step: float) -> np.ndarray:
        """Return sorted 1-D samples at ``step`` spacing inside all intervals."""
        if self.empty or step <= 0:
            return np.zeros(0, dtype=np.float64)
        pts: list[float] = []
        for iv in self.intervals:
            n = max(1, int(np.floor((iv.hi - iv.lo) / step)) + 1)
            xs = iv.lo + step * np.arange(n, dtype=np.float64)
            xs = xs[xs <= iv.hi + 1e-12]
            pts.extend(xs.tolist())
        return np.unique(np.asarray(pts, dtype=np.float64))

    def to_pairs(self) -> list[tuple[float, float]]:
        return [(iv.lo, iv.hi) for iv in self.intervals]

    def __repr__(self) -> str:
        return f"IntervalSet({self.to_pairs()})"


def run_length_true_mask(mask: np.ndarray, xs: np.ndarray) -> IntervalSet:
    """Compress a boolean mask over sorted ``xs`` into merged intervals."""
    if mask.size == 0:
        return IntervalSet.none()
    if xs.size != mask.size:
        raise ValueError("mask and xs must have same length")
    intervals: list[Interval] = []
    i = 0
    while i < mask.size:
        if not mask[i]:
            i += 1
            continue
        j = i
        while j + 1 < mask.size and mask[j + 1]:
            j += 1
        intervals.append(Interval(float(xs[i]), float(xs[j])))
        i = j + 1
    return IntervalSet(intervals)
