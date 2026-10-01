"""Quota guard for the LLM: never exceeds N requests per rolling minute or M per day.

State lives in a small JSON file protected by an exclusive file lock, so every gunicorn worker process in
the container shares one counter. Linux only (fcntl). Replicas on other machines do NOT share it: give each
replica its own share of the quota (LLM_RPM / LLM_RPD).
"""

import fcntl
import json
import os
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

_PACIFIC = ZoneInfo("America/Los_Angeles")  # Google's daily quota resets at midnight Pacific


class QuotaLimiter:
    def __init__(self, directory: str | Path, rpm: int, rpd: int, clock: Callable[[], float] = time.time) -> None:
        self.dir = Path(directory)
        self.rpm, self.rpd, self.clock = rpm, rpd, clock

    @contextmanager
    def _locked(self) -> Iterator[dict]:
        self.dir.mkdir(parents=True, exist_ok=True)
        with open(self.dir / "quota.lock", "w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            try:
                state = self._load()
                yield state
                self._save(state)
            finally:
                fcntl.flock(lock, fcntl.LOCK_UN)

    def _load(self) -> dict:
        try:
            state = json.loads((self.dir / "quota.json").read_text())
        except (OSError, ValueError):
            state = {}
        now = self.clock()
        day = datetime.fromtimestamp(now, _PACIFIC).strftime("%Y-%m-%d")
        if state.get("day") != day:
            state = {"day": day, "count": 0, "calls": []}
        state["calls"] = [t for t in state.get("calls", []) if now - t < 60]
        return state

    def _save(self, state: dict) -> None:
        tmp = self.dir / f"quota.json.{os.getpid()}.tmp"
        tmp.write_text(json.dumps(state))
        tmp.replace(self.dir / "quota.json")  # atomic

    def try_acquire(self) -> bool:
        """Reserve one request *before* making it. Every attempt counts, successful or not, as Google counts it."""
        with self._locked() as st:
            if len(st["calls"]) >= self.rpm or st["count"] >= self.rpd:
                return False
            st["calls"].append(self.clock())
            st["count"] += 1
            return True

    def status(self) -> dict:
        with self._locked() as st:
            return {
                "remaining_this_minute": max(self.rpm - len(st["calls"]), 0),
                "remaining_today": max(self.rpd - st["count"], 0),
                "limit_per_minute": self.rpm,
                "limit_per_day": self.rpd,
            }
