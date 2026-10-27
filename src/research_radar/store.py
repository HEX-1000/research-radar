"""Everything the app remembers, as small JSON files under <home>/state."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from .models import Report, SkippedSource


class Store:
    def __init__(self, state_dir: Path) -> None:
        self.dir = state_dir
        self.reports_dir = state_dir / "reports"

    # low level -------------------------------------------------------------

    def _read(self, path: Path, default: Any) -> Any:
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return default
        except ValueError:
            # A damaged file should not stop the app. Keep a copy and start clean.
            path.replace(path.with_suffix(path.suffix + ".broken"))
            return default

    def _write(self, path: Path, data: Any) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
        os.replace(tmp, path)

    # seen items --------------------------------------------------------------

    def seen(self) -> dict[str, dict[str, Any]]:
        return self._read(self.dir / "seen.json", {})

    def save_seen(self, seen: dict[str, dict[str, Any]]) -> None:
        self._write(self.dir / "seen.json", seen)

    # per source last successful scan ------------------------------------------------

    def last_runs(self) -> dict[str, str]:
        return self._read(self.dir / "last_run.json", {})

    def save_last_runs(self, runs: dict[str, str]) -> None:
        self._write(self.dir / "last_run.json", runs)

    # vectors of reported items (needed to learn from votes) --------------------------

    def vectors(self) -> dict[str, list[float]]:
        return self._read(self.dir / "vectors.json", {})

    def save_vectors(self, vectors: dict[str, list[float]]) -> None:
        self._write(self.dir / "vectors.json", vectors)

    # votes ---------------------------------------------------------------

    def votes(self) -> dict[str, int]:
        return self._read(self.dir / "feedback.json", {})

    def vote_vectors(self) -> tuple[list[list[float]], list[list[float]]]:
        votes = self.votes()
        vectors = self.vectors()
        likes = [vectors[u] for u, v in votes.items() if v > 0 and u in vectors]
        dislikes = [vectors[u] for u, v in votes.items() if v < 0 and u in vectors]
        return likes, dislikes

    def set_vote(self, date: str, url: str, vote: int) -> int:
        """Set a vote. Voting the same way twice clears it. Returns the stored vote."""
        votes = self.votes()
        current = votes.get(url, 0)
        vote = 0 if current == vote else vote
        if vote:
            votes[url] = vote
        else:
            votes.pop(url, None)
        self._write(self.dir / "feedback.json", votes)
        day = self.load_day(date)
        for report in day["reports"]:
            if report.url == url:
                report.vote = vote
        self.save_day(date, day["reports"], day["skipped"], day["warnings"], day["ranking"])
        return vote

    # what each delivery channel has already been sent ------------------------------

    def delivered(self) -> dict[str, list[str]]:
        return self._read(self.dir / "delivered.json", {})

    def mark_delivered(self, channel: str, keys: list[str]) -> None:
        data = self.delivered()
        merged = sorted(set(data.get(channel, [])) | set(keys))
        data[channel] = merged[-1000:]  # keys start with the date, so the oldest go first
        self._write(self.dir / "delivered.json", data)

    # daily reports -------------------------------------------------------------

    def _day_path(self, date: str) -> Path:
        return self.reports_dir / f"{date}.json"

    def load_day(self, date: str) -> dict[str, Any]:
        raw = self._read(self._day_path(date), {})
        votes = self.votes()
        reports = [Report.from_dict(r) for r in raw.get("reports", [])]
        for report in reports:
            report.vote = votes.get(report.url, 0)
        return {
            "reports": reports,
            "skipped": [SkippedSource(**s) for s in raw.get("skipped", [])],
            "warnings": list(raw.get("warnings", [])),
            "ranking": raw.get("ranking", ""),
        }

    def save_day(
        self,
        date: str,
        reports: list[Report],
        skipped: list[SkippedSource],
        warnings: list[str],
        ranking: str,
    ) -> None:
        self._write(
            self._day_path(date),
            {
                "reports": [r.to_dict() for r in reports],
                "skipped": [{"source": s.source, "reason": s.reason} for s in skipped],
                "warnings": warnings,
                "ranking": ranking,
            },
        )

    def merge_day(
        self,
        date: str,
        new_reports: list[Report],
        skipped: list[SkippedSource],
        warnings: list[str],
        ranking: str,
    ) -> list[Report]:
        """Add a scan's reports to the day's file. A later scan replaces the same URL."""
        day = self.load_day(date)
        by_url = {r.url: r for r in day["reports"]}
        for report in new_reports:
            by_url[report.url] = report
        merged = list(by_url.values())
        self.save_day(date, merged, skipped, warnings, ranking)
        return merged

    def report_dates(self) -> list[str]:
        if not self.reports_dir.exists():
            return []
        return sorted(p.stem for p in self.reports_dir.glob("*.json"))
