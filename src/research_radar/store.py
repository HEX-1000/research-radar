"""What the app remembers as small JSON files under <home>/state: day reports, votes, deliveries,
source windows and topic marks. Every item a scan received lives in state/radar.db (candidates.py)."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from . import candidates
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

    # per source last successful scan ------------------------------------------------

    def last_runs(self) -> dict[str, str]:
        return self._read(self.dir / "last_run.json", {})

    def save_last_runs(self, runs: dict[str, str]) -> None:
        self._write(self.dir / "last_run.json", runs)

    # fingerprints of the topics at the last scan, to notice new or edited ones ------------

    def topic_marks(self) -> dict[str, str] | None:
        """None when no scan has recorded them yet."""
        data = self._read(self.dir / "topics.json", None)
        return data if isinstance(data, dict) else None

    def save_topic_marks(self, marks: dict[str, str]) -> None:
        self._write(self.dir / "topics.json", marks)

    # votes, by URL; kept in the candidate store (feedback.json before schema 2) ----------------

    def votes(self) -> dict[str, int]:
        return candidates.read_votes(self.dir)

    def set_vote(self, date: str, url: str, vote: int, via: str = "tui") -> int:
        """Set a vote. Voting the same way twice clears it. Returns the stored vote."""
        vote = 0 if self.votes().get(url, 0) == vote else vote
        candidates.write_vote(self.dir, url, vote, via)
        day = self.load_day(date)
        for report in day["reports"]:
            if report.url == url:
                report.vote = vote
        self.save_day(date, day["reports"], day["scans"])
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
        """The day's reports, the record of each scan that day, and what those scans had to say.

        `skipped` and `warnings` gather every scan of the day, without repeats. `ranking` is
        "keywords" when any scan of the day ranked by keywords; reports carry their own mode."""
        raw = self._read(self._day_path(date), {})
        votes = self.votes()
        reports = [Report.from_dict(r) for r in raw.get("reports", [])]
        for report in reports:
            report.vote = votes.get(report.url, 0)
        scans = raw.get("scans")
        if scans is None:  # written by an older version: one set of notes for the whole day
            scans = [
                {
                    "at": "",
                    "ranking": raw.get("ranking", ""),
                    "new": None,
                    "skipped": raw.get("skipped", []),
                    "warnings": raw.get("warnings", []),
                }
            ] if raw else []
        skipped: list[SkippedSource] = []
        warnings: list[str] = []
        for scan in scans:
            for entry in scan.get("skipped", []):
                source = SkippedSource(**entry)
                if source not in skipped:
                    skipped.append(source)
            warnings.extend(w for w in scan.get("warnings", []) if w not in warnings)
        modes = [s.get("ranking", "") for s in scans]
        ranking = "keywords" if "keywords" in modes else (modes[-1] if modes else "")
        return {"reports": reports, "scans": scans, "skipped": skipped, "warnings": warnings, "ranking": ranking}

    def save_day(self, date: str, reports: list[Report], scans: list[dict[str, Any]]) -> None:
        self._write(self._day_path(date), {"reports": [r.to_dict() for r in reports], "scans": scans})

    def merge_day(
        self,
        date: str,
        new_reports: list[Report],
        skipped: list[SkippedSource],
        warnings: list[str],
        ranking: str,
        at: str = "",
    ) -> dict[str, Any]:
        """Add a scan's reports and notes to the day's file and return the day as load_day does.
        A later scan replaces a report with the same URL; the notes of earlier scans stay."""
        day = self.load_day(date)
        by_url = {r.url: r for r in day["reports"]}
        for report in new_reports:
            by_url[report.url] = report
        scans = day["scans"] + [
            {
                "at": at,
                "ranking": ranking,
                "new": len(new_reports),
                "skipped": [{"source": s.source, "reason": s.reason} for s in skipped],
                "warnings": list(warnings),
            }
        ]
        self.save_day(date, list(by_url.values()), scans)
        return self.load_day(date)

    def report_dates(self) -> list[str]:
        if not self.reports_dir.exists():
            return []
        return sorted(p.stem for p in self.reports_dir.glob("*.json"))
