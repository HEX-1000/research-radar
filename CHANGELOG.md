# Changelog

What changed in each version, newest first. After an update the app shows the entries
between your old version and the new one. Keep each entry to one short line a user can
understand. The version number lives in `src/research_radar/__init__.py`; raising it is
what makes an update reach everyone.

## 0.3.0

- A topic you add or edit is searched back `first_run_days` once, so it does not start empty.
- `radar schedule on` sets up a daily scan in the Windows Task Scheduler. It opens no window, writes `logs/scan.log` and runs after a missed start.
- After an update the app now lists what changed.

## 0.2.0

- The app updates itself when it opens. Turn it off with `auto_update: false`, or run `radar update`.
- `install.ps1` sets up Windows and checks Ollama and both models.
- The app says what to do when Ollama or a model is missing, and no longer starts a keyword-only scan on its own.
- `f1` or `?` shows all keys, `e` opens your `interests.yaml`.

## 0.1.0

- First version: GitHub, arXiv, Hugging Face and Codeberg sources, local ranking and notes, the terminal interface, the daily digest by page, email or Telegram.
