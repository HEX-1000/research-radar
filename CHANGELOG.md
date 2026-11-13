# Changelog

What changed in each version, newest first. After an update the app shows the entries
between your old version and the new one. Keep each entry to one short line a user can
understand. The version number lives in `src/research_radar/__init__.py`; raising it is
what makes an update reach everyone.

## Unreleased

- The rising check now asks about every repository from the last `rising_days`; a deleted one no longer stops it and is named once in the notes.
- A rising repository shows its license instead of "license not stated".
- Keywords of four letters or fewer match whole words only ("stac" no longer matches "stack"); longer ones match at the start of a word.
- Repositories and models whose name, description or topics offer cracked software, keygens or license keys are dropped.
- A second scan on the same day keeps the notes of the first, and the note says which items were matched by keywords.
- A misspelled setting or topic option in `interests.yaml` is now an error that names it, instead of being ignored.

- Small GitHub repositories with at least 10 stars from accounts with 2 followers or fewer are held back for a week and checked again; `trust_check: false` turns this off.
- The note a model writes for a repository now starts from its description and topics, then the README, and the model is told to ignore instructions inside the source text.

## 0.3.1

- Codeberg now refuses searches without an access token. Put one in `RADAR_CODEBERG_TOKEN` (see the README) and Codeberg works again.
- When Codeberg refuses, the scan asks once instead of for every topic, and the note says what the server answered.
- A broken `interests.yaml` no longer stops the daily scan: it uses the last working copy, and the digest, the messages and `radar doctor` name the line to fix.

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
