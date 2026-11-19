# Changelog

What changed in each version, newest first. After an update the app shows the entries
between your old version and the new one. Keep each entry to one short line a user can
understand. The version number lives in `src/research_radar/__init__.py`; raising it is
what makes an update reach everyone.

## 0.5.0

- The starter `interests.yaml` that `radar init` copies has two new example topics, "Example: Web accessibility tools" and "Example: Protein structure prediction". They are invented and marked as examples in the file. A file you already have is not changed; if it still holds the old example topics, the app treats them as examples as before.
- When a scheduled scan is skipped and no scan has been saved for two days or more, you get one message by email or Telegram (whichever is on) with the date of the last saved scan and the reason. It is sent once until a scan is saved again.
- A space, hyphen or underscore in a keyword or an exclude phrase matches any one of the three: `screen reader` finds `screen-reader` and `alt-text` finds `alt text`.
- Votes no longer change the ranking (`feedback_weight` is 0 by default). They are still kept, count as labels in `radar replay` and keep the item's data past 30 days.
- `radar doctor` shortens your email address (`a***@example.org`), Telegram chat id (`...123`) and bot name (`@abc***`), so its output is safe to post. `radar telegram-chat` still prints the full chat id.
- `radar label` has a new answer `w`: you want the item, but no topic fits. It is not asked about again, and `radar replay` counts it as a "no" for the topic it was asked under.
- A GitHub repository or model is dropped as a lure when its name and topics carry two different selling endings on one stem, such as `nvidia-broadcast-download` with `nvidia-broadcast-free` (endings: download, free, activated, crack, license). One ending alone, such as `ad-free`, is still fine.
- The trust check also holds a repository under 200 KB with fewer than 10 stars when its owner has no followers (from 10 stars it is still 2 followers or fewer). `radar replay` and the follower counts the scan asks for use the same rule.
- Updates now come from a `stable` branch of the repository, which GitHub moves to a commit of `main` only after every test has passed for it, so a copy never installs code that was not checked. Copies on 0.2.0 to 0.4.0 read `main` and switch to `stable` with this update.

## 0.4.0

- A watched GitHub feed is named `owner/name`, also when you list its `commits.atom`, and its items are releases, commits or (any other feed) articles. A short commit message gets "New commit to owner/name: ...", and the count of earlier items says releases, commits or entries.
- A feed entry whose id is not a URL (a bare number such as `1234`) is stored with its feed's address, so two feeds with the same short id are no longer taken for one entry.
- The scan asks GitHub for the follower count of small, starred repositories that it did not show but that came close (up to 10 owners per scan), so `radar replay` can say how the trust check does on them.
- The `huggingface-models` source is off by default, because it matches on tags only and no model ever reached the bar. A file that names the source keeps what it says.
- A new `feeds` source reads Atom and RSS. A topic's `watch` list names GitHub repositories (`owner/name`, read through their releases feed) or feed URLs; what appears there is shown under that topic without a bar and outside `max_items` and `max_per_topic`, at most one item per feed and scan, with "and N earlier entries since ..." when there were more. A topic's `feeds: urls:` lists blogs whose posts are scored like any other item. `radar doctor` reads every feed once.
- `radar label` also asks about GitHub repositories whose only signal is a `-` vote, and `radar replay` names the repositories behind "others caught" and "lures not caught".
- Telegram votes are saved page by page, so a failure while reading a later page no longer loses the votes of an earlier one.
- Telegram sends one message per item, each with `+` and `-` buttons. A vote is recorded at the next scan (also at a skipped scheduled scan) and the button then shows a check mark. Only the first message of a digest makes a sound, and an item counts as delivered as soon as its message is sent.
- `radar label` asks whether recent candidates fit your topics and saves each answer; `radar replay` shows, for each topic, what every bar from 0.30 to 0.60 would have kept, missed and let through, and how the trust check does on your labelled repositories.
- A topic can set its own `min_score` and `max_per_topic` (`max_per_topic: 0` mutes it). An item goes to the topic whose bar it clears by the most, and `radar scan --explain` shows the bar next to each score.
- The rising check now asks about every repository from the last `rising_days`; a deleted one no longer stops it and is named once in the notes.
- A rising repository shows its license instead of "license not stated".
- Keywords of four letters or fewer match whole words only ("stac" no longer matches "stack"); longer ones match at the start of a word.
- Repositories and models whose name, description or topics offer cracked software, keygens or license keys are dropped.
- A second scan on the same day keeps the notes of the first, and the note says which items were matched by keywords.
- A misspelled key in `interests.yaml` (setting, source, topic option or delivery option) is ignored and named in a notice, in the scan output, `radar doctor`, the digest and the interface; the scan still runs.
- A scan that can only match by keywords (`--no-llm`, Ollama down, or the embedding failing) is now a preview: it shows its items and saves and sends nothing, so the next scan covers the same window. The scheduled scan exits as skipped.
- Two scans cannot run at once: the second stops with "Another scan has been running since HH:MM." A lock older than 2 hours is taken over.
- Small GitHub repositories with at least 10 stars from accounts with 2 followers or fewer are held back for a week and checked again; `trust_check: false` turns this off.
- The note a model writes for a repository now starts from its description and topics, then the README, and the model is told to ignore instructions inside the source text.
- Every item a scan receives is kept in `state/radar.db`, which replaces `seen.json` and `vectors.json`. The first scan moves them over and renames them to `*.migrated`. A version older than this one finds no `seen.json` and would report old items again.
- An item fetched again is not scored or shown again. An edited topic, or a new embedding model, gets one fresh look at the items in the window.
- The same paper from arXiv and from Hugging Face papers is one item.
- arXiv is searched from 5 days before the last good run, because a paper is announced a few days after it is submitted; papers already seen are not shown again.
- An arXiv search reads further pages (up to 5 pages of 100) until a page comes back short.
- `state/radar.db` is now schema 2: it holds your votes (`vote`) and has a `label` table that `radar label` writes to. A schema 1 file is upgraded in place the first time a scan or a vote writes to it, and `feedback.json` is imported and renamed to `feedback.json.migrated`. An older version finds no `feedback.json` and would not see your votes.
- A vector is kept for an item you voted on or labelled, not only for shown ones. Every connection to the store waits up to 10 seconds when another one is writing.

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
