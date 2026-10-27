# research-radar

A daily research radar for the terminal that runs on your own computer.

You describe what you work on and what you are curious about. Research Radar looks at new GitHub repositories, arXiv papers, Hugging Face papers and models, and Codeberg repositories, keeps the few that fit your interests, and writes a short note about each one. The matching and the summaries are done by a local [Ollama](https://ollama.com) model, so there is no account, no API key for a language model, no usage limit, and your interests never leave your machine.

It does one thing: scan, filter and summarise. There is no chat.

## How a scan works

1. Each source fetches what is new since the last successful scan of that source.
2. Items you have already been shown are dropped.
3. An embedding model (`nomic-embed-text` by default) compares each item with the topic descriptions in your `interests.yaml`. This step needs no language model and cuts hundreds of items to a few dozen.
4. The best few, with a cap per topic and a cap overall, go to a chat model that writes one sentence on what the item is, one on why it fits, and a recommendation: install, read, watch or skip.
5. The result is saved as a markdown note, `notes/YYYY/MM/YYYY-MM-DD.md`, and shown in the terminal interface.

The chat model only rewrites the text it is given (a README, an abstract). Links, star counts, licenses and dates come from the source's own fields, never from the model. If a topic has nothing good, the note says so instead of padding.

If Ollama is not running, a scan still works: items are matched by keywords and described with the first sentence of their own text. The note says so.

## Install

You need Python 3.10 or newer and Ollama.

```
git clone https://github.com/HEX-1000/research-radar
cd research-radar
pip install -e .
```

Install the models once:

```
ollama pull nomic-embed-text
ollama pull <any chat model you like>
```

If `chat_model` is empty in your settings, the first installed non-embedding model is used. A 7B to 8B model is enough, because it only summarises short texts.

## First run

```
radar init      # creates ~/.research-radar/interests.yaml from the example
radar doctor    # checks Ollama and every source from your machine
radar           # opens the terminal interface
```

Edit `interests.yaml` before the first real scan. The example has two topics. Replace them with your own, and write each description the way you would explain it to a colleague: what you work on, which tools you use, what you want to hear about. The better the description, the better the matching. The file documents every setting.

To keep your data somewhere else, set `RADAR_HOME` or pass `--home`.

## The terminal interface

| Key | Action |
| --- | --- |
| `s` | scan now |
| `+` / `-` | mark the selected item useful or not useful (press again to clear) |
| `o` | open the link in your browser |
| `[` / `]` | older day / newer day |
| `q` | quit |

On launch it catches up automatically if any source has not been scanned today (`auto_scan_on_launch`). Votes are kept and nudge later rankings: items that look like ones you marked `+` move up, items that look like ones you marked `-` move down. This is a simple heuristic, not a trained model.

## Scheduling (optional)

The app has no timer and no background process. Because every scan starts from the last successful run of each source, nothing is lost if the computer was off for a few days: the next scan collects the whole gap, up to `max_catchup_days`.

If you want scans without opening the interface, `radar scan` does the same work from the command line.

On Windows, create a Task Scheduler task that runs `radar scan`, and on the Settings tab enable "Run task as soon as possible after a scheduled start is missed". On Linux or macOS, a cron line such as `0 19 * * * radar scan` works. Useful flags: `--days N` to look back a fixed number of days, `--no-llm` to skip Ollama, `--explain` to print the scores of the top 30 candidates so you can tune `min_score`.

## Sources and their limits

| Source | What it reads | Notes |
| --- | --- | --- |
| `github` | new repositories from the search API, per query in your topics | Without a token the search API allows about 10 requests a minute, with one about 30. Set the environment variable `GITHUB_TOKEN` to a token with no scopes. GitHub has no official trending API, so "new" means created since the last scan. Repositories you were shown earlier that gained many stars are reported once as "rising". |
| `arxiv` | the official Atom API, one query per topic | Requests are spaced about 3 seconds apart, as arXiv asks. |
| `huggingface-papers` | the daily papers list for each day in the window | |
| `huggingface-models` | currently trending models | Matches on tags only, so it is the noisiest source. |
| `forgejo` | repository search on Codeberg or any Gitea or Forgejo server | Add servers under `sources.forgejo.instances`. |

Each source is a small module in `src/research_radar/sources/`. A source takes the date window and your topics and returns a list of items, so adding one is a short file. Pull requests are welcome.

## Privacy

- Your interests, notes, votes and history live in your data folder and are not sent anywhere.
- The only network traffic is requests to the sources above and to your local Ollama.
- There is no telemetry.
- `.gitignore` excludes `interests.yaml`, `notes/`, `state/`, vote files and `.env`, so running the app from inside a clone cannot commit personal data by accident.

## Status

Version 0.1. The test suite covers the pipeline, ranking, notes, the interface and every source parser. The parsers for GitHub, arXiv, Hugging Face and Codeberg were written from their documented response formats and tested against saved examples. They have not yet been checked against the live services from the author's machine, so run `radar doctor` first. It reads one real item from each source and reports `FAIL` with a reason when a format has changed. If it does, please open an issue with the output.

The default `min_score` of 0.55 has not been tuned. If you get too much or too little, change it and look at `radar scan --explain`.

## Development

```
pip install -e ".[dev]"
pytest
```

Tests run offline with fake sources and a fake Ollama.

## License

MIT. See `LICENSE`.
