# Security

## Supported versions

The latest version only. Installed copies update themselves when the interface opens, unless `auto_update: false` is set or the program was installed with a plain `pip install`. Versions before 0.2.0 have no update check.

## Reporting a vulnerability

Report it privately through this repository's Security tab: "Report a vulnerability". Please do not open a public issue for a vulnerability. Leave tokens, passwords, your email address and your Telegram chat id out of the report.

## In scope

- The self-update. It downloads code from this repository (`git pull` or the repository ZIP) and runs it. An update that installs code from anywhere else, or writes outside the program folder or over the data folder, `.git` or `.venv`, is in scope.
- The secrets. The SMTP password, the Telegram bot token, the Codeberg or other Forgejo token and the GitHub token (`GITHUB_TOKEN`) are read from environment variables. Each belongs to one host: the mail server in `interests.yaml`, `api.telegram.org`, the Forgejo servers it is named for, `api.github.com`. A secret sent to another host, written to a file, or shown in an error message or in `radar doctor` is in scope.
- Delivery. A digest or a message that goes to an address or a chat other than the ones in `interests.yaml`, and a vote accepted from a Telegram chat other than `delivery.telegram.chat_id`.
- Text from the sources. Titles, descriptions, READMEs, abstracts and feed entries are written by strangers. Markup in them that runs in the digest page, the email or a Telegram message, a feed that makes the program read a local file or run out of memory, and source text that puts a link into a note that is not in the item's own fields.
- The daily task (`radar schedule on`): a task that runs anything other than a scan of this installation, or runs with more rights than yours.
- Anything else that sends your interests, votes, labels or notes off the computer.

## Out of scope

- Vulnerabilities in Ollama, Python, the packages Research Radar uses, or the services it reads from and sends to (GitHub, arXiv, Hugging Face, Codeberg, Telegram, your mail server). Report those to their projects.
- What the model writes in a note, as long as the links, numbers and dates come from the source.
- What follows from your own settings: `delivery.email.security: none` sends the mail password without encryption, and an `ollama_url` on another machine sends the item texts there.
- An attacker who can already run programs as you or read your environment variables.

There is no promise about response time.
