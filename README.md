# research-radar

A daily research log. Every evening at 19:45 (Turkey time) an assistant scans GitHub, Hugging Face, arXiv and a few ranking lists, picks the new tools, models and papers that fit the topics below, and writes a short note.

## Topics

HEX 1000 (CS/AI education channel), remote sensing / geology / archaeology, HEX INDUSTRIES (energy storage and digital logic hardware), the Token Optimization System, and finance tools. Details are in `interests.md`.

## Files

- `notes/YYYY/MM/YYYY-MM-DD.md`: daily notes
- `interests.md`: the topics and what to look for. This file is the assistant's main guide.
- `feedback.md`: a summary of feedback given on earlier notes
- `state/seen.json`: items already reported, so the same thing is not written up twice

## Note format

Each item says three things: what it is, why it fits these topics, and a recommendation (install, read, watch or skip). Every claim has a source link. Anything that could not be verified is marked "unverified".

## Feedback

In a note's "Feedback" section, write the item name followed by `+` (useful) or `-` (not useful) and a short comment. The next run moves these into `feedback.md` and adjusts what it picks.

Notes are generated automatically. Check the source before installing or relying on anything.
