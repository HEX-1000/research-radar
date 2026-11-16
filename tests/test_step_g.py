"""Step G: `radar label` (pool, order, answers) and `radar replay` (counts per bar, trust check)."""

from __future__ import annotations

import math
import sqlite3
from dataclasses import replace
from datetime import datetime, timedelta, timezone

import pytest
import yaml

from research_radar import candidates, cli, labelling
from research_radar.candidates import CandidateStore
from research_radar.config import Topic, example_text, init_home, parse_config
from research_radar.labelling import bar_rows, build_pool, format_replay, replay, run_label, run_replay
from research_radar.llm import OllamaError

NOW = datetime(2026, 11, 14, 12, 0, tzinfo=timezone.utc)
MODEL = "nomic-embed-text"
RECENT = (NOW - timedelta(days=1)).isoformat()
OLD = (NOW - timedelta(days=20)).isoformat()


def make_cfg(tmp_path, bars=(0.40, None), **settings):
    data = yaml.safe_load(example_text())
    data["settings"].update({"min_score": 0.50, "embed_model": MODEL, **settings})
    cfg = parse_config(data, tmp_path / "home")
    topics = [
        Topic("Alpha", "alpha things", keywords=["alphakey"], min_score=bars[0]),
        Topic("Beta", "beta things", min_score=bars[1]),
        Topic("Gamma", "gamma things", min_score=0.30),
    ]
    return replace(cfg, topics=topics)


class Db:
    """A schema 2 store written by hand, the way the scans leave it."""

    def __init__(self, cfg):
        cfg.state_dir.mkdir(parents=True, exist_ok=True)
        self.path = cfg.state_dir / "radar.db"
        self.conn = sqlite3.connect(self.path, isolation_level=None)
        self.conn.execute("PRAGMA foreign_keys = ON")
        self.conn.executescript(candidates.SCHEMA)
        self.conn.execute("PRAGMA user_version = 2")
        self.conn.execute("INSERT INTO scan VALUES (1, ?, '1', ?, 'c', '{}', 0)", (OLD, MODEL))
        self.conn.execute("INSERT INTO scan VALUES (2, ?, '1', ?, 'c', '{}', 0)", (RECENT, MODEL))

    def item(self, key, title=None, text="", topic=None, kind="repo", published="2026-11-10T08:00:00Z", source=None):
        source = source if source is not None else key.split(":")[0]
        url = f"https://example.org/{key.replace(':', '/')}"
        self.conn.execute(
            "INSERT INTO item (key, kind, url, title, text, published, first_seen, last_seen, state, state_on, topic)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'seen', '2026-11-13', ?)",
            (key, kind, url, title or key, text, published, RECENT, RECENT, topic),
        )
        self.conn.execute("INSERT INTO link (key, source, url) VALUES (?, ?, ?)", (key, source, url))
        return key

    def decide(self, key, topic, score, outcome="below_bar", scan=2):
        self.conn.execute("INSERT INTO decision VALUES (?, ?, ?, ?, ?)", (scan, key, topic, score, outcome))

    def label(self, key, topic, value):
        self.conn.execute("INSERT INTO label VALUES (?, ?, ?, '2026-11-13T00:00:00+00:00')", (key, topic, value))

    def vote(self, key, value):
        self.conn.execute("INSERT INTO vote VALUES (?, ?, '2026-11-13T00:00:00+00:00', 'tui')", (key, value))

    def vector(self, key, similarity, axis=0, model=MODEL):
        """A vector whose cosine with topic axis `axis` is `similarity`."""
        vec = [0.0, 0.0, 0.0]
        vec[axis] = similarity
        vec[(axis + 1) % 3] = math.sqrt(1 - similarity**2)
        self.conn.execute("INSERT INTO vector VALUES (?, ?, ?)", (key, model, candidates._pack(vec)))

    def score(self, key, topic, similarity, hits=0, mark="m", model=MODEL):
        self.conn.execute("INSERT INTO score VALUES (?, ?, ?, ?, ?, ?, 2)", (key, topic, mark, model, similarity, hits))

    def observe(self, key, size_kb, stars, followers, at="2026-11-01T00:00:00+00:00"):
        self.conn.execute("INSERT INTO observation VALUES (?, ?, ?, 0, ?, ?)", (key, at, stars, size_kb, followers))

    def rows(self, sql):
        return self.conn.execute(sql).fetchall()


class AxisOllama:
    """Topic texts get a fixed axis: Alpha 0, Beta 1, Gamma 2. Keyword lists are a second vector."""

    def __init__(self, up=True):
        self.up = up

    def ping(self):
        return self.up

    def embed(self, texts, kind="document"):
        if not self.up:
            raise OllamaError("down")
        axes = {"alpha": 0, "beta": 1, "gamma": 2}
        out = []
        for text in texts:
            vec = [0.0, 0.0, 0.0]
            vec[next(i for name, i in axes.items() if name in text.lower())] = 1.0
            out.append(vec)
        return out


def open_store(cfg, write=True):
    return CandidateStore.open(cfg.state_dir, embed_model=MODEL, now=NOW, write=write)


def pool_of(cfg, days=14):
    with open_store(cfg, write=False) as store:
        return build_pool(store, cfg, NOW, days)


# the pool ---------------------------------------------------------------------------------------


def test_pool_applies_band_outcome_topic_and_days(tmp_path):
    cfg = make_cfg(tmp_path)  # Alpha bar 0.40, Beta 0.50 (settings), Gamma 0.30
    db = Db(cfg)
    for key, topic, score, outcome in [
        ("github:in-band", "Alpha", 0.33, "below_bar"),  # bar 0.40, band down to 0.32
        ("github:below-band", "Alpha", 0.31, "below_bar"),
        ("github:shown", "Alpha", 0.70, "shown"),
        ("github:held", "Beta", 0.55, "held"),
        ("github:topic-full", "Beta", 0.52, "topic_full"),
        ("github:day-full", "Gamma", 0.35, "day_full"),
        ("github:lure", "", 0.0, "lure"),
        ("github:excluded", "Alpha", 0.9, "excluded"),
        ("github:watch", "Alpha", 0.0, "watch"),
        ("github:gone-topic", "Old topic", 0.9, "shown"),
    ]:
        db.item(key)
        db.decide(key, topic, score, outcome)
    db.item("github:old-only")
    db.decide("github:old-only", "Alpha", 0.45, "shown", scan=1)  # 20 days ago, outside the 14 days
    assert sorted(c.key for c in pool_of(cfg)) == sorted(
        ["github:in-band", "github:shown", "github:held", "github:topic-full", "github:day-full"]
    )
    assert "github:old-only" in {c.key for c in pool_of(cfg, days=30)}


def test_pool_uses_the_latest_decision_row_in_the_window(tmp_path):
    cfg = make_cfg(tmp_path)
    db = Db(cfg)
    db.item("github:a")
    db.decide("github:a", "Alpha", 0.45, "shown", scan=1)
    db.decide("github:a", "Alpha", 0.10, "below_bar", scan=2)  # the latest row is far below the band
    db.item("github:b")
    db.decide("github:b", "Alpha", 0.10, "below_bar", scan=1)
    db.decide("github:b", "Alpha", 0.45, "shown", scan=2)
    assert [c.key for c in pool_of(cfg, days=30)] == ["github:b"]


def test_labelled_and_voted_items_are_left_out(tmp_path):
    cfg = make_cfg(tmp_path)
    db = Db(cfg)
    for key in ("github:labelled-here", "github:labelled-elsewhere", "github:lure", "github:voted", "github:plain"):
        db.item(key)
        db.decide(key, "Alpha", 0.45, "shown")
    db.label("github:labelled-here", "Alpha", "no")
    db.label("github:labelled-elsewhere", "Beta", "fits")  # a label for another topic does not count
    db.label("github:lure", "*", "lure")
    db.vote("github:voted", 1)  # a - vote on a repository is asked about (step J)
    assert sorted(c.key for c in pool_of(cfg)) == ["github:labelled-elsewhere", "github:plain"]


def test_pool_order_is_nearest_to_the_bar_then_one_topic_at_a_time(tmp_path):
    cfg = make_cfg(tmp_path)  # Alpha 0.40, Beta 0.50, Gamma 0.30
    db = Db(cfg)
    for key, topic, score in [
        ("github:a-far", "Alpha", 0.60),  # 0.20 from the bar
        ("github:a-near", "Alpha", 0.41),
        ("github:a-below", "Alpha", 0.35),  # 0.05 from the bar
        ("github:b-near", "Beta", 0.52),
        ("github:b-far", "Beta", 0.80),
        ("github:g-only", "Gamma", 0.31),
    ]:
        db.item(key)
        db.decide(key, topic, score, "shown")
    order = [c.key for c in pool_of(cfg)]
    assert order == [
        "github:a-near", "github:b-near", "github:g-only",  # the nearest of each topic, in interests.yaml order
        "github:a-below", "github:b-far",
        "github:a-far",
    ]


# radar label ------------------------------------------------------------------------------------


def scripted(answers, db=None, seen=None):
    """An `ask` that returns the answers in turn; `seen` collects the label rows before each question."""
    iterator = iter(answers)

    def ask(prompt):
        if seen is not None:
            seen.append(db.rows("SELECT key, topic, value FROM label ORDER BY key, topic"))
        try:
            return next(iterator)
        except StopIteration:
            raise EOFError

    return ask


def three_items(tmp_path):
    cfg = make_cfg(tmp_path)
    db = Db(cfg)
    for key, score in [("github:one", 0.41), ("github:two", 0.42), ("github:three", 0.43)]:
        db.item(key, title=f"Title of {key}", text="text " * 200)
        db.decide(key, "Alpha", score, "below_bar")
    return cfg, db


def test_each_answer_type_writes_its_rows_at_once(tmp_path):
    cfg, db = three_items(tmp_path)
    shown, seen, opened = [], [], []
    code = run_label(
        cfg,
        ask=scripted(["y", "2", "l"], db, seen),
        show=shown.append,
        open_url=opened.append,
        now=NOW,
    )
    assert code == 0
    assert db.rows("SELECT key, topic, value FROM label ORDER BY key, topic") == [
        ("github:one", "Alpha", "fits"),
        ("github:three", "*", "lure"),
        ("github:two", "Alpha", "no"),  # a digit: fits that topic, no for this one
        ("github:two", "Beta", "fits"),
    ]
    # Before each question the earlier answers were already in the file.
    assert [len(rows) for rows in seen] == [0, 1, 3]
    assert not opened


def test_n_writes_no_and_the_digit_of_the_same_topic_means_yes(tmp_path):
    cfg, db = three_items(tmp_path)
    run_label(cfg, ask=scripted(["n", "1", "q"]), show=lambda _: None, now=NOW)
    assert db.rows("SELECT key, topic, value FROM label ORDER BY key") == [
        ("github:one", "Alpha", "no"),
        ("github:two", "Alpha", "fits"),
    ]


def test_w_saves_a_no_and_a_wanted_row_and_the_item_leaves_the_pool(tmp_path):
    cfg, db = three_items(tmp_path)
    run_label(cfg, ask=scripted(["w", "q"]), show=lambda _: None, now=NOW)
    rows = db.rows("SELECT key, topic, value FROM label")
    key = rows[0][0]
    assert sorted((t, v) for _, t, v in rows) == [("*", "wanted"), ("Alpha", "no")]
    assert key not in {c.key for c in pool_of(cfg)}
    set_up = labelling.labelled_set(CandidateStore.open(cfg.state_dir, embed_model="m", now=NOW), cfg.topics)
    entries, lures = set_up[0], set_up[2]
    assert [v for (k, t), v in entries.items() if k == key] == ["no"] and key not in lures


def test_o_opens_the_link_and_asks_again_s_skips_q_ends(tmp_path):
    cfg, db = three_items(tmp_path)
    shown, opened, asked = [], [], []
    ask = scripted(["o", "nonsense", "s", "y", "q"])
    run_label(cfg, ask=lambda p: asked.append(p) or ask(p), show=shown.append, open_url=opened.append, now=NOW)
    assert opened == ["https://example.org/github/one"]
    assert len(asked) == 5
    assert db.rows("SELECT key, topic, value FROM label") == [("github:two", "Alpha", "fits")]  # one was skipped
    assert any(line.startswith("Not an answer") for line in shown)
    assert db.rows("SELECT COUNT(*) FROM label")[0][0] == 1  # nothing for the third: q ended it
    assert shown[-1] == "Labelled 1 item."


def test_the_screen_shows_the_item_and_a_cut_text(tmp_path):
    cfg, _ = three_items(tmp_path)
    shown = []
    run_label(cfg, ask=scripted(["q"]), show=shown.append, now=NOW)
    text = "\n".join(shown)
    assert "  1  Alpha (bar 0.40)" in text and "  3  Gamma (bar 0.30)" in text
    assert "[3 left] Alpha · bar 0.40 · score 0.410 · below_bar" in text
    assert "Title of github:one" in text
    assert "repo · github · 2026-11-10" in text
    assert "https://example.org/github/one" in text
    body = next(line for line in shown if line.startswith("text text"))
    assert len(body) == 400


def test_an_item_without_text_says_so(tmp_path):
    cfg = make_cfg(tmp_path)
    db = Db(cfg)
    db.item("github:bare")
    db.decide("github:bare", "Alpha", 0.45, "shown")
    shown = []
    run_label(cfg, ask=scripted(["q"]), show=shown.append, now=NOW)
    assert "(no text stored)" in shown


def test_end_of_input_quits_and_an_empty_pool_says_so(tmp_path):
    cfg, db = three_items(tmp_path)
    assert run_label(cfg, ask=scripted(["y"]), show=lambda _: None, now=NOW) == 0  # then EOF
    assert db.rows("SELECT COUNT(*) FROM label")[0][0] == 1
    shown = []
    assert run_label(cfg, days=14, ask=scripted([]), show=shown.append, now=NOW + timedelta(days=60)) == 0
    assert shown == ["Nothing to label from the last 14 days."]


def test_label_without_a_store_says_so(tmp_path):
    cfg = make_cfg(tmp_path)
    shown = []
    assert run_label(cfg, show=shown.append) == 1
    assert shown == ["No candidate store yet. Run a scan first."]
    assert not (cfg.state_dir / "radar.db").exists()  # the command never creates the store


def test_label_upgrades_a_schema_1_store(tmp_path):
    cfg, db = three_items(tmp_path)
    db.conn.executescript("DROP TABLE label; DROP TABLE vote;")
    db.conn.execute("PRAGMA user_version = 1")
    run_label(cfg, ask=scripted(["y", "q"]), show=lambda _: None, now=NOW)
    assert db.rows("SELECT key, topic, value FROM label") == [("github:one", "Alpha", "fits")]


def test_the_cli_commands_work_on_the_data_folder(tmp_path, capsys):
    home = tmp_path / "home"
    init_home(home)
    assert cli.main(["--home", str(home), "label"]) == 1
    assert "No candidate store yet. Run a scan first." in capsys.readouterr().out
    assert cli.main(["--home", str(home), "replay"]) == 1
    assert "No candidate store yet. Run a scan first." in capsys.readouterr().out
    args = cli.build_parser().parse_args(["label", "--days", "5"])
    assert (args.command, args.days) == ("label", 5)


# radar replay -----------------------------------------------------------------------------------


def alpha_set(db):
    """Alpha (bar 0.40): fits scoring 0.9 and 0.55, no scoring 0.45 and 0.2."""
    for key, sim, value in [
        ("github:f1", 0.9, "fits"),
        ("github:f2", 0.55, "fits"),
        ("github:n1", 0.45, "no"),
        ("github:n2", 0.2, "no"),
    ]:
        db.item(key)
        db.vector(key, sim)
        db.label(key, "Alpha", value)


def table(result, name="Alpha"):
    row = next(r for r in result.topics if r.topic.name == name)
    return {bar: (kept, noise, missed) for bar, kept, noise, missed in bar_rows(row)}


def test_replay_counts_kept_noise_and_missed_per_bar(tmp_path):
    cfg = make_cfg(tmp_path)
    db = Db(cfg)
    alpha_set(db)
    result = replay(cfg, AxisOllama())
    assert result.mode == "embeddings"
    row = result.topics[0]
    assert (len(row.fits), len(row.no)) == (2, 2)
    rows = table(result)
    assert list(rows)[0] == 0.30 and list(rows)[-1] == 0.60
    assert rows[0.30] == (2, 1, 0)  # the 0.45 no is above it, the 0.2 no is not
    assert rows[0.44] == (2, 1, 0)
    assert rows[0.46] == (2, 0, 0)
    assert rows[0.54] == (2, 0, 0)
    assert rows[0.56] == (1, 0, 1)  # the 0.55 fit is missed


def test_the_bar_range_is_limited_to_the_scores_plus_one_step(tmp_path):
    cfg = make_cfg(tmp_path)
    db = Db(cfg)
    for key, sim, value in [("github:f", 0.47, "fits"), ("github:n", 0.43, "no")]:
        db.item(key)
        db.vector(key, sim)
        db.label(key, "Alpha", value)
    bars = [bar for bar, *_ in bar_rows(replay(cfg, AxisOllama()).topics[0])]
    assert bars == [0.40, 0.42, 0.44, 0.46, 0.48, 0.50]  # 0.40 and 0.50 are the extra steps; 0.40 is current


def test_the_current_bar_is_marked_and_added_when_off_the_grid(tmp_path):
    cfg = make_cfg(tmp_path, bars=(0.43, None))
    db = Db(cfg)
    for key, sim, value in [("github:f", 0.47, "fits"), ("github:n", 0.43, "no")]:
        db.item(key)
        db.vector(key, sim)
        db.label(key, "Alpha", value)
    result = replay(cfg, AxisOllama())
    lines = format_replay(result)
    assert "Alpha: 1 fits, 1 no" in lines
    assert any(line.startswith("0.43") and line.endswith("< current") for line in lines)
    assert sum(line.endswith("< current") for line in lines) == 1


def test_the_current_bar_has_a_row_outside_the_range_of_the_scores(tmp_path):
    cfg = make_cfg(tmp_path, bars=(0.34, None), min_score=0.66)  # Alpha 0.34, Beta 0.66 from settings
    db = Db(cfg)
    for key, sim, value, topic, axis in [
        ("github:f", 0.47, "fits", "Alpha", 0),
        ("github:n", 0.43, "no", "Alpha", 0),
        ("github:bf", 0.47, "fits", "Beta", 1),
        ("github:bn", 0.43, "no", "Beta", 1),
    ]:
        db.item(key)
        db.vector(key, sim, axis)
        db.label(key, topic, value)
    alpha, beta, _ = replay(cfg, AxisOllama()).topics
    assert bar_rows(alpha)[0] == (0.34, 1, 1, 0)  # below the rows shown: everything is at or above it
    assert [bar for bar, *_ in bar_rows(alpha)][1:] == [0.40, 0.42, 0.44, 0.46, 0.48, 0.50]
    assert bar_rows(beta)[-1] == (0.66, 0, 0, 1)  # above the grid: the fit is missed
    lines = format_replay(replay(cfg, AxisOllama()))
    assert "0.34      1      1       0   < current" in lines
    assert "0.66      0      0       1   < current" in lines


def test_a_bar_with_three_decimals_is_counted_at_its_exact_value(tmp_path):
    cfg = make_cfg(tmp_path, bars=(0.435, None))
    db = Db(cfg)
    for key, sim, value in [("github:f", 0.47, "fits"), ("github:n", 0.432, "no")]:
        db.item(key)
        db.vector(key, sim)
        db.label(key, "Alpha", value)
    lines = format_replay(replay(cfg, AxisOllama()))
    assert "0.435     1      0       0   < current" in lines  # the 0.432 no is below 0.435, not below 0.43
    assert "0.42      1      1       0" in lines


def test_keyword_hits_add_to_the_score(tmp_path):
    cfg = make_cfg(tmp_path)
    db = Db(cfg)
    db.item("github:kw", title="alphakey tool")  # one hit adds 0.03
    db.vector("github:kw", 0.50)
    db.label("github:kw", "Alpha", "fits")
    row = replay(cfg, AxisOllama()).topics[0]
    assert row.fits == [pytest.approx(0.53, abs=1e-4)]


def test_a_vote_counts_as_a_label_for_the_shown_topic_when_there_is_none(tmp_path):
    cfg = make_cfg(tmp_path)
    db = Db(cfg)
    for key, topic, value, vote in [
        ("github:up", "Alpha", None, 1),
        ("github:down", "Alpha", None, -1),
        ("github:both", "Alpha", "no", 1),  # a label for the shown topic wins over the vote
        ("github:other", "Alpha", "fits", -1),  # a label for another topic does not stop the vote
    ]:
        db.item(key, topic=topic)
        db.vector(key, 0.7)
        db.vote(key, vote)
        if value:
            db.label(key, "Alpha" if key != "github:other" else "Beta", value)
    row = replay(cfg, AxisOllama()).topics[0]
    assert (len(row.fits), len(row.no)) == (1, 3)  # up is fits; down, both and other are no


def test_a_lure_label_counts_as_no_for_every_topic(tmp_path):
    cfg = make_cfg(tmp_path)
    db = Db(cfg)
    db.item("github:lure")
    db.vector("github:lure", 0.7)
    db.label("github:lure", "*", "lure")
    db.item("github:own")
    db.vector("github:own", 0.7)
    db.label("github:own", "*", "lure")
    db.label("github:own", "Alpha", "fits")  # the lure label wins: it is the later answer
    db.item("github:liked", topic="Alpha")
    db.vector("github:liked", 0.7)
    db.label("github:liked", "*", "lure")
    db.vote("github:liked", 1)  # and it wins over a vote
    result = replay(cfg, AxisOllama())
    for row in result.topics:
        assert (len(row.fits), len(row.no)) == (0, 3)


def test_labels_for_topics_that_are_gone_are_counted_apart(tmp_path):
    cfg = make_cfg(tmp_path)
    db = Db(cfg)
    db.item("github:x")
    db.vector("github:x", 0.7)
    db.label("github:x", "Removed topic", "fits")
    db.item("github:y", topic="Other removed topic")
    db.vote("github:y", 1)
    result = replay(cfg, AxisOllama())
    assert result.gone == 2
    assert result.labelled == 0
    assert "2 labels for topics no longer in interests.yaml." in format_replay(result)


def test_items_without_a_vector_are_not_scored(tmp_path):
    cfg = make_cfg(tmp_path)
    db = Db(cfg)
    alpha_set(db)
    db.conn.execute("DELETE FROM vector WHERE key = 'github:n1'")
    db.conn.execute(
        "INSERT INTO vector VALUES ('github:f1', 'other-model', ?)", (candidates._pack([1.0, 0.0, 0.0]),)
    )  # a vector of another model does not count either
    db.conn.execute("DELETE FROM vector WHERE key = 'github:f1' AND model = ?", (MODEL,))
    result = replay(cfg, AxisOllama())
    row = result.topics[0]
    assert (len(row.fits), len(row.no), row.not_scored) == (1, 1, 2)
    assert "Alpha: 1 fits, 1 no (2 not scored)" in format_replay(result)


def test_without_ollama_replay_uses_score_rows_of_the_current_topic_text(tmp_path):
    cfg = make_cfg(tmp_path)
    db = Db(cfg)
    mark = labelling.topic_fingerprint(cfg.topics[0])
    for key, value, sim, hits, row_mark, model in [
        ("github:ok-fit", "fits", 0.50, 1, mark, MODEL),  # 0.50 + 0.03
        ("github:ok-no", "no", 0.30, 0, mark, MODEL),
        ("github:old-text", "fits", 0.9, 0, "an older mark", MODEL),  # topic text changed since
        ("github:old-model", "fits", 0.9, 0, mark, "older-model"),
        ("github:never-scored", "no", None, 0, mark, MODEL),
    ]:
        db.item(key)
        db.label(key, "Alpha", value)
        if sim is not None:
            db.score(key, "Alpha", sim, hits, row_mark, model)
    for ollama in (None, AxisOllama(up=False)):
        result = replay(cfg, ollama)
        assert result.mode == "stored scores"
        assert result.left_out == 3
        alpha = result.topics[0]
        assert alpha.fits == [pytest.approx(0.53)] and alpha.no == [pytest.approx(0.30)]
        lines = format_replay(result)
        assert lines[0].startswith("Ollama is not ready")
        assert "3 labels left out." in lines[0]


def test_trust_check_counts_lures_and_others_from_the_first_observation(tmp_path):
    cfg = make_cfg(tmp_path)
    db = Db(cfg)
    for key, kind, observation in [
        ("github:lure-caught", "lure", (4, 50, 0)),
        ("github:lure-late", "lure", (4, 50, 9)),  # followers above 2: not caught
        ("github:lure-unknown", "lure", (4, 50, None)),  # thin by size and stars, owner never asked
        ("github:lure-big", "lure", (900, 50, 0)),  # too big for the rule
        ("github:lure-unseen", "lure", None),
        ("github:fit-caught", "fits", (133, 12, 1)),
        ("github:fit-safe", "fits", (5000, 400, 80)),
        ("github:no-small-few-stars", "no", (10, 3, 0)),  # under 10 stars: held when the owner has 0 followers
        ("github:no-small-few-stars-followed", "no", (10, 3, 1)),  # under 10 stars and 1 follower: not caught
    ]:
        db.item(key)
        db.label(key, "*" if kind == "lure" else "Alpha", kind)
        if observation:
            db.observe(key, *observation)
    db.observe("github:lure-caught", 4000, 900, 500, at="2026-11-05T00:00:00+00:00")  # later rows are ignored
    db.item("arxiv:2611.00001", kind="paper")
    db.label("arxiv:2611.00001", "Alpha", "fits")  # not a repository
    trust = replay(cfg, AxisOllama()).trust
    assert trust.lures == {"caught": 1, "not caught": 2, "unknown": 1, "no observation": 1}
    assert trust.others == {"caught": 2, "not caught": 2, "unknown": 0, "no observation": 0}
    assert trust.minus_only == 0
    lines = format_replay(replay(cfg, AxisOllama()))
    assert any(line.startswith("Trust check on labelled GitHub repositories") for line in lines)


def test_trust_check_counts_a_plus_vote_as_other_and_leaves_a_lone_minus_vote_out(tmp_path):
    cfg = make_cfg(tmp_path)
    db = Db(cfg)
    for key, vote, label, observation in [
        ("github:liked", 1, None, (5000, 400, 80)),  # a + vote says it is not a lure
        ("github:disliked", -1, None, (4, 50, 0)),  # a - vote does not say which: left out
        ("github:disliked-too", -1, None, (900, 50, 0)),
        ("github:disliked-judged", -1, ("Alpha", "no"), (4, 50, 0)),  # asked in `radar label`: an other
        ("github:disliked-lure", -1, ("*", "lure"), (4, 50, 0)),
        ("arxiv:2611.00002", -1, None, None),  # not a repository
    ]:
        db.item(key, topic="Alpha")
        db.vote(key, vote)
        if label:
            db.label(key, *label)
        if observation:
            db.observe(key, *observation)
    trust = replay(cfg, AxisOllama()).trust
    assert trust.lures == {"caught": 1, "not caught": 0, "unknown": 0, "no observation": 0}
    assert trust.others == {"caught": 1, "not caught": 1, "unknown": 0, "no observation": 0}
    assert trust.minus_only == 2
    lines = format_replay(replay(cfg, AxisOllama()))
    assert lines[-1] == "2 repositories with only a - vote are left out: a - vote does not say whether it is a lure. `radar label` asks about them."


def test_replay_reads_a_schema_1_store_without_writing(tmp_path):
    cfg = make_cfg(tmp_path)
    db = Db(cfg)
    alpha_set(db)
    db.conn.executescript("DROP TABLE label; DROP TABLE vote;")
    db.conn.execute("PRAGMA user_version = 1")
    before = db.path.read_bytes()
    result = replay(cfg, AxisOllama())
    assert result.labelled == 0
    assert db.path.read_bytes() == before  # replay never upgrades the file


def test_replay_says_so_without_labels_or_a_store(tmp_path):
    cfg = make_cfg(tmp_path)
    shown = []
    assert run_replay(cfg, AxisOllama(), show=shown.append) == 1
    assert shown == ["No candidate store yet. Run a scan first."]
    Db(cfg)
    shown.clear()
    assert run_replay(cfg, AxisOllama(), show=shown.append) == 0
    assert shown == ["No labels yet. Run `radar label`."]


def test_replay_prints_the_table(tmp_path):
    cfg = make_cfg(tmp_path)
    db = Db(cfg)
    alpha_set(db)
    shown = []
    assert run_replay(cfg, AxisOllama(), show=shown.append) == 0
    assert "Alpha: 2 fits, 2 no" in shown
    assert " bar   kept  noise  missed" in shown
    assert "0.40      2      1       0   < current" in shown
    assert "Beta: 0 fits, 0 no" in shown
