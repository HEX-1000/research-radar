"""Loading interests.yaml and locating the data home."""

from __future__ import annotations

import copy
import os
import shutil
from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path
from typing import Any

import yaml

DEFAULT_SETTINGS: dict[str, Any] = {
    "ollama_url": "http://localhost:11434",
    "embed_model": "nomic-embed-text",
    "chat_model": "",
    "summary_language": "English",
    "min_score": 0.55,
    "max_items": 8,
    "max_per_topic": 3,
    "first_run_days": 3,
    "max_catchup_days": 14,
    "auto_scan_on_launch": True,
    "github_token_env": "GITHUB_TOKEN",
    "rising_days": 14,
    "feedback_weight": 0.2,
    "notes_dir": "",
}

DEFAULT_SOURCES: dict[str, dict[str, Any]] = {
    "github": {"enabled": True, "min_stars": 5, "per_query": 10},
    "arxiv": {"enabled": True, "max_results": 100},
    "huggingface-papers": {"enabled": True},
    "huggingface-models": {"enabled": True, "limit": 25},
    "forgejo": {"enabled": True, "instances": ["codeberg.org"], "per_query": 10},
}

SOURCE_KEYS = tuple(DEFAULT_SOURCES)

# Secrets never go in this file. The settings name the environment variable that holds them.
DEFAULT_DELIVERY: dict[str, Any] = {
    "html_file": True,  # write digests/YYYY-MM-DD.html for every scan
    "send_when_empty": False,  # send a "nothing new" message too
    "retry_days": 3,  # items that failed to send are retried for this many days
    "email": {
        "enabled": False,
        "smtp_host": "smtp.gmail.com",
        "smtp_port": 587,
        "security": "starttls",  # starttls, ssl or none
        "username": "",
        "password_env": "RADAR_SMTP_PASSWORD",
        "from": "",  # default: the username
        "to": "",  # one address or a list
    },
    "telegram": {
        "enabled": False,
        "token_env": "RADAR_TELEGRAM_TOKEN",
        "chat_id": "",
    },
}


class ConfigError(Exception):
    """interests.yaml is missing or does not make sense."""


@dataclass
class Topic:
    name: str
    description: str
    keywords: list[str] = field(default_factory=list)
    exclude: list[str] = field(default_factory=list)
    options: dict[str, dict[str, Any]] = field(default_factory=dict)  # per source

    def source_option(self, source: str, key: str, default: Any = None) -> Any:
        return self.options.get(source, {}).get(key, default)

    def embed_text(self) -> str:
        parts = [f"{self.name}. {self.description.strip()}"]
        if self.keywords:
            parts.append("Keywords: " + ", ".join(self.keywords))
        return " ".join(parts)


@dataclass
class Config:
    home: Path
    settings: dict[str, Any]
    sources: dict[str, dict[str, Any]]
    exclude: list[str]
    topics: list[Topic]
    delivery: dict[str, Any] = field(default_factory=lambda: copy.deepcopy(DEFAULT_DELIVERY))

    @property
    def digests_dir(self) -> Path:
        return self.home / "digests"

    def channel_enabled(self, channel: str) -> bool:
        return bool(self.delivery.get(channel, {}).get("enabled", False))

    @property
    def notes_dir(self) -> Path:
        custom = str(self.settings.get("notes_dir") or "").strip()
        return Path(custom).expanduser() if custom else self.home / "notes"

    @property
    def state_dir(self) -> Path:
        return self.home / "state"

    def source_enabled(self, name: str) -> bool:
        return bool(self.sources.get(name, {}).get("enabled", False))

    def github_token(self) -> str | None:
        env = str(self.settings.get("github_token_env") or "")
        return os.environ.get(env) or None if env else None


def resolve_home(explicit: str | os.PathLike[str] | None = None) -> Path:
    """The data home: --home, then $RADAR_HOME, then ~/.research-radar."""
    if explicit:
        return Path(explicit).expanduser()
    env = os.environ.get("RADAR_HOME")
    if env:
        return Path(env).expanduser()
    return Path.home() / ".research-radar"


def example_text() -> str:
    return resources.files("research_radar").joinpath("interests.example.yaml").read_text(encoding="utf-8")


def init_home(home: Path) -> Path:
    """Create the data home and copy the example interests file. Never overwrites."""
    home.mkdir(parents=True, exist_ok=True)
    target = home / "interests.yaml"
    if not target.exists():
        target.write_text(example_text(), encoding="utf-8")
    return target


def _as_list(value: Any, where: str) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, list) and all(isinstance(v, (str, int, float)) for v in value):
        return [str(v) for v in value]
    raise ConfigError(f"{where} must be a list of strings")


def parse_config(data: dict[str, Any], home: Path) -> Config:
    if not isinstance(data, dict):
        raise ConfigError("interests.yaml must contain a mapping at the top level")

    settings = copy.deepcopy(DEFAULT_SETTINGS)
    settings.update(data.get("settings") or {})

    sources = copy.deepcopy(DEFAULT_SOURCES)
    for name, opts in (data.get("sources") or {}).items():
        if name not in sources:
            raise ConfigError(f"unknown source '{name}' (known: {', '.join(SOURCE_KEYS)})")
        if isinstance(opts, bool):
            opts = {"enabled": opts}
        sources[name].update(opts or {})

    topics: list[Topic] = []
    raw_topics = data.get("topics") or []
    if not isinstance(raw_topics, list) or not raw_topics:
        raise ConfigError("interests.yaml needs a non-empty 'topics' list")
    seen_names: set[str] = set()
    for i, raw in enumerate(raw_topics, start=1):
        if not isinstance(raw, dict):
            raise ConfigError(f"topic #{i} must be a mapping")
        name = str(raw.get("name") or "").strip()
        description = str(raw.get("description") or "").strip()
        if not name:
            raise ConfigError(f"topic #{i} has no name")
        if not description:
            raise ConfigError(f"topic '{name}' has no description")
        if name in seen_names:
            raise ConfigError(f"topic name '{name}' is used twice")
        seen_names.add(name)
        options = {k: (raw[k] or {}) for k in SOURCE_KEYS if isinstance(raw.get(k), dict)}
        # Short names in the file: github, arxiv, forgejo. Hugging Face sources have
        # no per-topic options, they match by text.
        topics.append(
            Topic(
                name=name,
                description=description,
                keywords=_as_list(raw.get("keywords"), f"topic '{name}' keywords"),
                exclude=_as_list(raw.get("exclude"), f"topic '{name}' exclude"),
                options=options,
            )
        )

    return Config(
        home=home,
        settings=settings,
        sources=sources,
        exclude=_as_list(data.get("exclude"), "exclude"),
        topics=topics,
        delivery=_parse_delivery(data.get("delivery")),
    )


def _parse_delivery(raw: Any) -> dict[str, Any]:
    delivery = copy.deepcopy(DEFAULT_DELIVERY)
    if raw is None:
        return delivery
    if not isinstance(raw, dict):
        raise ConfigError("delivery must be a mapping")
    for key, value in raw.items():
        if key not in delivery:
            raise ConfigError(f"unknown delivery option '{key}' (known: {', '.join(delivery)})")
        if isinstance(delivery[key], dict):
            if isinstance(value, bool):
                value = {"enabled": value}
            if not isinstance(value, dict):
                raise ConfigError(f"delivery.{key} must be a mapping")
            unknown = [k for k in value if k not in delivery[key]]
            if unknown:
                raise ConfigError(f"unknown option '{unknown[0]}' in delivery.{key} (known: {', '.join(delivery[key])})")
            delivery[key].update(value)
        else:
            delivery[key] = value
    return delivery


def load_config(home: Path) -> Config:
    path = home / "interests.yaml"
    if not path.exists():
        raise ConfigError(f"no interests file at {path}. Run `radar init` first.")
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigError(f"{path} is not valid YAML: {exc}") from exc
    return parse_config(data or {}, home)


def copy_example_to(path: Path) -> None:
    """Used by tests and docs: write the example file somewhere."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with resources.as_file(resources.files("research_radar").joinpath("interests.example.yaml")) as src:
        shutil.copyfile(src, path)
