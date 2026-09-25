"""Durable row checkpoints for interrupted, single-writer validation runs."""
from __future__ import annotations

import fcntl
import json
import os
from pathlib import Path

from components.helpers import read_jsonl
from .common import dump_json
from .protocol import digest


class Checkpoint:
    def __init__(self, output, settings, *, resume=False):
        self.output = Path(output)
        self.output.parent.mkdir(parents=True, exist_ok=True)
        self.partial = self.output.with_name(self.output.name + ".partial")
        self.manifest = self.output.with_name(self.output.name + ".checkpoint.json")
        self.lock = self.output.with_name(self.output.name + ".lock").open("a")
        try:
            fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            self.lock.close()
            raise ValueError("another job owns this output; use a different output path")
        try:
            # JSON round-trip once, so tuples/Paths compare equal to what the manifest stores.
            settings = json.loads(json.dumps(settings, sort_keys=True, default=str, allow_nan=False))
            self.settings = {"settings_sha256": digest(settings), "settings": settings}
            exists = self.output.exists() or self.partial.exists() or self.manifest.exists()
            if exists and not resume:
                raise ValueError("output/checkpoint already exists; use --resume with unchanged settings")
            if exists:
                if not self.manifest.exists() or json.loads(self.manifest.read_text()) != self.settings:
                    raise ValueError("checkpoint inputs, code or model settings differ; refusing to resume")
            else:
                dump_json(self.manifest, self.settings)
            self.finished = self.output.exists()
            path = self.output if self.finished else self.partial
            # Malformed partial rows fail visibly; they are never silently discarded.
            self.rows = read_jsonl(path) if path.exists() else []
        except Exception:
            self.lock.close()
            raise

    def append(self, row):
        if self.finished:
            raise ValueError("completed output is missing expected rows; refusing to extend it")
        with self.partial.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        self.rows.append(row)

    def finish(self):
        if not self.finished:
            if not self.partial.exists():
                self.partial.touch()
            os.replace(self.partial, self.output)
            self.finished = True

    def close(self):
        self.lock.close()
