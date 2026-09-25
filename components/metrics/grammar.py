"""Family B guardrail instruments: LanguageTool and a CoLA acceptability model.

Both are OPTIONAL external instruments behind small clients.  They see text
after ``normalize_player_ids`` (P3 -> Jordan, s12 -> statement 12), and they
report their own versions so that the frozen config can pin them (PLAN 8).

LanguageTool is called over its HTTP API (``/v2/check``), so any locally run
server version works: ``java -cp languagetool-server.jar org.languagetool.server.HTTPServer --port 8081``.
Grammar outcome (PLAN 8.2): ``grammar_errors_per_100_words`` counts only
matches whose LanguageTool category is GRAMMAR (or issue type ``grammar``);
spelling, punctuation and style are reported separately.

CoLA (PLAN 8.3): any Hugging Face sequence-classification checkpoint fine-tuned
on CoLA. The default is ``textattack/roberta-base-CoLA`` (public evaluation on
the model card); the choice is frozen at the end of the pilot together with the
revision hash the client records.
"""

from __future__ import annotations

import statistics
import hashlib
from pathlib import Path
from typing import List, Optional

import requests

from components.metrics.text_normalization import normalize_player_ids, word_count

LT_BUCKETS = ("grammar", "usage", "spelling", "punctuation", "style", "other")

# Confirmatory outcome = the GRAMMAR bucket only (PLAN 8.2). Word-choice and
# casing rules (CONFUSED_WORDS, COLLOCATIONS, SEMANTICS, CASING) are reported
# as ``usage`` so that lexical style differences cannot leak into the guardrail.
_USAGE_CATEGORIES = ("CASING", "CONFUSED_WORDS", "COLLOCATIONS", "SEMANTICS", "COMPOUNDING")
_STYLE_CATEGORIES = ("STYLE", "REDUNDANCY", "TYPOGRAPHY", "PLAIN_ENGLISH", "WIKIPEDIA", "MISC")
_STYLE_ISSUES = ("style", "register", "locale-violation", "whitespace", "redundancy",
                 "inconsistency", "non-conformance")


# Rules LanguageTool files under MISC with issueType "misspelling" although they
# are agreement/grammar phenomena. Extend after inspecting the pilot's matches;
# every match keeps its raw rule id, category and issue type so it can be re-bucketed.
GRAMMAR_RULE_IDS: frozenset = frozenset({"EN_A_VS_AN"})


def categorize_match(match: dict) -> str:
    rule = match.get("rule") or {}
    rule_id = str(rule.get("id") or "").upper()
    category = str((rule.get("category") or {}).get("id") or "").upper()
    issue = str(rule.get("issueType") or "").lower()
    # Explicit rule overrides, then category, then issue type: LanguageTool files
    # casing rules under CASING/"typographical" (not punctuation) and a-vs-an
    # under MISC/"misspelling" (not spelling).
    if rule_id in GRAMMAR_RULE_IDS or category == "GRAMMAR":
        return "grammar"
    if category in ("TYPOS", "MISSPELLING"):
        return "spelling"
    if category in _USAGE_CATEGORIES:            # LT tags many word-choice rules "misspelling"
        return "usage"
    if issue == "misspelling":
        return "spelling"
    if category == "PUNCTUATION":
        return "punctuation"
    if category in _STYLE_CATEGORIES or issue in _STYLE_ISSUES:
        return "style"
    if issue == "grammar":
        return "grammar"
    if issue == "typographical":
        return "punctuation"
    return "other"


class LanguageToolClient:
    """Thin client for a running LanguageTool HTTP server."""

    def __init__(self, base_url: str = "http://localhost:8081", language: str = "en-US",
                 timeout: int = 60, disabled_rules: Optional[List[str]] = None):
        self.base_url = base_url.rstrip("/")
        self.language = language
        self.timeout = timeout
        self.disabled_rules = list(disabled_rules or [])
        self._version: Optional[str] = None

    def check(self, text: str) -> dict:
        data = {"text": text, "language": self.language}
        if self.disabled_rules:
            data["disabledRules"] = ",".join(self.disabled_rules)
        response = requests.post(f"{self.base_url}/v2/check", data=data, timeout=self.timeout)
        response.raise_for_status()
        payload = response.json()
        software = payload.get("software") or {}
        if software.get("version"):
            self._version = f"{software.get('name', 'LanguageTool')} {software['version']}"
        return payload

    @property
    def version(self) -> Optional[str]:
        return self._version

    def summarize(self, text: str, *, normalize_ids: bool = True) -> dict:
        """Counts per bucket and the confirmatory grammar density."""
        instrument_text = normalize_player_ids(text) if normalize_ids else text
        payload = self.check(instrument_text)
        matches = payload.get("matches") or []
        counts = {bucket: 0 for bucket in LT_BUCKETS}
        details = []
        for match in matches:
            bucket = categorize_match(match)
            counts[bucket] += 1
            rule = match.get("rule") or {}
            details.append({"bucket": bucket, "rule_id": rule.get("id"),
                            "category": (rule.get("category") or {}).get("id"),
                            "issue_type": rule.get("issueType"),
                            "message": (match.get("message") or "")[:160],
                            "offset": match.get("offset"), "length": match.get("length")})
        n_words = word_count(instrument_text)
        return {
            "instrument": "languagetool", "version": self.version, "language": self.language,
            "n_words": n_words, "normalized_ids": normalize_ids,
            "n_matches": len(matches), **{f"lt_{k}": v for k, v in counts.items()},
            "grammar_errors_per_100_words": (100.0 * counts["grammar"] / n_words) if n_words else None,
            "usage_errors_per_100_words": (100.0 * counts["usage"] / n_words) if n_words else None,
            "all_errors_per_100_words": (100.0 * (counts["grammar"] + counts["usage"] + counts["spelling"]
                                                   + counts["punctuation"]) / n_words) if n_words else None,
            "matches": details,
        }

    def provenance(self) -> dict:
        return {"instrument": "languagetool", "base_url": self.base_url,
                "language": self.language, "version": self.version,
                "disabled_rules": list(self.disabled_rules)}


class ColaScorer:
    """Sentence acceptability probabilities from a CoLA-fine-tuned classifier."""

    def __init__(self, model_name: str = "textattack/roberta-base-CoLA",
                 revision: Optional[str] = None, acceptable_label_index: int = 1,
                 device: Optional[str] = None, batch_size: int = 16, max_length: int = 512):
        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise ImportError("CoLA scoring needs torch and transformers: "
                              "pip install torch transformers") from exc
        self.model_name = model_name
        self.revision = revision
        self.acceptable_label_index = int(acceptable_label_index)
        self.batch_size = int(batch_size)
        self.max_length = int(max_length)
        self._torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, revision=revision)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, revision=revision)
        self.model.eval()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.id2label = {int(k): v for k, v in (self.model.config.id2label or {}).items()}
        self.commit_hash = getattr(self.model.config, "_commit_hash", None)
        if Path(model_name).is_dir():
            hasher = hashlib.sha256()
            files = sorted(p for p in Path(model_name).iterdir()
                           if p.is_file() and (p.suffix in (".json", ".txt", ".model", ".safetensors", ".bin")))
            for path in files:
                hasher.update(path.name.encode())
                with path.open("rb") as stream:
                    for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                        hasher.update(chunk)
            self.commit_hash = "sha256:" + hasher.hexdigest()
        if self.acceptable_label_index not in self.id2label:
            raise ValueError("acceptable label index is outside the checkpoint labels")

    def score_sentences(self, sentences: List[str]) -> List[float]:
        torch = self._torch
        probs: List[float] = []
        with torch.no_grad():
            for start in range(0, len(sentences), self.batch_size):
                batch = sentences[start:start + self.batch_size]
                encoded = self.tokenizer(batch, return_tensors="pt", padding=True,
                                         truncation=False)
                if encoded["input_ids"].shape[1] > self.max_length:
                    raise ValueError(f"CoLA sentence exceeds frozen limit {self.max_length}; no text was truncated")
                encoded = encoded.to(self.device)
                logits = self.model(**encoded).logits
                soft = torch.softmax(logits, dim=-1)[:, self.acceptable_label_index]
                probs += [float(p) for p in soft.cpu()]
        return probs

    def summarize(self, sentences: List[str], *, normalize_ids: bool = True) -> dict:
        """Mean / median / 10th percentile of P(acceptable) over sentences (PLAN 8.3)."""
        texts = [normalize_player_ids(s) if normalize_ids else s for s in sentences if s.strip()]
        if not texts:
            return {"instrument": "cola", "n_sentences": 0, "cola_mean": None,
                    "cola_median": None, "cola_p10": None, "cola_min": None}
        probs = self.score_sentences(texts)
        ordered = sorted(probs)
        p10_index = max(0, min(len(ordered) - 1, int(round(0.10 * (len(ordered) - 1)))))
        return {"instrument": "cola", "model": self.model_name, "n_sentences": len(texts),
                "normalized_ids": normalize_ids,
                "cola_mean": statistics.fmean(probs), "cola_median": statistics.median(probs),
                "cola_p10": ordered[p10_index], "cola_min": ordered[0],
                "cola_per_sentence": probs}

    def provenance(self) -> dict:
        try:
            import torch
            import transformers
            versions = {"torch": torch.__version__, "transformers": transformers.__version__}
        except ImportError:  # pragma: no cover
            versions = {}
        return {"instrument": "cola", "model": self.model_name, "revision": self.revision,
                "commit_hash": self.commit_hash, "id2label": self.id2label,
                "acceptable_label_index": self.acceptable_label_index,
                "max_length": self.max_length, "truncation": False, **versions}
