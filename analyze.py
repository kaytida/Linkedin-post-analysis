"""Score each scraped LinkedIn post for AI vs human authorship.

Detectors share a common `Scorer` interface and come in three types:

    statistical   - style metrics only, no model (lexical diversity,
                    burstiness, punctuation, repetition, phrase density).
    hf_classifier - a fine-tuned Hugging Face text classifier (RoBERTa/BERT).
    zero_shot_lm  - a base causal LM scored GLTR-style (top-k rank + burstiness).

Per post, per detector, we record:
    p_ai_full - probability the whole post is AI-written.
    pct_ai    - character-weighted % of sentences flagged as AI.
    verdict   - "AI" / "Human" / "Mixed", derived from pct_ai.

Output: one CSV per detector (data/analysis_<name>.csv) plus data/summary.csv.

Usage:
    python analyze.py                              # all posts, all detectors
    python analyze.py --limit 50                   # quick run on first 50
    python analyze.py --detector fakespot_roberta  # one detector only
"""
from __future__ import annotations

import argparse
import math
import re
import sys
from collections import Counter

import pandas as pd

import config


def _cuda_available() -> bool:
    try:
        import torch
        return bool(torch.cuda.is_available())
    except Exception:
        return False


def _clip01(x: float) -> float:
    return max(0.0, min(1.0, x))


_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(])|\n+")


def split_sentences(text: str) -> list[str]:
    """Split into sentences, treating newlines as breaks too (LinkedIn posts
    use them like punctuation). Tiny fragments get merged into the previous."""
    parts = [p.strip() for p in _SENT_SPLIT.split(text or "") if p and p.strip()]
    merged: list[str] = []
    for p in parts:
        if merged and len(p.split()) < config.MIN_SENTENCE_WORDS:
            merged[-1] = f"{merged[-1]} {p}"
        else:
            merged.append(p)
    return merged


class Scorer:
    """Base class. Subclasses implement score(texts) -> list[float], returning
    P(AI) in [0, 1] for each input string."""

    name: str = "scorer"

    def score(self, texts: list[str]) -> list[float]:  # pragma: no cover
        raise NotImplementedError


_WORD_RE = re.compile(r"[A-Za-z0-9']+")
_PUNCT_RE = re.compile(r"[.,;:!?\"'()\[\]{}—–-]")

# Stock LLM / LinkedIn-AI phrasing. How often these show up is a strong tell.
_AI_PHRASES = (
    "leverage", "delve", "landscape", "unlock", "transformative",
    "cutting-edge", "cutting edge", "game-changer", "game changer",
    "seamless", "harness", "empower", "paradigm", "synergy", "synergies",
    "in today's", "it's not just", "heres the thing", "here's the thing",
    "lets dive", "let's dive", "navigate", "foster", "streamline",
    "elevate", "robust", "crucial", "unprecedented", "holistic",
    "actionable insights", "deep dive", "at the end of the day",
    "excited to share", "i'm thrilled", "im thrilled", "game changing",
    "in this ever-evolving", "ever-evolving", "revolutionize",
    "proud to announce", "i'm excited to", "im excited to",
    "here are key", "key takeaways", "food for thought",
)


class StatisticalScorer(Scorer):
    """Style-based AI scorer. No model weights, no GPU.

    Each feature is mapped to P(AI) via calibration anchors, then averaged:

      herdan_c          - lexical diversity, log(types)/log(tokens). Unlike raw
                          TTR, this stays stable on short posts.
      hapax_ratio       - share of words that appear exactly once.
      sent_len_cv       - variation in sentence length (humans are burstier).
      avg_sent_len      - mean words per sentence (AI posts run long).
      punct_ratio       - punctuation chars / total chars.
      bigram_repeat     - fraction of word-bigrams seen more than once.
      ai_phrase_density - stock LLM/LinkedIn-AI phrases per 100 words.

    Anchors live in cfg["calibration"] as *_human / *_ai endpoints; the
    direction of each feature is inferred from them.
    """

    FEATURES = (
        "herdan_c",
        "hapax_ratio",
        "sent_len_cv",
        "avg_sent_len",
        "punct_ratio",
        "bigram_repeat",
        "ai_phrase_density",
    )

    def __init__(self, cfg: dict):
        self.name = "statistical[stylometry]"
        self.cal = dict(cfg.get("calibration", {}))
        weights = cfg.get("feature_weights")
        if weights:
            self.weights = {k: float(weights.get(k, 0.0)) for k in self.FEATURES}
        else:
            n = sum(1 for f in self.FEATURES
                    if f"{f}_human" in self.cal and f"{f}_ai" in self.cal)
            w = 1.0 / max(1, n)
            self.weights = {f: w for f in self.FEATURES}
        # Optional post-average stretch. LinkedIn feature means sit in a narrow
        # mid band, so without this most posts pile up around 30% and get
        # labelled "Mixed".
        stretch = cfg.get("score_stretch") or {}
        self.stretch_lo = float(stretch["lo"]) if "lo" in stretch else None
        self.stretch_hi = float(stretch["hi"]) if "hi" in stretch else None
        print("  loading statistical stylometry scorer (no model) ...", flush=True)

    @staticmethod
    def _tokens(text: str) -> list[str]:
        return [w.lower() for w in _WORD_RE.findall(text or "")]

    @staticmethod
    def _cv(values: list[float]) -> float:
        if len(values) < 2:
            return 0.0
        mean = sum(values) / len(values)
        if mean <= 1e-8:
            return 0.0
        var = sum((v - mean) ** 2 for v in values) / len(values)
        return math.sqrt(var) / mean

    @staticmethod
    def _ai_phrase_density(text: str, n_words: int) -> float:
        low = (text or "").lower()
        hits = sum(low.count(p) for p in _AI_PHRASES)
        return 100.0 * hits / max(1, n_words)

    def _features(self, text: str) -> dict[str, float] | None:
        text = (text or "").strip()
        words = self._tokens(text)
        if len(words) < 5:
            return None

        counts = Counter(words)
        n = len(words)
        types = len(counts)
        hapax = sum(1 for c in counts.values() if c == 1)

        # Herdan's C = log(types) / log(tokens).
        herdan_c = (math.log(types) / math.log(n)) if n > 1 and types > 1 else 1.0

        sents = split_sentences(text) or [text]
        sent_lens = [float(len(self._tokens(s))) for s in sents]
        avg_sent_len = sum(sent_lens) / max(1, len(sent_lens))

        punct = len(_PUNCT_RE.findall(text))
        chars = max(1, len(text))

        bigrams = list(zip(words, words[1:]))
        if bigrams:
            bg_counts = Counter(bigrams)
            bigram_repeat = sum(1 for c in bg_counts.values() if c > 1) / len(bg_counts)
        else:
            bigram_repeat = 0.0

        return {
            "herdan_c": herdan_c,
            "hapax_ratio": hapax / n,
            "sent_len_cv": self._cv(sent_lens),
            "avg_sent_len": avg_sent_len,
            "punct_ratio": punct / chars,
            "bigram_repeat": bigram_repeat,
            "ai_phrase_density": self._ai_phrase_density(text, n),
        }

    def _feature_to_p(self, name: str, value: float) -> float | None:
        c = self.cal
        h_key, a_key = f"{name}_human", f"{name}_ai"
        if h_key not in c or a_key not in c:
            return None
        human, ai = float(c[h_key]), float(c[a_key])
        denom = ai - human
        if abs(denom) < 1e-8:
            return None
        return _clip01((value - human) / denom)

    def _stretch(self, raw: float) -> float:
        if self.stretch_lo is None or self.stretch_hi is None:
            return raw
        span = self.stretch_hi - self.stretch_lo
        if abs(span) < 1e-8:
            return raw
        return _clip01((raw - self.stretch_lo) / span)

    def _prob_ai(self, feats: dict[str, float]) -> float:
        num, den = 0.0, 0.0
        for name, value in feats.items():
            w = self.weights.get(name, 0.0)
            if w <= 0:
                continue
            p = self._feature_to_p(name, value)
            if p is None:
                continue
            num += w * p
            den += w
        if den <= 0:
            return 0.0
        return self._stretch(_clip01(num / den))

    def score(self, texts: list[str]) -> list[float]:
        out: list[float] = []
        for t in texts:
            feats = self._features(t)
            if feats is None:
                out.append(0.0)
            else:
                out.append(self._prob_ai(feats))
        return out


def _configure_hf_download() -> None:
    """Apply Hugging Face download settings before loading any model.

    Corporate proxies that MITM HTTPS with a self-signed cert break HF
    downloads. Patching ssl alone isn't enough — huggingface_hub builds its
    own requests Session, so we also override its HTTP backend.
    """
    import os

    if getattr(config, "HF_HUB_DISABLE_XET", True):
        os.environ["HF_HUB_DISABLE_XET"] = "1"
    if getattr(config, "HF_SSL_VERIFY", True):
        return

    import ssl
    ssl._create_default_https_context = ssl._create_unverified_context
    try:
        import urllib3
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    except Exception:
        pass

    try:
        import requests
        from huggingface_hub import configure_http_backend

        def _insecure_backend() -> "requests.Session":
            session = requests.Session()
            session.verify = False
            return session

        configure_http_backend(backend_factory=_insecure_backend)
        print("  HF download: SSL verify disabled (corporate proxy mode)",
              flush=True)
    except Exception as e:
        print(f"  warning: could not patch HF HTTP backend: {e}", flush=True)


class HFClassifierScorer(Scorer):
    """Wraps a transformers text-classification pipeline. P(AI) is the summed
    probability of every label matching an `ai_label_keywords` substring."""

    def __init__(self, cfg: dict):
        _configure_hf_download()
        from transformers import (
            AutoModelForSequenceClassification,
            AutoTokenizer,
            pipeline,
        )
        model_id = cfg["model_id"]
        self.name = f"hf_classifier[{model_id}]"
        self.ai_kw = [k.lower() for k in cfg["ai_label_keywords"]]

        print(f"  loading HF classifier: {model_id} ...", flush=True)
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        print("    tokenizer ready", flush=True)
        model = AutoModelForSequenceClassification.from_pretrained(model_id)
        print("    weights loaded", flush=True)
        device = 0 if _cuda_available() else -1
        self.clf = pipeline(
            "text-classification",
            model=model,
            tokenizer=tokenizer,
            top_k=None,
            truncation=True,
            max_length=config.MAX_TOKENS,
            device=device,
        )
        print(f"    pipeline ready (device={'cuda:0' if device == 0 else 'cpu'}, "
              f"batch_size={config.BATCH_SIZE})", flush=True)

    def _is_ai_label(self, label: str) -> bool:
        lab = label.lower()
        return any(k in lab for k in self.ai_kw)

    def _probs_from_results(self, results) -> list[float]:
        if isinstance(results, dict):
            results = [results]
        probs: list[float] = []
        for r in results:
            scores = r if isinstance(r, list) else [r]
            probs.append(sum(float(s["score"]) for s in scores
                             if self._is_ai_label(s["label"])))
        return probs

    def score(self, texts: list[str]) -> list[float]:
        if not texts:
            return []
        n = len(texts)
        bs = max(1, int(config.BATCH_SIZE))
        # Small call (a few sentences): run it in one go, skip progress logs.
        if n <= bs:
            return self._probs_from_results(
                self.clf(texts, batch_size=bs))

        out: list[float] = []
        for start in range(0, n, bs):
            chunk = texts[start:start + bs]
            out.extend(self._probs_from_results(
                self.clf(chunk, batch_size=bs)))
            done = min(start + bs, n)
            print(f"      inference {done}/{n}", flush=True)
        return out


class ZeroShotLMScorer(Scorer):
    """Scores text with a base causal LM using two classic GLTR signals:

      top_k_frac - share of tokens whose true id was in the LM's top-K
                   predictions. AI text sits higher on the model's own
                   distribution, so this fraction is elevated.
      burstiness - std / mean of per-token NLL. Humans mix predictable and
                   surprising tokens; AI is more uniform, so this drops.

    Both are mapped linearly to P(AI) using human/AI anchors in
    cfg["calibration"].
    """

    def __init__(self, cfg: dict):
        _configure_hf_download()
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        model_id = cfg["model_id"]
        self.name = f"zero_shot_lm[{model_id}]"
        self.top_k = int(cfg.get("top_k_ai_signal", 10))
        self.cal = dict(cfg["calibration"])

        print(f"  loading base LM: {model_id} ...", flush=True)
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        # GPT-2 tokenizers have no pad token; reuse EOS so batching works.
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(model_id)
        self.model.eval()
        self.device = "cuda" if _cuda_available() else "cpu"
        self.model.to(self.device)
        self._torch = torch

    def _features(self, text: str) -> tuple[float, float] | None:
        """Return (top_k_frac, burstiness) or None if text is too short."""
        torch = self._torch
        enc = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=1024,
        )
        input_ids = enc["input_ids"].to(self.device)
        if input_ids.shape[1] < 2:
            return None
        with torch.no_grad():
            logits = self.model(input_ids).logits  # (1, T, V)
        shift_logits = logits[:, :-1, :]           # predictions for tokens 1..T
        shift_targets = input_ids[:, 1:]           # actual tokens 1..T

        # Per-token NLL (natural log).
        log_probs = torch.log_softmax(shift_logits, dim=-1)
        target_lp = log_probs.gather(-1, shift_targets.unsqueeze(-1)).squeeze(-1)
        nll = (-target_lp[0]).float().cpu()

        # Per-token rank of the true token in the LM's sorted logits.
        target_logits = shift_logits.gather(-1, shift_targets.unsqueeze(-1))
        ranks = (shift_logits > target_logits).sum(dim=-1).squeeze(0).cpu()

        top_k_frac = float((ranks < self.top_k).float().mean().item())
        mean_nll = float(nll.mean().item())
        std_nll = float(nll.std(unbiased=False).item())
        burstiness = std_nll / (mean_nll + 1e-8)
        return top_k_frac, burstiness

    def _prob_ai(self, top_k_frac: float, burstiness: float) -> float:
        c = self.cal
        # Interpolate between the human (0) and AI (1) anchors.
        tk = (top_k_frac - c["top_k_frac_human"]) / max(
            1e-8, c["top_k_frac_ai"] - c["top_k_frac_human"])
        bu = (c["burstiness_human"] - burstiness) / max(
            1e-8, c["burstiness_human"] - c["burstiness_ai"])
        p = c["w_top_k"] * _clip01(tk) + c["w_burstiness"] * _clip01(bu)
        return _clip01(p)

    def score(self, texts: list[str]) -> list[float]:
        out: list[float] = []
        for t in texts:
            if not t or not t.strip():
                out.append(0.0)
                continue
            feats = self._features(t)
            if feats is None:
                out.append(0.0)
                continue
            out.append(self._prob_ai(*feats))
        return out


def make_scorer(cfg: dict) -> Scorer:
    dtype = cfg.get("type", "hf_classifier")
    if dtype == "statistical":
        return StatisticalScorer(cfg)
    if dtype == "hf_classifier":
        return HFClassifierScorer(cfg)
    if dtype == "zero_shot_lm":
        return ZeroShotLMScorer(cfg)
    raise ValueError(
        f"Unknown detector type {dtype!r}. "
        f"Known: statistical, hf_classifier, zero_shot_lm.")


# Verdict thresholds: a post is "AI" if pct_ai >= *_AI_THRESHOLD, "Human" if
# pct_ai <= *_HUMAN_THRESHOLD, else "Mixed". Model classifiers and statistical
# stylometry get separate pairs so they can be tuned independently. A single
# detector can override these via cfg["verdict_thresholds"] in config.py.

# Model classifiers (e.g. fakespot_roberta):
DEFAULT_AI_THRESHOLD = 60.0
DEFAULT_HUMAN_THRESHOLD = 20.0

# Statistical stylometry (no model):
STAT_AI_THRESHOLD = 57.0
STAT_HUMAN_THRESHOLD = 25.0


def default_thresholds(detector_type: str) -> tuple[float, float]:
    """(ai_threshold, human_threshold) for a detector type."""
    if detector_type == "statistical":
        return STAT_AI_THRESHOLD, STAT_HUMAN_THRESHOLD
    return DEFAULT_AI_THRESHOLD, DEFAULT_HUMAN_THRESHOLD


def verdict(pct_ai: float,
            ai_thr: float = DEFAULT_AI_THRESHOLD,
            human_thr: float = DEFAULT_HUMAN_THRESHOLD) -> str:
    if pct_ai >= ai_thr:
        return "AI"
    if pct_ai <= human_thr:
        return "Human"
    return "Mixed"


def analyse_posts(df: pd.DataFrame, detector_name: str,
                  detector_cfg: dict) -> pd.DataFrame:
    print(f"  building scorer for {detector_name} "
          f"({detector_cfg.get('type', 'hf_classifier')}) ...", flush=True)
    scorer = make_scorer(detector_cfg)

    def_ai, def_human = default_thresholds(detector_cfg.get("type", "hf_classifier"))
    thr = detector_cfg.get("verdict_thresholds") or {}
    ai_thr = float(thr.get("ai", def_ai))
    human_thr = float(thr.get("human", def_human))
    print(f"  verdict thresholds: AI>={ai_thr:g}  Human<={human_thr:g}",
          flush=True)

    texts = df["text"].astype(str).tolist()
    n_posts = len(texts)
    print(f"  scoring whole posts ({n_posts}) ...", flush=True)
    # Chunk so long HF runs report progress before the sentence pass starts.
    chunk = max(25, int(config.BATCH_SIZE) * 2)
    p_ai_full: list[float] = []
    for start in range(0, n_posts, chunk):
        batch = texts[start:start + chunk]
        p_ai_full.extend(scorer.score(batch))
        done = min(start + chunk, n_posts)
        print(f"    whole-post {done}/{n_posts}", flush=True)

    print("  scoring per sentence ...", flush=True)
    rows: list[dict] = []
    for i, (_, row) in enumerate(df.iterrows(), 1):
        sents = split_sentences(str(row["text"]))
        # Only score sentences long enough to carry signal. Short fragments
        # (one-liners, hashtags, CTAs) would otherwise count as a confident
        # P(AI)=0 vote and inflate the "Human" share.
        scoreable = [s for s in sents
                     if len(s.split()) >= config.MIN_SENTENCE_WORDS]
        if not scoreable:
            rows.append({"pct_ai": 0.0, "n_sentences": 0, "n_sent_ai": 0})
        else:
            if i == 1 or i % 10 == 0:
                print(f"    post {i}/{n_posts}  "
                      f"({len(scoreable)}/{len(sents)} sentences) ...",
                      flush=True)
            sent_probs = scorer.score(scoreable)
            weights = [max(1, len(s)) for s in scoreable]  # char weights
            # Character-weighted mean of sentence P(AI). A hard 0.5 cutoff
            # collapses when scores sit in a mid band (common for stylometry).
            total_w = sum(weights)
            pct_ai = 100.0 * sum(w * p for w, p in zip(weights, sent_probs)) / total_w
            rows.append({
                "pct_ai": round(pct_ai, 2),
                "n_sentences": len(scoreable),
                "n_sent_ai": sum(1 for p in sent_probs if p >= 0.5),
            })
        if i % 25 == 0 or i == n_posts:
            print(f"    {i}/{n_posts} posts scored", flush=True)

    print(f"  done — {n_posts} posts scored", flush=True)
    out = df[["post_url", "author_name", "keyword", "word_count"]].copy()
    out["detector"] = detector_name
    out["detector_type"] = detector_cfg.get("type", "hf_classifier")
    out["p_ai_full"] = [round(p, 4) for p in p_ai_full]
    out["pct_ai"] = [r["pct_ai"] for r in rows]
    out["n_sentences"] = [r["n_sentences"] for r in rows]
    out["n_sent_ai"] = [r["n_sent_ai"] for r in rows]
    out["verdict"] = [verdict(r["pct_ai"], ai_thr, human_thr) for r in rows]
    return out


def summarise(all_results: pd.DataFrame) -> pd.DataFrame:
    grp = all_results.groupby("detector")
    summary = grp.agg(
        posts=("post_url", "count"),
        detector_type=("detector_type", "first"),
        mean_p_ai_full=("p_ai_full", "mean"),
        mean_pct_ai=("pct_ai", "mean"),
        share_ai=("verdict", lambda s: (s == "AI").mean()),
        share_human=("verdict", lambda s: (s == "Human").mean()),
        share_mixed=("verdict", lambda s: (s == "Mixed").mean()),
    ).round(4).reset_index()
    return summary


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Detect AI-written LinkedIn posts.")
    p.add_argument("--limit", type=int, default=None,
                   help="Only analyse the first N posts (for quick tests).")
    p.add_argument("--detector", default=None,
                   help=f"Run only this detector. One of: {list(config.DETECTORS)}")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    if not config.POSTS_CSV.exists():
        print(f"ERROR: {config.POSTS_CSV} not found. Run scrape.py first.",
              file=sys.stderr)
        return 2

    df = pd.read_csv(config.POSTS_CSV)
    df = df[df["text"].astype(str).str.strip().str.len() > 0].reset_index(drop=True)
    if args.limit:
        df = df.head(args.limit)
    print(f"Analysing {len(df)} posts.")

    detectors = config.DETECTORS
    if args.detector:
        if args.detector not in detectors:
            print(f"ERROR: unknown detector {args.detector!r}. "
                  f"Known: {list(detectors)}", file=sys.stderr)
            return 2
        detectors = {args.detector: detectors[args.detector]}

    all_results: list[pd.DataFrame] = []
    written: list[str] = []
    for name, cfg in detectors.items():
        print(f"\n=== detector: {name} ({cfg.get('type', 'hf_classifier')}) ===")
        try:
            result = analyse_posts(df, name, cfg)
        except Exception as e:
            print(f"!! detector {name} failed: {e}", file=sys.stderr)
            continue

        out_path = config.analysis_csv_path(name)
        result.to_csv(out_path, index=False, encoding="utf-8")
        print(f"  wrote {out_path}", flush=True)
        written.append(str(out_path))
        all_results.append(result)

    if not all_results:
        print("ERROR: no detector produced results.", file=sys.stderr)
        return 1

    summary = summarise(pd.concat(all_results, ignore_index=True))
    summary.to_csv(config.SUMMARY_CSV, index=False, encoding="utf-8")

    print("\nPer-detector results:")
    for path in written:
        print(f"  -> {path}")
    print(f"Summary          -> {config.SUMMARY_CSV}\n")
    print(summary.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
