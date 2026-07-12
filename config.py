"""Central configuration - edit the values below directly.

SECURITY: this file holds your Apify API token in plain text, so it is
git-ignored by default. Don't commit it or share it publicly.
"""
from __future__ import annotations

from pathlib import Path

# --- Paths -----------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

RAW_POSTS_JSONL = DATA_DIR / "raw_posts.jsonl"  # full raw records, nothing lost
POSTS_CSV = DATA_DIR / "posts.csv"              # cleaned posts (scrape output)
# Per-detector results are written as data/analysis_<detector_name>.csv
ANALYSIS_CSV_TEMPLATE = "analysis_{detector}.csv"
SUMMARY_CSV = DATA_DIR / "summary.csv"          # aggregate comparison across detectors
COMPARISON_CSV = DATA_DIR / "comparison.csv"    # per-post cross-detector join
CHARTS_DIR = DATA_DIR / "charts"                # summarize.py figures
CHARTS_DIR.mkdir(exist_ok=True)


def analysis_csv_path(detector_name: str) -> Path:
    """Path for one detector's per-post results."""
    return DATA_DIR / ANALYSIS_CSV_TEMPLATE.format(detector=detector_name)

# --- Apify scraping --------------------------------------------------------
# Get a token from https://console.apify.com/account/integrations
APIFY_API_TOKEN = "APIFY_TOKEN_HERE"

# Apify actor used to scrape LinkedIn posts by keyword.
# https://apify.com/harvestapi/linkedin-post-search
APIFY_ACTOR_ID = "harvestapi/linkedin-post-search"

# Keywords to search. The total post budget is split evenly across them.
SEARCH_KEYWORDS = [
    "artificial intelligence",
    "machine learning",
    "startup",
]

# Total number of posts to scrape across all keywords combined.
TOTAL_POSTS = 1000

# "relevance" or "date_posted"
SORT_TYPE = "relevance"

# --- Detection models ------------------------------------------------------
# Detector types (each entry's "type" picks one):
#
#   "statistical"   - style metrics only, no model or GPU.
#                     Fields: calibration (human/AI anchors per feature),
#                     feature_weights (optional, defaults to equal).
#   "hf_classifier" - a fine-tuned RoBERTa/BERT classifier on Hugging Face.
#                     Field: ai_label_keywords (case-insensitive label
#                     substrings that mean "AI / machine-generated").
#   "zero_shot_lm"  - base causal LM scored GLTR-style (rank + burstiness).
#                     Fields: top_k_ai_signal, calibration.
DETECTORS = {
    # 1) Statistical, no model.
    "statistical_stylometry": {
        "type": "statistical",
        # Anchors tuned to observed LinkedIn ranges (not essays):
        # human endpoint -> P(AI)=0, AI endpoint -> P(AI)=1.
        "calibration": {
            # Herdan's C (log types / log tokens): higher = more diverse/human.
            "herdan_c_human": 0.97,
            "herdan_c_ai": 0.90,
            # Hapax is noisy on short posts, so anchors are wide + low weight.
            "hapax_ratio_human": 0.85,
            "hapax_ratio_ai": 0.40,
            # Burstiness / length.
            "sent_len_cv_human": 0.75,
            "sent_len_cv_ai": 0.20,
            "avg_sent_len_human": 10.0,
            "avg_sent_len_ai": 22.0,
            # LinkedIn posts use less punctuation than essays.
            "punct_ratio_human": 0.040,
            "punct_ratio_ai": 0.015,
            "bigram_repeat_human": 0.00,
            "bigram_repeat_ai": 0.06,
            # Stock LLM / LinkedIn-AI phrases per 100 words.
            "ai_phrase_density_human": 0.0,
            "ai_phrase_density_ai": 1.5,
        },
        "feature_weights": {
            "herdan_c": 0.15,
            "hapax_ratio": 0.05,       # noisy on short text
            "sent_len_cv": 0.20,
            "avg_sent_len": 0.15,
            "punct_ratio": 0.05,       # weak on LinkedIn
            "bigram_repeat": 0.10,
            "ai_phrase_density": 0.30, # strongest surface signal
        },
        # Spread the weighted feature mean toward the AI end:
        # P(AI) = clip((raw - lo) / (hi - lo)). Without this most posts sit in
        # a soft mid band and get labelled "Mixed".
        "score_stretch": {"lo": 0.17, "hi": 0.44},
        # Verdict bands live in analyze.py (STAT_AI_THRESHOLD /
        # STAT_HUMAN_THRESHOLD). To override just this detector, add:
        # "verdict_thresholds": {"ai": .., "human": ..}
    },
    # 2) Model-based classifier (recommended for LinkedIn). Labels: Human / AI.
    "fakespot_roberta": {
        "type": "hf_classifier",
        "model_id": "fakespot-ai/roberta-base-ai-text-detection-v1",
        "ai_label_keywords": ["ai", "fake", "llm", "generated", "machine", "label_1"],
    },
    # Optional / legacy detectors — uncomment to also run them.
    # HC3 ChatGPT essay detector. Barely fires on LinkedIn (median ~0.001).
    # "hc3_chatgpt": {
    #     "type": "hf_classifier",
    #     "model_id": "Hello-SimpleAI/chatgpt-detector-roberta",
    #     "ai_label_keywords": ["chatgpt", "ai", "fake", "llm", "generated", "label_1"],
    # },
    # RAID-trained detector. Strong on news/wiki, over-flags LinkedIn (~0.93).
    # "tmr_raid": {
    #     "type": "hf_classifier",
    #     "model_id": "Oxidane/tmr-ai-text-detector",
    #     "ai_label_keywords": ["label_1", "ai", "fake", "llm", "generated", "machine"],
    # },
}

# Sentences shorter than this are skipped in sentence-level scoring (too little
# signal to classify reliably).
MIN_SENTENCE_WORDS = 5

# Max tokens per inference call. Longer posts are truncated for the whole-post
# score; sentence-level scoring still covers the full text.
MAX_TOKENS = 512

# Batch size for sentence-level inference.
BATCH_SIZE = 16

# --- Hugging Face download (corporate proxy / SSL) -------------------------
# On networks that MITM HTTPS with a self-signed cert, set HF_SSL_VERIFY=False
# to skip verification for HF downloads, and disable the Xet CDN.
HF_SSL_VERIFY = False
HF_HUB_DISABLE_XET = True
