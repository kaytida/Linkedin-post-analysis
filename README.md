# linkedin-post-analysis

Scrape ~1,000 LinkedIn posts via the Apify API and then estimate, for each
post, how much of it was likely AI-generated.

The project is deliberately in **two independent stages** so you can rerun
one without touching the other:

| Stage | Script | Output |
|---|---|---|
| 1. Scrape | `scrape.py` | `data/posts.csv` (+ full `data/raw_posts.jsonl`) |
| 2. Analyse | `analyze.py` | `data/analysis_<detector>.csv` (one file per detector) and `data/summary.csv` |

## How the AI/human decision works

Every post is scored by several independent detectors of **different types**
so you can cross-check them. Two primary detector types are built in:

### Type A: statistical stylometry (`statistical`) — no neural model

Pure metric-based scoring. Features extracted from the text itself:

- **TTR / hapax ratio** – lexical diversity (AI prose is often more uniform).
- **Sentence / word length CV** – burstiness; humans vary length more.
- **Punctuation ratio**, **bigram repetition**, **character entropy** –
  surface-style tells common in templated LLM LinkedIn posts.

Each feature is linearly mapped to `P(AI)` using human/AI calibration
anchors in `config.DETECTORS["statistical_stylometry"]`. Fast, no GPU,
no model download.

### Type B: fine-tuned Hugging Face classifiers (`hf_classifier`) — RoBERTa / BERT

Standard sequence-classification models trained to output `Human` vs
`AI/ChatGPT/Fake`.

- `Hello-SimpleAI/chatgpt-detector-roberta` – HC3 ChatGPT detector (`hc3_chatgpt`).

### Optional: zero-shot base-LM scoring (`zero_shot_lm`)

No fine-tuning: run text through a small causal LM (`distilgpt2`) and
compute GLTR-style top-K rank fraction + NLL burstiness. Useful for
catching newer LLMs the classifiers were never trained on.

### Per-post outputs (for every detector)

- **`p_ai_full`** – the detector's probability that the *whole post* is
  AI-generated.
- **`pct_ai`** – the "how much of the post was AI" figure. We split the
  post into sentences, score each one, then take the character-weighted
  share of sentences flagged as AI (probability ≥ 0.5). This gives a sane
  answer for mixed posts where a human wrote a hook and pasted an AI body.
- **`verdict`** – bucket derived from `pct_ai`:
  `AI` if ≥ 70, `Human` if ≤ 30, otherwise `Mixed`.

No detector is perfect – short posts, heavy emoji use, and non-English
text all reduce accuracy – so treat scores as signals, not proofs. The
per-detector `share_ai` numbers in `summary.csv` disagreeing by more than
a few points usually means the posts are borderline.

### Adding your own detector

Edit `config.DETECTORS` and add an entry with a `type`:

```python
"my_roberta": {
    "type": "hf_classifier",
    "model_id": "some-org/some-hf-model",
    "ai_label_keywords": ["fake", "ai", "generated"],
},
"my_stats": {
    "type": "statistical",
    "calibration": {"ttr_human": 0.72, "ttr_ai": 0.55, ...},
},
```

To add a whole new *kind* of detector (e.g. LLM-as-judge), subclass
`Scorer` in `analyze.py`, implement `score(texts) -> list[float]`, and
register it in `make_scorer()`.

## Setup

```powershell
cd linkedin-post-analysis
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# On CPU-only machines, if `torch` install is huge/slow you can instead run:
# pip install torch --index-url https://download.pytorch.org/whl/cpu
```

Then open `config.py` and set:

- `APIFY_API_TOKEN` – grab from <https://console.apify.com/account/integrations>.
- `SEARCH_KEYWORDS`, `TOTAL_POSTS`, `SORT_TYPE` – tweak to taste.

> `config.py` is git-ignored in this project because it now holds your API
> token in plain text. If you clone this repo fresh, restore it from
> version control history or your own backup before running the scripts.

## Usage

### 1. Scrape 1,000 posts

```powershell
python scrape.py
```

or with overrides:

```powershell
python scrape.py --keywords "artificial intelligence,startup,leadership" --total 1000
python scrape.py --actor harvestapi/linkedin-post-search --sort date_posted
```

- The total budget is split evenly across your keywords.
- Every raw item returned by Apify is appended to `data/raw_posts.jsonl`
  **as it arrives**, so a crash mid-run doesn't lose data.
- `data/posts.csv` is the cleaned table with `text`, `post_url`, `author_name`,
  `author_headline`, engagement counts, `keyword`, `word_count`, etc.

### 2. Analyse

```powershell
python analyze.py                        # full run (all detectors)
python analyze.py --limit 50             # smoke test on 50 posts
python analyze.py --detector hc3_chatgpt # single detector
```

Outputs:

- `data/analysis_statistical_stylometry.csv` – statistical detector scores.
- `data/analysis_hc3_chatgpt.csv` – RoBERTa classifier scores.
- `data/summary.csv` – per detector: mean scores and share of posts in
  each verdict bucket. Also printed to the terminal at the end of the run.

## Configuration

Everything lives in `config.py`:

- `APIFY_API_TOKEN` – your Apify token (plain string).
- `APIFY_ACTOR_ID` – defaults to `harvestapi/linkedin-post-search`.
  Input uses `searchQueries` / `maxPosts` / `sortBy`; output normalisation
  handles that actor's nested `author` / `engagement` / `postedAt` fields.
- `SEARCH_KEYWORDS` – list of topics to search.
- `TOTAL_POSTS` – overall target (defaults to 1000).
- `SORT_TYPE` – `"relevance"` or `"date_posted"`.
- `DETECTORS` – add or remove detectors. Each entry needs a `type`
  (`"statistical"`, `"hf_classifier"`, or `"zero_shot_lm"`). See the
  "How the AI/human decision works" section above for the schema.

## Files

```
linkedin-post-analysis/
├── .gitignore
├── README.md
├── config.py        # holds your Apify token + settings (git-ignored)
├── requirements.txt
├── scrape.py
├── analyze.py
└── data/            # created on first run (gitignored)
    ├── raw_posts.jsonl
    ├── posts.csv
    ├── analysis_statistical_stylometry.csv
    ├── analysis_hc3_chatgpt.csv
    └── summary.csv
```

## Caveats

- Respect LinkedIn's ToS and applicable laws. Apify actors handle the
  scraping mechanics but you are responsible for how you use the data.
- The first analysis run downloads ~500MB of model weights into the
  HuggingFace cache. Subsequent runs reuse them.
- Sentence-level `pct_ai` is a best-effort heuristic. On very short posts
  (≤ 2 sentences) it collapses to the whole-post score in practice.
