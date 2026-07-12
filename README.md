# linkedin-post-analysis

How much of LinkedIn is actually written by AI?

This project scrapes a batch of LinkedIn posts and then guesses, for each one,
whether it was written by a human or an AI. It runs two very different
detectors and compares them, so you're not trusting a single black box:

- a **statistical** check that looks at writing style (word variety, sentence
  rhythm, punctuation, stock "LinkedIn AI" phrases) — no model, no GPU.
- a **fine-tuned model** from Hugging Face that was trained to spot AI text.

Each post gets a verdict (`Human`, `Mixed`, or `AI`) from both detectors, and
`summarize.py` turns the results into a handful of charts.

## Setup

You'll need Python 3.10+.

```powershell
cd linkedin-post-analysis
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Then open `config.py` and add your Apify token (used for scraping). You can
grab one from https://console.apify.com/account/integrations. While you're
there, tweak `SEARCH_KEYWORDS` and `TOTAL_POSTS` if you want.

## Running it

```powershell
python scrape.py       # 1. pull posts into data/posts.csv
python analyze.py      # 2. score every post with both detectors
python summarize.py    # 3. build the comparison charts in data/charts/
```

That's it. The charts and CSVs land in the `data/` folder.
