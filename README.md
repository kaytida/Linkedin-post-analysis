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
grab one from [https://console.apify.com/account/integrations](https://console.apify.com/account/integrations). While you're
there, tweak `SEARCH_KEYWORDS` and `TOTAL_POSTS` if you want.

## Running it

```powershell
python scrape.py       # 1. pull posts into data/posts.csv
python analyze.py      # 2. score every post with both detectors
python summarize.py    # 3. build the comparison charts in data/charts/
```

That's it. The charts and CSVs land in the `data/` folder.

## Results

I ran this on **987 LinkedIn posts** scraped across the keywords *artificial
intelligence*, *startup* and *machine learning*. The headline finding: almost
nothing reads as clearly human. Both detectors independently agreed a post was
human-written only **2%** of the time, while the vast majority of posts landed
in a grey zone where at least one detector saw AI fingerprints. Here's what each
chart shows.

### 1. Model agreement

Model agreement on AI vs human

Where the two detectors land relative to each other. Only **2%** of posts were
flagged human by *both*, **12%** were flagged AI by *both*, and a large **43%**
had exactly one detector calling AI. The big "only one AI" and "both mixed"
slices show the detectors rarely give a clean, confident human verdict.

### 2. Verdict breakdown by detector

Verdict breakdown by detector

How each detector votes on its own. The statistical stylometry check is the harsher critic — it labels only 39 posts human and 464 AI. Fakespot RoBERTa is more forgiving, calling 249 human and just 200 AI. Both pile most posts into "mixed", which tells us the two methods have very different sensitivity even when they broadly agree there's AI influence. But, we need to keep in mind that the Fakespot Roberta was trained to catch AI on much larger bodies of text rather than on small Linkedin Posts. This might be one reason it fails to accurately detect as most Linkedin posts on average were within 600 characters.

### 3. Whole-post AI probability, model vs model

Whole-post AI probability: model vs model

Each dot is a post; its position is the AI probability from each detector. If
they agreed, dots would hug the dashed diagonal — instead they scatter widely,
with heavy clustering along the left and right edges (one model very sure, the
other unsure). This is the clearest sign that the two detectors are measuring
different things and shouldn't be trusted individually.

### 4. Distribution of sentence-level AI share

Distribution of sentence-level AI share

How much of each post was scored as AI, sentence by sentence. The stylometry
detector (blue) is shifted right, peaking around 50–65% AI. Fakespot (red) is
flatter and shifted left. Very few posts sit below the 30% "human" line, so
even lenient reads still see meaningful AI content in most posts.

### 5. Consensus by keyword

Consensus by search keyword

The human/AI mix broken out by search term. The pattern is remarkably
consistent across *artificial intelligence*, *startup* and *machine learning* —
"might be AI" dominates every keyword. *Artificial intelligence* posts have the
largest "definitely AI" band, but no topic escapes the trend.

### 6. Verdict cross-tab

Verdict cross-tab (counts)

A count of every combination of the two detectors' verdicts. The darkest cells
are off the diagonal (e.g. 245 posts the stylometry check called AI but Fakespot
called mixed), confirming the disagreement seen in the scatter. The two models
only fully align on 20 human, 278 mixed and 119 AI posts.

**Takeaway:** on this batch of LinkedIn posts, unambiguously human writing is
rare (~~2%), outright AI is a clear minority (~~12%), and the overwhelming
majority sits in an ambiguous middle — which is exactly why running two
independent detectors is more honest than trusting any single score.