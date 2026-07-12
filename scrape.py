"""Scrape LinkedIn posts via Apify and save a clean CSV.

Usage:
    python scrape.py                 # uses .env defaults
    python scrape.py --keywords "ai,startup" --total 1000
    python scrape.py --actor harvestapi/linkedin-post-search

The script:
  1. Runs the configured Apify actor once per keyword, dividing the total
     post budget evenly across keywords.
  2. Streams every returned item to `data/raw_posts.jsonl` immediately, so
     nothing is lost if the run is interrupted.
  3. De-duplicates by post URL, then writes a normalised `data/posts.csv`
     with the fields we care about for downstream analysis.
"""
from __future__ import annotations

import argparse
import json
import sys
from typing import Any, Iterable

import pandas as pd
from apify_client import ApifyClient

import config


# --- CLI -------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Scrape LinkedIn posts via Apify.")
    p.add_argument("--keywords", default=None,
                   help="Comma-separated keywords. Overrides SEARCH_KEYWORDS.")
    p.add_argument("--total", type=int, default=None,
                   help="Total posts to collect. Overrides TOTAL_POSTS.")
    p.add_argument("--actor", default=None,
                   help="Apify actor id. Overrides APIFY_ACTOR_ID.")
    p.add_argument("--sort", default=None, choices=["relevance", "date_posted"],
                   help="Sort order for the actor.")
    return p.parse_args()


# --- Apify wrapper ---------------------------------------------------------
def run_actor(client: ApifyClient, actor_id: str, run_input: dict) -> Iterable[dict]:
    """Run an actor synchronously and yield each dataset item.

    apify-client >= 3 returns a pydantic `Run` model (attribute access /
    snake_case). Older clients returned a plain dict (`.get` / camelCase).
    """
    print(f"  -> starting actor '{actor_id}' with {run_input}")
    run = client.actor(actor_id).call(run_input=run_input)
    if not run:
        print("     actor returned no run info; skipping.", file=sys.stderr)
        return

    if isinstance(run, dict):
        dataset_id = run.get("defaultDatasetId") or run.get("default_dataset_id")
    else:
        dataset_id = getattr(run, "default_dataset_id", None) or getattr(
            run, "defaultDatasetId", None
        )

    if not dataset_id:
        print("     no dataset id returned; skipping.", file=sys.stderr)
        return
    print(f"     run succeeded; reading dataset {dataset_id}")
    yield from client.dataset(dataset_id).iterate_items()


def build_input(keyword: str, per_keyword: int, sort_type: str) -> dict:
    """Input for harvestapi/linkedin-post-search.

    Docs: https://apify.com/harvestapi/linkedin-post-search
    sortBy accepts 'relevance' or 'date' (newest first).
    """
    sort_by = "date" if sort_type in ("date_posted", "date") else "relevance"
    return {
        "searchQueries": [keyword],
        "maxPosts": per_keyword,
        "sortBy": sort_by,
        # Keep reactions/comments off so we only pay for posts.
        "scrapeReactions": False,
        "scrapeComments": False,
    }


# --- Normalisation ---------------------------------------------------------
# Flat fields (legacy actors) + nested harvestapi shapes.
POST_TEXT_FIELDS = ("text", "postText", "content", "description", "post_content")
POST_URL_FIELDS = ("linkedinUrl", "url", "postUrl", "post_url", "link")


def _pick(item: dict, keys: tuple[str, ...]) -> Any:
    for k in keys:
        v = item.get(k)
        if v not in (None, "", []):
            return v
    return None


def _author_field(item: dict, *keys: str) -> Any:
    author = item.get("author")
    if isinstance(author, dict):
        for k in keys:
            v = author.get(k)
            if v not in (None, "", []):
                return v
    elif isinstance(author, str) and "name" in keys:
        return author
    return None


def _posted_at(item: dict) -> Any:
    posted = item.get("postedAt")
    if isinstance(posted, dict):
        return posted.get("date") or posted.get("timestamp") or posted.get("postedAgoText")
    return _pick(item, ("postedAt", "postedDate", "date", "publishedAt", "time"))


def _engagement(item: dict, key: str) -> Any:
    eng = item.get("engagement")
    if isinstance(eng, dict) and eng.get(key) not in (None, ""):
        return eng.get(key)
    aliases = {
        "likes": ("likes", "likeCount", "numLikes", "totalReactions"),
        "comments": ("comments", "commentCount", "numComments"),
        "shares": ("shares", "shareCount", "reposts", "repostCount"),
    }
    return _pick(item, aliases.get(key, (key,)))


def normalise(item: dict, keyword: str) -> dict | None:
    # Skip reaction/comment rows if scrapeReactions/scrapeComments were enabled.
    if item.get("type") in ("reaction", "comment"):
        return None

    text = _pick(item, POST_TEXT_FIELDS)
    if not text or not isinstance(text, str):
        return None

    social = item.get("socialContent") if isinstance(item.get("socialContent"), dict) else {}
    post_url = (
        _pick(item, POST_URL_FIELDS)
        or social.get("shareUrl")
    )

    return {
        "post_url": post_url,
        "author_name": (
            _author_field(item, "name")
            or _pick(item, ("authorName", "author_name", "actorName"))
        ),
        "author_url": (
            _author_field(item, "linkedinUrl")
            or _pick(item, ("authorUrl", "author_url", "profileUrl", "actorUrl"))
        ),
        "author_headline": (
            _author_field(item, "info", "headline", "position")
            or _pick(item, ("authorHeadline", "headline", "authorTitle", "authorDescription"))
        ),
        "posted_at": _posted_at(item),
        "likes": _engagement(item, "likes"),
        "comments": _engagement(item, "comments"),
        "shares": _engagement(item, "shares"),
        "keyword": keyword,
        "text": text.strip(),
        "char_count": len(text.strip()),
        "word_count": len(text.split()),
    }


# --- Main ------------------------------------------------------------------
def main() -> int:
    args = parse_args()

    token = config.APIFY_API_TOKEN
    if not token:
        print("ERROR: APIFY_API_TOKEN missing. Copy .env.example -> .env and set it.",
              file=sys.stderr)
        return 2

    actor_id = args.actor or config.APIFY_ACTOR_ID
    total = args.total or config.TOTAL_POSTS
    sort_type = args.sort or config.SORT_TYPE
    keywords = (
        [k.strip() for k in args.keywords.split(",") if k.strip()]
        if args.keywords else list(config.SEARCH_KEYWORDS)
    )
    if not keywords:
        print("ERROR: no keywords provided.", file=sys.stderr)
        return 2

    per_keyword = max(1, total // len(keywords))
    print(f"Scraping ~{total} posts across {len(keywords)} keyword(s) "
          f"(~{per_keyword} each) via '{actor_id}'.")

    client = ApifyClient(token)

    raw_path = config.RAW_POSTS_JSONL
    raw_path.parent.mkdir(parents=True, exist_ok=True)

    collected: list[dict] = []
    seen_urls: set[str] = set()

    with raw_path.open("w", encoding="utf-8") as raw_f:
        for kw in keywords:
            print(f"[keyword] {kw!r}")
            got_here = 0
            try:
                for item in run_actor(client, actor_id,
                                      build_input(kw, per_keyword, sort_type)):
                    raw_f.write(json.dumps(item, ensure_ascii=False, default=str))
                    raw_f.write("\n")
                    row = normalise(item, kw)
                    if row is None:
                        continue
                    url = row.get("post_url")
                    if url and url in seen_urls:
                        continue
                    if url:
                        seen_urls.add(url)
                    collected.append(row)
                    got_here += 1
                    if got_here >= per_keyword:
                        break
            except Exception as e:  # keep partial results
                print(f"  !! actor call failed for {kw!r}: {e}", file=sys.stderr)
            print(f"  <- collected {got_here} usable posts for {kw!r} "
                  f"(running total: {len(collected)})")
            if len(collected) >= total:
                break

    if not collected:
        print("ERROR: no posts collected. Check your actor id and token.",
              file=sys.stderr)
        return 1

    df = pd.DataFrame(collected).head(total)
    df.to_csv(config.POSTS_CSV, index=False, encoding="utf-8")
    print(f"\nWrote {len(df)} posts -> {config.POSTS_CSV}")
    print(f"Raw dump preserved  -> {raw_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
