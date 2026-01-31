import os
import csv
import time
import argparse
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional

import requests
from dotenv import load_dotenv


# =========================
# CONFIG
# =========================

load_dotenv()

API_BASE_URL = "https://api.twitter.com/2"
BEARER_TOKEN = os.getenv("X_BEARER_TOKEN")
DEFAULT_MAX_RESULTS = 100
DEFAULT_LIMIT = 3200
DEFAULT_SLEEP = 1.0


class XApiError(Exception):
    pass


# =========================
# HELPERS
# =========================

def make_headers() -> Dict[str, str]:
    if not BEARER_TOKEN:
        raise RuntimeError("Missing X_BEARER_TOKEN. Put it in .env (X_BEARER_TOKEN=...).")
    return {
        "Authorization": f"Bearer {BEARER_TOKEN}",
        "User-Agent": "x-single-user-collector/1.0",
    }


def to_rfc3339(dt: datetime) -> str:
    """RFC3339 UTC with seconds."""
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def request_with_backoff(
    method: str,
    url: str,
    *,
    headers: Dict[str, str],
    params: Optional[Dict] = None,
    timeout: int = 30,
    max_retries: int = 8,
) -> requests.Response:
    """
    Retry on 429 and 5xx with exponential backoff up to ~15 min.
    """
    sleep_s = 60
    for attempt in range(1, max_retries + 1):
        resp = requests.request(method, url, headers=headers, params=params, timeout=timeout)

        if resp.status_code == 200:
            return resp

        if resp.status_code == 429 or (500 <= resp.status_code <= 599):
            print(f"[{resp.status_code}] backoff {sleep_s}s (attempt {attempt}/{max_retries})")
            time.sleep(sleep_s)
            sleep_s = min(sleep_s * 2, 15 * 60)
            continue

        # non-retryable
        return resp

    return resp


def get_user_id_by_username(username: str) -> str:
    username = username.strip().lstrip("@")
    url = f"{API_BASE_URL}/users/by/username/{username}"
    resp = request_with_backoff("GET", url, headers=make_headers())

    if resp.status_code != 200:
        raise XApiError(f"Username lookup failed for '{username}': {resp.status_code}: {resp.text[:300]}")

    j = resp.json()
    try:
        return j["data"]["id"]
    except Exception:
        raise XApiError(f"Unexpected response for '{username}': {j}")


def fetch_user_tweets(
    user_id: str,
    start_time: str,
    end_time: Optional[str],
    *,
    max_results: int = DEFAULT_MAX_RESULTS,
    limit: Optional[int] = DEFAULT_LIMIT,
    sleep_seconds: float = DEFAULT_SLEEP,
) -> List[Dict]:
    """
    Fetch tweets for a user ID, between start_time and end_time.
    """
    url = f"{API_BASE_URL}/users/{user_id}/tweets"

    params: Dict[str, str] = {
        "max_results": str(max_results),
        "start_time": start_time,
        "tweet.fields": "id,text,created_at,lang,public_metrics",
    }
    if end_time:
        params["end_time"] = end_time

    all_tweets: List[Dict] = []
    next_token: Optional[str] = None

    while True:
        if next_token:
            params["pagination_token"] = next_token
        else:
            params.pop("pagination_token", None)

        resp = request_with_backoff("GET", url, headers=make_headers(), params=params)

        if resp.status_code != 200:
            raise XApiError(f"Fetch failed for user_id={user_id}: {resp.status_code}: {resp.text[:300]}")

        data = resp.json()
        tweets = data.get("data", []) or []
        meta = data.get("meta", {}) or {}
        next_token = meta.get("next_token")

        if not tweets:
            break

        all_tweets.extend(tweets)

        if limit is not None and len(all_tweets) >= limit:
            all_tweets = all_tweets[:limit]
            break

        if not next_token:
            break

        time.sleep(sleep_seconds)

    return all_tweets


def save_tweets_to_csv(username: str, user_id: str, tweets: List[Dict], out_path: str) -> None:
    fieldnames = [
        "x_handle",
        "x_id",
        "tweet_id",
        "created_at",
        "lang",
        "retweet_count",
        "reply_count",
        "like_count",
        "quote_count",
        "text",
    ]

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()

        for t in tweets:
            m = t.get("public_metrics", {}) or {}
            w.writerow({
                "x_handle": username,
                "x_id": user_id,
                "tweet_id": t.get("id"),
                "created_at": t.get("created_at"),
                "lang": t.get("lang"),
                "retweet_count": m.get("retweet_count"),
                "reply_count": m.get("reply_count"),
                "like_count": m.get("like_count"),
                "quote_count": m.get("quote_count"),
                "text": (t.get("text") or "").replace("\n", " "),
            })


def build_output_name(username: str, run_date: datetime, out_dir: str) -> str:
    safe_user = username.strip().lstrip("@")
    day = run_date.astimezone(timezone.utc).strftime("%Y-%m-%d")
    return os.path.join(out_dir, f"X_posts_{safe_user}_{day}.csv")


# =========================
# CLI
# =========================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Collect X posts for a username in a time window and save to CSV."
    )
    p.add_argument("--username", required=True, help="X username, e.g. storaenso or @storaenso")

    group = p.add_mutually_exclusive_group(required=False)
    group.add_argument("--days", type=int, default=182, help="Collect last N days (default: 182 ~ 6 months)")
    # If you prefer explicit window:
    p.add_argument("--start", type=str, default=None, help="RFC3339 e.g. 2025-01-01T00:00:00Z")
    p.add_argument("--end", type=str, default=None, help="RFC3339 e.g. 2026-01-31T00:00:00Z (default: now)")

    p.add_argument("--limit", type=int, default=DEFAULT_LIMIT, help="Max tweets to collect (default 3200)")
    p.add_argument("--max-results", type=int, default=DEFAULT_MAX_RESULTS, help="Per page (5..100), default 100")
    p.add_argument("--sleep", type=float, default=DEFAULT_SLEEP, help="Sleep between pages (default 1.0s)")
    p.add_argument("--out-dir", type=str, default="data", help="Output folder (default: data)")

    return p.parse_args()


def compute_window(args: argparse.Namespace) -> tuple[str, Optional[str]]:
    now = datetime.now(timezone.utc)

    # If start provided, use it
    if args.start:
        start_time = args.start
        end_time = args.end if args.end else to_rfc3339(now)
        return start_time, end_time

    # Else use last N days
    start_dt = now - timedelta(days=args.days)
    start_time = to_rfc3339(start_dt)
    end_time = to_rfc3339(now)
    return start_time, end_time


def main() -> None:
    args = parse_args()
    username = args.username.strip().lstrip("@")

    os.makedirs(args.out_dir, exist_ok=True)

    start_time, end_time = compute_window(args)
    print("start_time:", start_time)
    print("end_time:", end_time)

    user_id = get_user_id_by_username(username)
    print("resolved x_id:", user_id)

    tweets = fetch_user_tweets(
        user_id=user_id,
        start_time=start_time,
        end_time=end_time,
        max_results=args.max_results,
        limit=args.limit,
        sleep_seconds=args.sleep,
    )

    out_path = build_output_name(username, datetime.now(timezone.utc), args.out_dir)
    save_tweets_to_csv(username, user_id, tweets, out_path)

    print(f"Collected {len(tweets)} tweets → {out_path}")


if __name__ == "__main__":
    main()
