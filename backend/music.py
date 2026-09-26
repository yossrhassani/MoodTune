"""
Maps a detected mood to a Deezer search query and returns playable tracks.

Per the report (Section 3.6): mood-specific query phrases with random
offsets for variety across repeated scans, playlist fallback if a search
returns nothing, filtering to tracks with a preview URL, and de-duplication
by artist so results aren't dominated by a single act.
"""

from __future__ import annotations

import random

import requests

DEEZER_SEARCH_URL = "https://api.deezer.com/search"

# Mood -> list of query phrases (report: "mood-specific query phrases with
# random offsets"). Several phrases per mood keep repeated scans varied.
MOOD_QUERIES = {
    "calm": ["lofi calm focus", "ambient calm piano", "soft acoustic chill"],
    "energetic": ["upbeat energetic pop", "feel good dance", "high energy workout"],
    "low": ["healing acoustic hope", "gentle comfort songs", "soft sad piano"],
    "positive": ["feel good happy pop", "sunny upbeat songs", "positive vibes playlist"],
    "stressed": ["stress relief ambient", "calming instrumental", "deep breathing music"],
}

FALLBACK_QUERY = "chill instrumental playlist"


def _search_deezer(query: str, limit: int = 15) -> list[dict]:
    try:
        response = requests.get(
            DEEZER_SEARCH_URL,
            params={"q": query, "limit": limit},
            timeout=5,
        )
        response.raise_for_status()
        return response.json().get("data", [])
    except requests.RequestException:
        return []


def _dedupe_by_artist(tracks: list[dict], max_per_artist: int = 1) -> list[dict]:
    seen: dict[str, int] = {}
    result = []
    for track in tracks:
        artist = track.get("artist", {}).get("name", "unknown")
        seen[artist] = seen.get(artist, 0)
        if seen[artist] < max_per_artist:
            result.append(track)
            seen[artist] += 1
    return result


def get_recommendations(mood: str, limit: int = 5) -> dict:
    queries = MOOD_QUERIES.get(mood, [FALLBACK_QUERY])
    query = random.choice(queries)

    raw_tracks = _search_deezer(query)
    if not raw_tracks:
        query = FALLBACK_QUERY
        raw_tracks = _search_deezer(query)

    # Only tracks with a real preview URL are playable in-app.
    playable = [t for t in raw_tracks if t.get("preview")]
    deduped = _dedupe_by_artist(playable)

    tracks = [
        {
            "title": t.get("title"),
            "artist": t.get("artist", {}).get("name"),
            "preview_url": t.get("preview"),
            "cover": t.get("album", {}).get("cover_medium"),
        }
        for t in deduped[:limit]
    ]

    return {
        "mood": mood,
        "query": query,
        "strategy": "mood_query" if tracks else "fallback",
        "tracks": tracks,
    }
