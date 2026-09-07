"""
Step 2 of the pipeline: genuine web/social-media discovery via a live
reverse-image search.

Primary path (automated):
    Local image -> SerpApi Image API (upload) -> image_id
    -> SerpApi Google Lens engine -> structured visual_matches
    -> operator picks the real candidate that looks correct.

Fallback path (manual, human-in-the-loop, used only if the primary
path is unavailable/unhelpful):
    Script opens Google Lens or Bing Visual Search in a normal browser
    (webbrowser.open only -- no automation, no scraping) and the
    operator pastes back the real source-page URL they found.

This module performs DISCOVERY only. It does not download the
candidate image, does not compare faces, and does not claim the
candidate is a confirmed match -- that is Step 3's job.

Nothing in this module ever prints or saves the SerpApi API key.
"""

import os
import sys
import json
import webbrowser
from datetime import datetime, timezone
from urllib.parse import urlparse

import requests
from dotenv import load_dotenv

SERPAPI_IMAGE_UPLOAD_URL = "https://serpapi.com/image"
SERPAPI_SEARCH_URL = "https://serpapi.com/search"
GOOGLE_LENS_MANUAL_URL = "https://lens.google.com/"
BING_VISUAL_SEARCH_MANUAL_URL = "https://www.bing.com/visualsearch"

DEFAULT_OUTPUT_PATH = os.path.join("data", "output", "search_result.json")

# SerpApi's documented Image API upload limit.
MAX_UPLOAD_BYTES = 500 * 1024

# Domains that are search-engine result pages, not real source pages.
REJECTED_SOURCE_DOMAINS = {
    "tineye.com",
    "www.tineye.com",
    "google.com",
    "www.google.com",
    "lens.google.com",
    "images.google.com",
    "bing.com",
    "www.bing.com",
}


class SearchStageError(Exception):
    """Base exception for all search_stage errors."""


class ImageNotFoundError(SearchStageError):
    """Raised when the given image path does not exist."""


class ImageLoadError(SearchStageError):
    """Raised when the image exists but is empty or too large to upload."""


class InvalidURLError(SearchStageError):
    """Raised when an operator-supplied source URL fails validation."""


class SerpApiRequestError(SearchStageError):
    """Raised when the SerpApi upload/search call fails for any reason."""


# --------------------------------------------------------------------------
# Validation helpers
# --------------------------------------------------------------------------

def validate_image_path(image_path):
    """Confirm the query image exists, is non-empty, and fits SerpApi's
    500KB upload limit (checked locally so a bad image never wastes a
    search request)."""
    if not os.path.isfile(image_path):
        raise ImageNotFoundError(f"Image not found: {image_path}")

    size = os.path.getsize(image_path)
    if size == 0:
        raise ImageLoadError(f"Image file is empty: {image_path}")
    if size > MAX_UPLOAD_BYTES:
        raise ImageLoadError(
            f"Image is {size // 1024} KB, which exceeds SerpApi's "
            f"{MAX_UPLOAD_BYTES // 1024} KB upload limit. "
            "Resize/compress the image and try again."
        )
    return image_path


def validate_source_url(url):
    """
    Minimal, honest URL validation:
        - must be non-empty
        - must use http:// or https://
        - must have a domain
        - must not be a known search-engine domain (we need the SOURCE
          page, not a Google/Bing/TinEye results page)
    """
    if url is None:
        raise InvalidURLError("URL cannot be empty.")

    url = url.strip()
    if not url:
        raise InvalidURLError("URL cannot be empty.")

    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise InvalidURLError("URL must start with http:// or https://")
    if not parsed.netloc:
        raise InvalidURLError("URL is malformed (missing domain).")
    if parsed.netloc.lower() in REJECTED_SOURCE_DOMAINS:
        raise InvalidURLError(
            "That looks like a search-engine URL (Google/Bing/TinEye), "
            "not a source page. Paste the URL of the actual page where "
            "the match was found."
        )

    return url


def load_api_key():
    """Load SERPAPI_KEY from the environment / .env file. Returns None
    (never raises) if it isn't set -- the caller decides how to react."""
    load_dotenv()
    key = os.environ.get("SERPAPI_KEY")
    if key is None:
        return None
    key = key.strip()
    return key or None


# --------------------------------------------------------------------------
# SerpApi calls (network errors are never allowed to leak the API key)
# --------------------------------------------------------------------------

def upload_image_to_serpapi(image_path, api_key, timeout=30):
    """Upload a local image to SerpApi's Image API and return its image_id."""
    try:
        with open(image_path, "rb") as image_file:
            files = {"image": image_file}
            data = {"api_key": api_key}
            response = requests.post(
                SERPAPI_IMAGE_UPLOAD_URL, files=files, data=data, timeout=timeout
            )
    except requests.exceptions.RequestException as exc:
        # Deliberately do NOT include str(exc): connection-error messages
        # from requests can embed the full request URL/params, which would
        # leak the API key into terminal output or logs.
        raise SerpApiRequestError(
            f"Network error while uploading image to SerpApi ({type(exc).__name__})."
        ) from exc

    if response.status_code != 200:
        raise SerpApiRequestError(
            f"SerpApi image upload failed (HTTP {response.status_code})."
        )

    try:
        payload = response.json()
    except ValueError as exc:
        raise SerpApiRequestError(
            "SerpApi image upload returned an unreadable response."
        ) from exc

    image_id = payload.get("image_id") if isinstance(payload, dict) else None
    if not image_id:
        raise SerpApiRequestError("SerpApi image upload did not return an image_id.")

    return image_id


def run_google_lens_search(image_id, api_key, timeout=30):
    """Run exactly one Google Lens search via SerpApi and return the raw
    JSON response."""
    params = {"engine": "google_lens", "image_id": image_id, "api_key": api_key}
    try:
        response = requests.get(SERPAPI_SEARCH_URL, params=params, timeout=timeout)
    except requests.exceptions.RequestException as exc:
        raise SerpApiRequestError(
            f"Network error while querying SerpApi Google Lens ({type(exc).__name__})."
        ) from exc

    if response.status_code != 200:
        raise SerpApiRequestError(
            f"SerpApi search failed (HTTP {response.status_code})."
        )

    try:
        payload = response.json()
    except ValueError as exc:
        raise SerpApiRequestError(
            "SerpApi search returned an unreadable response."
        ) from exc

    if isinstance(payload, dict) and payload.get("error"):
        raise SerpApiRequestError(f"SerpApi returned an error: {payload['error']}")

    return payload


# --------------------------------------------------------------------------
# Response parsing (tolerant of missing/unexpected fields)
# --------------------------------------------------------------------------

def _domain_from_url(url):
    try:
        return urlparse(url).netloc or "Unknown"
    except (ValueError, TypeError):
        return "Unknown"


def parse_candidates(raw_response):
    """
    Safely extract candidate source pages from a SerpApi Google Lens
    response. Only entries with a usable http(s) link are kept -- pure
    image-thumbnail entries with no source page are skipped. Tolerates
    a missing/malformed response entirely (returns []).
    """
    if not isinstance(raw_response, dict):
        return []

    visual_matches = raw_response.get("visual_matches")
    if not isinstance(visual_matches, list):
        return []

    candidates = []
    for item in visual_matches:
        if not isinstance(item, dict):
            continue

        url = item.get("link")
        if not isinstance(url, str) or not url.startswith(("http://", "https://")):
            continue

        title = item.get("title")
        if not isinstance(title, str) or not title:
            title = "Untitled"

        source = item.get("source")
        if not isinstance(source, str) or not source:
            source = _domain_from_url(url)

        candidate = {"title": title, "source": source, "url": url}

        snippet = item.get("snippet")
        if isinstance(snippet, str) and snippet:
            candidate["snippet"] = snippet

        candidates.append(candidate)

    return candidates[:10]


# --------------------------------------------------------------------------
# Terminal output
# --------------------------------------------------------------------------

def _print_banner(image_path):
    print("=" * 40)
    print("STEP 2: VISUAL WEB SEARCH")
    print("=" * 40)
    print()
    print("Search engine:")
    print("SerpApi -> Google Lens")
    print()
    print("Input:")
    print(image_path)
    print()
    print("One SerpApi search will be performed for this run.")
    print()
    print("Searching the web...")
    print()


def _print_candidates(candidates):
    print("Search completed.")
    print()
    print(f"Candidate results found: {len(candidates)}")
    print()
    for i, candidate in enumerate(candidates, start=1):
        print(f"[{i}]")
        print(f"Title: {candidate.get('title', 'Untitled')}")
        print(f"Source: {candidate.get('source', 'Unknown')}")
        print(f"URL: {candidate.get('url', '')}")
        print()


def _print_no_candidates():
    print("Search completed, but no suitable candidate webpage was found.")
    print()
    print("SerpApi Google Lens did not produce a usable candidate.")
    print()


def _print_missing_key_help():
    print("SERPAPI_KEY is not set.")
    print()
    print("To configure it:")
    print("1. Copy .env.example to .env")
    print("2. Open .env and set SERPAPI_KEY=<your real key>")
    print("3. Get a free key at https://serpapi.com/ (no card required)")
    print()
    print("SerpApi Google Lens did not produce a usable candidate.")
    print()


def _print_api_error(message):
    print(f"SerpApi request failed: {message}")
    print()
    print("SerpApi Google Lens did not produce a usable candidate.")
    print()


def _print_fallback_menu():
    print("Fallback options:")
    print("1. Open Google Lens manually")
    print("2. Open Bing Visual Search manually")
    print("3. Exit")
    print()


def _print_manual_instructions(engine_name):
    print("=" * 40)
    print(f"MANUAL FALLBACK: {engine_name}")
    print("=" * 40)
    print()
    print(f"{engine_name} is being opened in your browser.")
    print()
    print("MANUAL ACTION REQUIRED:")
    print("1. Upload the same input image.")
    print("2. Perform the visual/reverse-image search.")
    print("3. Look through the results.")
    print("4. Select a useful result that contains the image/post.")
    print("5. Open the SOURCE PAGE (not the search-results page).")
    print("6. Copy the SOURCE PAGE URL.")
    print("7. Paste the URL below.")
    print()


def _print_discovered_pending(result, output_path):
    print()
    print("=" * 40)
    print("Candidate discovered - face verification pending.")
    print(f"Search engine: {result['search_engine']}")
    print(f"URL: {result['selected_candidate']['url']}")
    print(f"Result saved to: {output_path}")
    print("Status: SUCCESS")
    print("=" * 40)


# --------------------------------------------------------------------------
# Candidate selection (SerpApi path)
# --------------------------------------------------------------------------

def select_candidate_interactive(candidates, prompt=input):
    """
    Ask the operator to pick a candidate by number, then confirm it.
    Returns the chosen candidate dict, or None if the operator enters 0
    ("none of these are useful") -- the caller then falls back to the
    manual search menu.
    """
    print("=" * 40)
    print("SELECT A CANDIDATE")
    print("=" * 40)
    print()

    while True:
        raw = prompt("Enter candidate number (or 0 for none of these):\n> ")
        raw = (raw or "").strip()

        try:
            idx = int(raw)
        except ValueError:
            print("Invalid input: please enter a number.")
            continue

        if idx == 0:
            return None

        if idx < 1 or idx > len(candidates):
            print(f"Invalid selection: choose a number between 0 and {len(candidates)}.")
            continue

        chosen = candidates[idx - 1]
        print()
        print("Selected candidate:")
        print(f"Title: {chosen.get('title', 'Untitled')}")
        print(f"Source: {chosen.get('source', 'Unknown')}")
        print(f"URL: {chosen.get('url', '')}")
        print()

        confirm = prompt(
            "Is this the source page containing the matching image/content? (y/n):\n> "
        )
        if (confirm or "").strip().lower() in ("y", "yes"):
            return chosen
        print()


# --------------------------------------------------------------------------
# Manual fallback (human-in-the-loop, no automation)
# --------------------------------------------------------------------------

def manual_google_lens_fallback(browser_open=webbrowser.open, prompt=input):
    _print_manual_instructions("Google Lens")
    browser_open(GOOGLE_LENS_MANUAL_URL)
    raw_url = prompt("Source URL:\n> ")
    url = validate_source_url(raw_url)
    return {"title": None, "source": None, "url": url}


def manual_bing_fallback(browser_open=webbrowser.open, prompt=input):
    _print_manual_instructions("Bing Visual Search")
    browser_open(BING_VISUAL_SEARCH_MANUAL_URL)
    raw_url = prompt("Source URL:\n> ")
    url = validate_source_url(raw_url)
    return {"title": None, "source": None, "url": url}


def run_fallback_menu(browser_open=webbrowser.open, prompt=input):
    """
    Returns (search_engine, search_method, candidate_dict), or
    (None, None, None) if the operator chooses to exit.
    """
    _print_fallback_menu()
    while True:
        choice = (prompt("Enter choice (1/2/3):\n> ") or "").strip()

        if choice == "1":
            candidate = manual_google_lens_fallback(browser_open, prompt)
            return "google_lens_manual", "human_in_the_loop", candidate
        if choice == "2":
            candidate = manual_bing_fallback(browser_open, prompt)
            return "bing_visual_search_manual", "human_in_the_loop", candidate
        if choice == "3":
            return None, None, None

        print("Invalid choice. Enter 1, 2, or 3.")


# --------------------------------------------------------------------------
# Persistence
# --------------------------------------------------------------------------

def build_result(query_image, search_engine, search_method, selected_candidate, all_candidates):
    return {
        "query_image": query_image,
        "search_engine": search_engine,
        "search_method": search_method,
        "selected_candidate": selected_candidate,
        "all_candidates": all_candidates,
        "search_completed": True,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }


def save_result(result, output_path=DEFAULT_OUTPUT_PATH):
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    return output_path


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------

def search_for_match(
    image_path,
    api_key=None,
    browser_open=webbrowser.open,
    prompt=input,
    output_path=DEFAULT_OUTPUT_PATH,
):
    """
    Runs Step 2: discover a candidate source page for image_path.

    api_key / browser_open / prompt are injectable so automated tests
    can run fully offline (default to the real env lookup / webbrowser
    / input()).

    Returns the saved result dict, or None if the operator exits the
    fallback menu without providing a candidate.
    """
    validate_image_path(image_path)
    _print_banner(image_path)

    key = api_key if api_key is not None else load_api_key()

    candidates = []
    needs_fallback = False

    if not key:
        _print_missing_key_help()
        needs_fallback = True
    else:
        try:
            image_id = upload_image_to_serpapi(image_path, key)
            raw_response = run_google_lens_search(image_id, key)
            candidates = parse_candidates(raw_response)
        except SerpApiRequestError as exc:
            _print_api_error(str(exc))
            needs_fallback = True

    selected = None
    search_engine = "serpapi_google_lens"
    search_method = "api"

    if not needs_fallback:
        if not candidates:
            _print_no_candidates()
            needs_fallback = True
        else:
            _print_candidates(candidates)
            selected = select_candidate_interactive(candidates, prompt)
            if selected is None:
                needs_fallback = True

    if needs_fallback:
        print()
        print("=" * 40)
        print("AUTOMATED SEARCH FAILED")
        print("Browser fallback is disabled.")
        print("No browser will be opened.")
        print("Status: FAILED")
        print("=" * 40)
        return None

    result = build_result(
        image_path,
        search_engine,
        search_method,
        selected,
        candidates
    )

    save_result(result, output_path)
    _print_discovered_pending(result, output_path)

    return result


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv

    if len(argv) != 1:
        print("Usage: python -m src.search_stage <image_path>")
        return 2

    image_path = argv[0]

    try:
        result = search_for_match(image_path)
    except (ImageNotFoundError, ImageLoadError, InvalidURLError) as exc:
        print("=" * 40)
        print(f"ERROR: {exc}")
        print("Status: FAILED")
        print("=" * 40)
        return 1

    return 0 if result is not None else 1


if __name__ == "__main__":
    sys.exit(main())
